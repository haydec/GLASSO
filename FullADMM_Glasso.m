clear
clc
% Define a known precision matrix (inverse covariance) with a known sparsity pattern
lowVariance = false;
if lowVariance
    true_precision = [ 1.0, -0.5,  0.0,  0.0,  0.0;
                      -0.5,  1.0, -0.4,  0.0,  0.0;
                       0.0, -0.4,  1.0, -0.3,  0.0;
                       0.0,  0.0, -0.3,  1.0, -0.2;
                       0.0,  0.0,  0.0, -0.2,  1.0];
else
    true_precision = [ 10.0, -1.8, -1.5, -1.2, -0.9;
                  -1.8,  10.0, -1.6, -1.3, -1.0;
                  -1.5, -1.6,  10.0, -1.4, -1.1;
                  -1.2, -1.3, -1.4,  10.0, -1.2;
                  -0.9, -1.0, -1.1, -1.2,  10.0];
end

% Ensure the matrix is symmetric and positive definite
true_precision = (true_precision + true_precision') / 2;
eigvals = eig(true_precision);
if any(eigvals <= 0)
    error('Precision matrix is not positive definite');
end

% Compute the true covariance matrix as the inverse of the precision matrix
true_covariance = inv(true_precision);

% Number of samples to generate
n_samples = 10000;

% Generate data from a multivariate normal distribution
rng(0);  % Set random seed for reproducibility
data = mvnrnd(zeros(1, 5), true_covariance, n_samples);

%plot(data)

% Define the sample size and reshape data into segments
% ============================================================================
sampleSize = 10000;  % Define sample size (number of time steps in each segment)
% ============================================================================
[numObservations, numVariables] = size(data);
numFullSamples = floor(numObservations / sampleSize);

% Extract only the data that fits into full samples
sampledData = data(1:numFullSamples * sampleSize, :);

% Reshape into samples with dimensions [sampleSize, numVariables, numFullSamples]
samples = reshape(sampledData, sampleSize, numVariables, numFullSamples);


% Calculate the empirical covariance matrices for each segment
S = zeros(numVariables, numVariables, numFullSamples);  % Initialize covariance matrix storage
for t = 1:numFullSamples
    S(:, :, t) = cov(squeeze(samples(:, :, t)));  % Calculate covariance for each segment
end


% Initialize ADMM parameters
rho = 1;                      % Initial rho value
tau_increase = 1.1;            % Factor to increase rho
tau_decrease = 1.1;            % Factor to decrease rho
max_rho = 10;                  % Maximum rho value
min_rho = 0.01;                % Minimum rho value


lambda = 0.1;
beta = 0;
penalty_type = 'l2';

max_iter = 100000;
tol = 1e-4;


% Parameters:
% S - Empirical covariance matrices for each time point (3D array)
% n - Number of observations
% rho - ADMM penalty parameter
% lambda - Regularization parameter for sparsity
% beta - Regularization parameter for temporal consistency
% penalty_type - Type of penalty ('l1', 'l2', 'laplacian', 'l_inf')
% max_iter - Maximum number of ADMM iterations
% tol - Convergence tolerance for residuals

% Initialize variables
[p, ~, T] = size(S);        % p: dimension of each matrix, T: number of time points
Z0 = zeros(p, p, T);        % Initialize Z0 (sparsity) 1 - T 
Z1 = zeros(p, p, T);        % Initialize Z1 (temporal consistency) 1 - T-1
Z2 = zeros(p, p, T);        % Initialize Z2 (temporal consistency) 2 - T

U0 = zeros(p, p, T);        % Initialize U0 (dual variable for sparsity) 1 - T 
U1 = zeros(p, p, T);        % Initialize U1 (dual variable for Z1) 1 - T-1
U2 = zeros(p, p, T);        % Initialize U2 (dual variable for Z2) 2 - T

% For Debugging 
primal_resArray = zeros(1,max_iter);
dual_resArray = zeros(1,max_iter);


% ADMM iteration loop
for k = 1:max_iter
    Z0_prev = Z0;
    Z1_prev = Z1;
    Z2_prev = Z2;
    % Step (a): Update Theta using the theta_update function
    Theta = theta_update(Z0, Z1, Z2, U0, U1, U2, S, sampleSize, rho);
    

    % Step (b): Update Z using the z_update function
    [Z0,Z1,Z2] = z_update(Theta, U0, U1, U2, lambda, beta, rho, penalty_type);



    % Step (c): Update dual variables U using the u_update function
    [U0, U1, U2] = u_update(U0, U1, U2, Theta, Z0, Z1, Z2);

    

    % Calculate Primal Residuals and and Dual Residuals
    primal_residual = 0;
    for t = 1:T
        primal_residual = primal_residual + norm(Theta(:,:,t) - Z0(:,:,t), 'fro')^2;
    end
    for t = 2:T
        primal_residual = primal_residual + norm(Theta(:,:,t-1) - Z1(:,:,t-1), 'fro')^2 + norm(Theta(:,:,t) - Z2(:,:,t), 'fro')^2;
    end
    primal_residual = sqrt(primal_residual);


    dual_residual = 0;
    for t = 1:T
        dual_residual = dual_residual + norm(rho * (Z0(:,:,t) - Z0_prev(:,:,t)), 'fro')^2;
    end
    for t = 2:T
        dual_residual = dual_residual + norm(rho * (Z1(:,:,t-1) - Z1_prev(:,:,t-1)), 'fro')^2 + norm(rho * (Z2(:,:,t) - Z2_prev(:,:,t)), 'fro')^2;
    end
    dual_residual = sqrt(dual_residual);

    
    % Adaptive adjustment of rho based on residuals
    if primal_residual > 10 * dual_residual
        % Increase rho if primal residual is much larger
        rho = min(rho * tau_increase, max_rho);
    elseif dual_residual > 10 * primal_residual
        % Decrease rho if dual residual is much larger
        rho = max(rho / tau_decrease, min_rho);
    end
    

    % Check convergence
    if primal_residual < tol && dual_residual < tol
        disp(['Converged at iteration ', num2str(k)]);
        break;
    end

    % Display progress every 10 iterations
    if mod(k, 10) == 0
        fprintf('Iteration %d: Primal Residual = %.6f, Dual Residual = %.6f\n', k, primal_residual, dual_residual);
    end
end

% If maximum iterations are reached
if k == max_iter
    disp('Reached maximum iterations without full convergence');
    fprintf('Iteration %d: Primal Residual = %.6f, Dual Residual = %.6f\n', k, primal_residual, dual_residual);
end
    disp(Theta)

% Non AdMM Method
%Theta0t = repmat(eye(size(S, 1)), 1, 1, numFullSamples);  % Initial guess as identity matrices
%Theta_optimal = fitTimeVaryingGraphicalLassoSmoothing(Theta0t, S, lambda, beta, numFullSamples,false);

%
function Theta = theta_update(Z0, Z1, Z2, U0, U1, U2, emp_cov_mat, n, rho)
    % Parameters:
    % Z0, Z1, Z2 - Consensus variables for each constraint
    % U0, U1, U2 - Dual variables for each constraint
    % emp_cov_mat - Empirical covariance matrices
    % n - Number of samples
    % rho - ADMM penalty parameter

    % Constants
    eta = n / (3 * rho);  % Adjusted penalty parameter for the proximal operator
    num_timestamps = size(emp_cov_mat, 3); % Number of time points (assuming emp_cov_mat is [p, p, T])

    % Initialize Theta
    Theta = zeros(size(emp_cov_mat, 1), size(emp_cov_mat, 2), num_timestamps);

    % Loop over each time point
    for i = 1:num_timestamps
        % Calculate A for the current time step
        A = (Z0(:,:,i) + Z1(:,:,i) + Z2(:,:,i) - U0(:,:,i) - U1(:,:,i) - U2(:,:,i)) / 3;
        Asym = (A + A') / 2;  % Ensure symmetry of A
        
        % Calculate M
        M = (1/eta) * Asym - emp_cov_mat(:,:,i);
        
        % Eigen decomposition
        [Q, D] = eig(M);  % Eigen decomposition of M
        d = diag(D);      % Extract eigenvalues
        
        % Compute the sqrt matrix component-wise
        sqrt_matrix = sqrt(d.^2 + ( 4/eta) * eye(size(d)));
        
        % Create the diagonal matrix used in the update
        diagonal = diag(d + sqrt_matrix);
        
        % Update Theta for this time step
        Theta(:,:,i) = real((eta/2) * ( Q * diagonal * Q') );
        
        % Check for complex numbers in Theta
        if any(imag(Theta(:,:,i)) ~= 0)
            error('Complex numbers detected in the matrix Theta at timestamp %d.', i);
        end
    end
end




function [Z0,Z1,Z2] = z_update(Theta, U0, U1, U2, lambda, beta, rho, penalty_type)
    % Parameters:
    % Theta - current values of Theta matrices for each time point
    % U - dual variables
    % lambda - regularization parameter for sparsity (Z_{i,0})
    % beta - regularization parameter for temporal consistency (Z_{i-1,1}, Z_{i,2})
    % rho - ADMM penalty parameter
    % penalty_type - type of penalty ('l1', 'l2', 'laplacian', or 'l_inf')


    % Initialize Z components
    [pr, pc, T] = size(Theta);
    Z0 = zeros(pr, pc, T);        % Z_{i,0} for each time step
    Z1 = zeros(pr, pc, T);      % Z_{i-1,1} for temporal consistency (T-1 entries)
    Z2 = zeros(pr, pc, T);      % Z_{i,2} for temporal consistency (T-1 entries)
    
    % PART 1  Z_{i,0} Update: Element-wise soft-thresholding for sparsity

        


    for i = 1:T
        A = Theta(:,:,i) + U0(:,:,i);
        Z0(:,:,i) = soft_threshold_odd(A, lambda, rho);
        Z0(:,:,i) = (Z0(:,:,i) + Z0(:,:,i)')/2;
    end

    % Part 2 (Z_{i-1,1}, Z_{i,2}) Update: Proximal operator for temporal consistency

    for i = 2:T
        
        Adiff = Theta(:,:,i) - Theta(:,:,i-1) + U2(:,:,i)  - U1(:,:,i-1) ; 
        
        switch penalty_type
            case 'l1'
                % Element-wise L1 penalty (soft-threshold each element)
                E = L1_element_wise(Adiff, beta, rho);

            case 'l2'
                % Group Lasso L2 Penalty (block-wise soft thresholding)
                E = L2_group_lasso(Adiff, beta, rho);

            otherwise
                error('Unknown penalty type');
        end

        A = Theta(:,:,i-1) + Theta(:,:,i)  + U2(:,:,i) + U1(:,:,i-1);
        Z1(:,:,i-1) = 0.5*(A - E);
        Z2(:,:,i)   = 0.5*(A + E);
        Z1(:,:,i-1) = (Z1(:,:,i-1) + Z1(:,:,i-1)')/2;
        Z2(:,:,i) = (Z2(:,:,i) + Z2(:,:,i)')/2;
     

    end

    
    function Z0 = soft_threshold_odd(A, lambda, rho)
        % Soft-thresholding function for off-diagonal elements based on the 
        % given proximal operator for the ℓ_od,1-norm.
        
        % Calculate the threshold parameter
        parameter = lambda / rho;
        dimension = size(A, 1);
        
        % Initialize Z as a copy of A for in-place modification
        Z0 = ones(dimension);
        Z0 = A;
        
        % Apply soft-thresholding to off-diagonal elements
        for ii = 1:dimension
            for jj = 1:dimension
                if ii ~= jj  % Only apply to off-diagonal elements
                    if abs(A(ii, jj)) <= parameter
                        Z0(ii, jj) = 0;  % Set to zero if within threshold
                    else
                        % Shrink the value by the threshold
                        Z0(ii, jj) = sign(A(ii, jj)) * (abs(A(ii, jj)) - parameter);
                    end
                end
            end
        end
    end


    
    function E = L1_element_wise(A, beta, rho)
        % The element-wise l1 penalty function.
        % Used in (Z1, Z2) update
    
        eta = 2 * beta / rho;
        dimension = size(A, 1);
        E = zeros(dimension, dimension);
    
        % Apply element-wise thresholding
        for ii = 1:dimension
            for jj = 1:dimension
                if abs(A(ii, jj)) > eta
                    E(ii, jj) = sign(A(ii, jj)) * (abs(A(ii, jj)) - eta);
                end
            end
        end
    end
    
    function E = L2_group_lasso(A, beta, rho)
        % The Group Lasso l2 penalty function.
        % Used in (Z1, Z2) update
    
        eta = 2 * beta / rho;
        dimension = size(A, 1);
        E = zeros(dimension, dimension);
    
        % Apply column-wise l2 norm soft-thresholding
        for j = 1:dimension
            l2_norm = norm(A(:, j));
            if l2_norm > eta
                E(:, j) = (1 - eta / l2_norm) * A(:, j);
            end
        end
    end
    

end


function [U0, U1, U2] = u_update(U0, U1, U2, thetas, Z0, Z1, Z2)
    % Updates dual variables U0, U1, and U2 based on the difference between thetas and Z matrices

    num_blocks = size(thetas, 3);  % Number of blocks

    % Update U0 for each block
    for i = 1:num_blocks
        U0(:,:,i) = U0(:,:,i) + (thetas(:,:,i) - Z0(:,:,i));
    end

    % Update U1 and U2 for each block, starting from the second
    for i = 2:num_blocks
        U1(:,:,i-1) = U1(:,:,i-1) + ( thetas(:,:,i-1) - Z1(:,:,i-1) );
        U2(:,:,i) = U2(:,:,i) + ( thetas(:,:,i) - Z2(:,:,i) );        
    end
end
