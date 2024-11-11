clc
clear
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
n_samples = 1000000;

% Generate data from a multivariate normal distribution
rng(0);  % Set random seed for reproducibility
data = mvnrnd(zeros(1, 5), true_covariance, n_samples);

%plot(data)

% Define the sample size and reshape data into segments
% ============================================================================
sampleSize = 100000;  % Define sample size (number of time steps in each segment)
% ============================================================================
[numObservations, numVariables] = size(data);
numFullSamples = floor(numObservations / sampleSize);

% Extract only the data that fits into full samples
sampledData = data(1:numFullSamples * sampleSize, :);

% Reshape into samples with dimensions [sampleSize, numVariables, numFullSamples]
samples = reshape(sampledData, sampleSize, numVariables, numFullSamples);


rho = 1;
lambda = 0.1; % Sparsity Penalty 
beta = 12.5; % Time Varying Penalty


alpha = 1; % alpha is the over-relaxation parameter

t_start = tic;
%Global constants and defaults
QUIET    = 0;
MAX_ITER = 100;
ABSTOL   = 1e-6;
RELTOL   = 1e-4;

S = zeros(size(samples,2),size(samples,2),size(samples,3));
%Data preprocessing
for t = 1:1:size(samples,3)
    S(:,:,t) = cov(samples(:,:,t));
end
n = size(S,1);
T = size(S,3);

%ADMM solver
Theta_hat = zeros(size(S,1),size(S,2),size(S,3));
Z0 = zeros(size(S,1),size(S,2),size(S,3));
Z1 = zeros(size(S,1),size(S,2),size(S,3));
Z2 = zeros(size(S,1),size(S,2),size(S,3));

U0 = zeros(size(S,1),size(S,2),size(S,3));
U1 = zeros(size(S,1),size(S,2),size(S,3));
U2 = zeros(size(S,1),size(S,2),size(S,3));


eta = numFullSamples / (3 * rho);  % Adjusted penalty parameter for the proximal operator

Variables = 1 + (lambda > 0) + (beta > 0);
disp(Variables)
for k = 1:MAX_ITER
    
    for t = 1:1:T

        % Theta Update
        A = ( ( Z0(:,:,t) + Z1(:,:,t) +  Z2(:,:,t)) - (U0(:,:,t) + U1(:,:,t) +  U2(:,:,t)) )/Variables;
        Asym = (A + A')/2;
        disp(Asym)
        M = (1/eta)*(Asym) - S(:,:,t); % (rho)*(Asym) - S;
        [Q,L] = eig(M);
        es = diag(L);
        xi = (es + sqrt( es.^2 + 4*(1/eta) )) ./( 2*( 1/eta )); % (es + sqrt(es.^2 + 4*rho))./(2*rho);
        Theta =  Q*diag(xi)*Q';


        % z-update with relaxation
        Z0old = Z0(:,:,t);
        Theta_hat(:,:,t) = alpha*Theta + (1 - alpha)*Z0old;
        
        Z0(:,:,t) = soft_threshold_odd(Theta_hat(:,:,t) + U0(:,:,t), lambda,rho);
        
        if t > 1 & beta > 0

            Adiff = Theta_hat(:,:,t) - Theta_hat(:,:,t-1) + U2(:,:,t)  - U1(:,:,t-1) ; 
            
            E = L1_element_wise(Adiff, beta, rho);
            
            Asum = Theta_hat(:,:,t-1) + Theta_hat(:,:,t)  + U2(:,:,t) + U1(:,:,t-1);
            Z1(:,:,t-1) = 0.5*(Asum - E);
            Z2(:,:,t)   = 0.5*(Asum + E);

        end

        
        % u-update (Dual Variable) 
        U0(:,:,t) = U0(:,:,t) + ( Theta_hat(:,:,t) - Z0(:,:,t) );
        if t > 1 & beta > 0
            U1(:,:,t-1) = U1(:,:,t-1) + ( Theta_hat(:,:,t-1) - Z1(:,:,t-1) );
            U2(:,:,t) = U2(:,:,t) + ( Theta_hat(:,:,t) - Z2(:,:,t) );
        end


    end



    % diagnostics, reporting, termination checks
    %history.objval(k)  = objective(S, Theta, Z0,Z1,Z2, lambda,beta,T,type);

    %history.r_norm(k)  = norm(X - Z, 'fro');
    %history.s_norm(k)  = norm(-rho*(Z - Zold),'fro');

    %history.eps_pri(k) = sqrt(n*n)*ABSTOL + RELTOL*max(norm(X,'fro'), norm(Z,'fro'));
    %history.eps_dual(k)= sqrt(n*n)*ABSTOL + RELTOL*norm(rho*U,'fro');

    %{
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    %}


end

disp(Theta_hat)


function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end

function e = soft_threshold_odd(a, lambda, rho)
    % soft_threshold_odd - Off-diagonal soft-thresholding function for Lasso penalty
    % Computes the Z0 update with the off-diagonal soft-threshold operator
    
    % Calculate the threshold parameter
    parameter = lambda / rho;
    
    % Get the dimension of the matrix
    dimension = size(a, 1);
    
    % Initialize the result matrix as an identity matrix
    e = a;
    
    % Apply the soft-thresholding to off-diagonal elements
    for i = 1:dimension
        for j = i + 1:dimension
            if abs(a(i, j)) > parameter
                result = sign(a(i, j)) * (abs(a(i, j)) - parameter);
                e(i, j) = result;
                e(j, i) = result;
            end
        end
    end
end


function obj = objective(S, Theta, Z0, Z1, Z2, lambda,beta,T,type)

    obj1 = zeros(1,T);
    for t = 1:1:T
        obj1(t) = trace(S*Theta(:,:,t)) - log(det(Theta(:,:,t))) + lambda*norm(Z0(:,:,t), 1);
    end

    obj2 = zeros(1,T);
    for t = 2:1:T
        obj2(t) =  beta*TimePenalty(Z2(:,:,t),Z1(:,:,t),type);
    end

    obj = obj1 + obj2;


end



function E = L1_element_wise(A, beta, rho)
        % The element-wise l1 penalty function.
        % Used in (Z1, Z2) update
    
        eta = (2*beta) / rho;
        dimension = size(A, 1);
        E = zeros(dimension);
    
        % Apply element-wise thresholding
        for ii = 1:dimension
            for jj = 1:dimension
                if abs(A(ii, jj)) > eta
                    
                    result = sign(A(ii, jj)) * (abs(A(ii, jj)) - eta);
                    E(ii, jj) = result;
                    E(jj, ii) = result;
                end
            end
        end
end