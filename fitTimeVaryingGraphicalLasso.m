% Main function to fit a time-varying Graphical Lasso model with optional Lasso penalty and PSD constraint
function Theta_optimal = fitTimeVaryingGraphicalLasso(Theta0, S, lasso_penalty,num_samples, enforce_psd)
    % Fits a time-varying graphical lasso model with optional sparsity and PSD constraint.
    % Minimizes the sum of graphical lasso objectives over all time points.
    %
    % Input:
    % - Theta0: Initial precision matrix (nxnxT)
    % - S: Empirical covariance matrix (nxnxT)
    % - lasso_penalty: Regularization parameter for sparsity
    % - enforce_psd: Boolean flag to enforce positive semidefinite constraint
    % Output:
    % - Theta_optimal: Estimated precision matrix with sparsity (nxnxT)

    [n, ~, T] = size(S);          % Dimensions: n x n covariance matrix for T time points
    d = n * (n + 1) / 2;          % Number of unique elements in upper triangular matrix
    Theta0_vec = [];              % Initialize vector for initial parameters

    % Prepare initial values for optimization
    for t = 1:T
        % Extract upper triangular elements of Theta0 at each time slice
        Theta_upper = Theta0(:,:,t);
        Theta0_vec = [Theta0_vec; Theta_upper(triu(true(n)))];
    end

    % Define the objective function as the sum of individual objectives
    objective = @(Theta_vec) sumGraphicalLassoObjective(Theta_vec, S, lasso_penalty, n, T,num_samples);

    % Optimization options
    Display = 'final'; % Change to 'iter' for debugging or observing convergence
     options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'Display', Display, ...            % Shows iteration information; use 'none' if you prefer no display
    'TolFun', 1e-10, ...               % Function tolerance (smaller values increase precision)
    'TolX', 1e-10, ...                 % Step size tolerance (smaller values increase precision)
    'MaxIterations', 5000, ...         % Maximum number of iterations (increase if needed)
    'MaxFunctionEvaluations', 1e6);    % Increase the maximum function evaluations

    % Define the PSD constraints for all time points, if required
    if enforce_psd
        nonlcon = @(Theta_vec) positiveSemidefiniteConstraints(Theta_vec, n, T);
    else
        nonlcon = [];
    end

    % Solve the optimization problem
    [Theta_optimal_vec, ~] = fmincon(objective, Theta0_vec, [], [], [], [], [], [], nonlcon, options);

    % Reshape optimized vector back into 3D precision matrix
    Theta_optimal = zeros(n, n, T);
    idx = 1;
    for t = 1:T
        Theta_upper = Theta_optimal_vec(idx:idx + d - 1);
        Theta = zeros(n, n);
        Theta(triu(true(n))) = Theta_upper;
        Theta_optimal(:, :, t) = Theta + triu(Theta, 1)';
        idx = idx + d;
    end
end


% Sum of graphical lasso objectives across all time points
function f = sumGraphicalLassoObjective(Theta_vec, S, lasso_penalty, n, T, num_samples)
    % Compute the sum of graphical lasso objectives across time points
    % Input:
    % - Theta_vec: Concatenated upper triangular elements across all time points
    % - S: Empirical covariance matrix (nxnxT)
    % - lasso_penalty: Regularization parameter for sparsity
    % - n: Dimension of precision matrix
    % - T: Number of time points

    d = n * (n + 1) / 2;  % Number of upper triangular elements
    f = 0;
    idx = 1;

    for t = 1:T
        % Extract the upper triangular part for the current time point
        Theta_upper = Theta_vec(idx:idx + d - 1);
        
        % Reconstruct symmetric precision matrix for the time point
        Theta = zeros(n, n);
        Theta(triu(true(n))) = Theta_upper;
        Theta = Theta + triu(Theta, 1)';

        % Calculate the graphical lasso objective for this time point
        if det(Theta) <= 0
            f = f + inf; % Return a high value if Theta is not invertible
        else
            log_det_term = -log(det(Theta));
            trace_term = trace(S(:, :, t) * Theta);
            l1_penalty = lasso_penalty * sum(abs(Theta(:)));  % Lasso (L1) penalty

            % Sum objective value for this time point
            f = f + num_samples*(log_det_term + trace_term) + l1_penalty;
        end
        idx = idx + d;
    end
end

% Positive Semidefinite Constraints across all time points
function [c, ceq] = positiveSemidefiniteConstraints(Theta_vec, n, T)
    % Non-linear constraints for positive semidefiniteness across all time points

    d = n * (n + 1) / 2;  % Number of upper triangular elements
    c = [];  % Initialize inequality constraints
    ceq = [];  % No equality constraints

    idx = 1;
    for t = 1:T
        % Extract and reconstruct the precision matrix for each time point
        Theta_upper = Theta_vec(idx:idx + d - 1);
        Theta = zeros(n, n);
        Theta(triu(true(n))) = Theta_upper;
        Theta = Theta + triu(Theta, 1)';

        % Eigenvalues for positive semidefiniteness
        eigvals = eig(Theta);
        c = [c; -eigvals]; % Ensure all eigenvalues are >= 0
        idx = idx + d;
    end
end
