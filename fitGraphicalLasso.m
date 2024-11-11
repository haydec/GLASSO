% Main function to fit Graphical Lasso model with optional Lasso penalty and PSD constraint
function Theta_optimal = fitGraphicalLasso(D, lasso_penalty, enforce_psd)
    % Fit a graphical lasso model with Lasso regularization using fmincon in MATLAB.
    % Input:
    % - S: Empirical covariance matrix
    % - lasso_penalty: Regularization parameter for sparsity
    % - enforce_psd: Boolean flag, if true, enforces positive semidefinite constraint
    % Output:
    % - Theta_optimal: Estimated precision matrix with sparsity    
    S = cov(D);
    Theta0 = eye(size(S,1),size(S,2));
    d = size(S, 1);  % Dimension of the covariance matrix   
    Theta_upper0 = Theta0(triu(true(d))); % Extract upper triangular part

    % Set up the objective function with Lasso penalty
    objective = @(Theta_upper) graphicalLassoObjective(Theta_upper, S, lasso_penalty, d);
    
    % Display option
    Display = 'none'; % Change to 'iter' for debugging or observing convergence

    % Optimization options
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', Display);

    % Choose whether to enforce the positive semidefinite constraint
    if enforce_psd
        % Solve with PSD constraint
        [Theta_upper_optimal, ~] = fmincon(objective, Theta_upper0, [], [], [], [], [], [], ...
                                           @(Theta_upper) positiveSemidefiniteConstraint(Theta_upper, d), options);
    else
        % Solve without PSD constraint
        [Theta_upper_optimal, ~] = fmincon(objective, Theta_upper0, [], [], [], [], [], [], [], options);
    end

    % Reconstruct the symmetric matrix from the optimized upper triangular part
    Theta_optimal = zeros(d, d);
    Theta_optimal(triu(true(d))) = Theta_upper_optimal;
    Theta_optimal = Theta_optimal + triu(Theta_optimal, 1)';  % Make it symmetric
end

% Objective function for Graphical Lasso with Lasso penalty
function f = graphicalLassoObjective(Theta_upper, S, lasso_penalty, d)
    % Objective function for the Graphical Lasso with Lasso penalty
    % Input:
    % - Theta_upper: Upper triangular elements of the precision matrix
    % - S: Covariance matrix
    % - lasso_penalty: Regularization parameter for sparsity
    % - d: Dimension of Theta

    % Reconstruct the symmetric matrix from the upper triangular elements
    Theta = zeros(d, d);
    Theta(triu(true(d))) = Theta_upper;
    Theta = Theta + triu(Theta, 1)';

    % Objective: -log(det(Theta)) + trace(S*Theta) + lasso_penalty * sum(abs(Theta(:)))
    if det(Theta) <= 0
        f = inf; % Return a high value if Theta is not invertible
    else
        log_det_term = -log(det(Theta));
        trace_term = trace(S * Theta);
        l1_penalty = lasso_penalty * sum(abs(Theta(:)));  % Lasso (L1) penalty

        % Objective function value
        f = log_det_term + trace_term + l1_penalty;
    end
end

% Positive Semidefinite Constraint
function [c, ceq] = positiveSemidefiniteConstraint(Theta_upper, d)
    % Non-linear constraint for positive semidefiniteness of Theta
    
    % Reconstruct the symmetric matrix from the upper triangular elements
    Theta = zeros(d, d);
    Theta(triu(true(d))) = Theta_upper;
    Theta = Theta + triu(Theta, 1)';

    % Eigenvalues of Theta (for positive semidefiniteness constraint)
    eigvals = eig(Theta);
    c = -eigvals;  % Ensure all eigenvalues are >= 0 (for semidefiniteness)
    ceq = [];      % No equality constraints
end
