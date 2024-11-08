% Main function to fit Time-Varying Graphical Lasso model with optional Lasso penalty and PSD constraint
function Theta_optimal = fitTimeVaryingGraphicalLasso2(Theta0, S, lasso_penalty, beta, enforce_psd)
    % Fit a time-varying graphical lasso model with Lasso regularization and optional temporal regularization.
    % Input:
    % - Theta0: Initial precision matrix (nxnxT)
    % - S: Empirical covariance matrix (nxnxT)
    % - lasso_penalty: Regularization parameter for sparsity
    % - beta: Temporal regularization parameter (for smoothness between time points)
    % - enforce_psd: Boolean flag, if true, enforces positive semidefinite constraint
    % Output:
    % - Theta_optimal: Estimated precision matrices with sparsity and temporal regularization (nxnxT)

    [d, ~, T] = size(S);  % Dimensions: d x d covariance matrix for T time points
    Theta_upper0 = [];
    
    % Prepare initial values for optimization by extracting upper triangular elements across time points
    for t = 1:T
        Theta_upper = Theta0(:,:,t);
        Theta_upper0 = [Theta_upper0; Theta_upper(triu(true(d)))];
    end

    % Define the objective function as the sum of individual objectives
    objective = @(Theta_upper) sumGraphicalLassoObjective(Theta_upper, S, lasso_penalty, beta, d, T);

    % Optimization options
    Display = 'none'; % Change to 'iter' for debugging
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', Display);

    % Define the PSD constraints for all time points, if required
    if enforce_psd
        nonlcon = @(Theta_upper) timeVaryingPositiveSemidefiniteConstraint(Theta_upper, d, T);
    else
        nonlcon = [];
    end

    % Solve the optimization problem
    [Theta_upper_optimal, ~] = fmincon(objective, Theta_upper0, [], [], [], [], [], [], nonlcon, options);

    % Reconstruct the optimized precision matrices for each time point
    Theta_optimal = zeros(d, d, T);
    idx = 1;
    num_elements = d * (d + 1) / 2;
    for t = 1:T
        Theta_upper_t = Theta_upper_optimal(idx:idx + num_elements - 1);
        Theta = zeros(d, d);
        Theta(triu(true(d))) = Theta_upper_t;
        Theta_optimal(:, :, t) = Theta + triu(Theta, 1)';
        idx = idx + num_elements;
    end
end

% Sum of graphical lasso objectives across all time points with temporal regularization
function f = sumGraphicalLassoObjective(Theta_upper, S, lasso_penalty, beta, d, T)
    % Compute the sum of graphical lasso objectives across time points with temporal regularization
    % Input:
    % - Theta_upper: Concatenated upper triangular elements across all time points
    % - S: Empirical covariance matrix (nxnxT)
    % - lasso_penalty: Regularization parameter for sparsity
    % - beta: Temporal regularization parameter for smoothness
    % - d: Dimension of precision matrix
    % - T: Number of time points

    f = 0;
    idx = 1;
    num_elements = d * (d + 1) / 2;
    prev_Theta = [];

    for t = 1:T
        % Extract the upper triangular part for the current time point
        Theta_upper_t = Theta_upper(idx:idx + num_elements - 1);

        % Reconstruct symmetric precision matrix for the time point
        Theta = zeros(d, d);
        Theta(triu(true(d))) = Theta_upper_t;
        Theta = Theta + triu(Theta, 1)';

        % Calculate the graphical lasso objective for this time point
        if det(Theta) <= 0
            f = f + inf; % Return a high value if Theta is not invertible
        else
            log_det_term = -log(det(Theta));
            trace_term = trace(S(:, :, t) * Theta);
            l1_penalty = lasso_penalty * sum(abs(Theta(:)));

            % Sum objective value for this time point
            f = f + (log_det_term + trace_term + l1_penalty);
        end

        % Temporal smoothness penalty
        if t > 1 && beta > 0
            smoothness_penalty = beta * norm(Theta - prev_Theta, 'fro')^2;
            f = f + smoothness_penalty;
        end

        prev_Theta = Theta;
        idx = idx + num_elements;
    end
end

% Positive Semidefinite Constraints across all time points
function [c, ceq] = timeVaryingPositiveSemidefiniteConstraint(Theta_upper, d, T)
    % Non-linear constraints for positive semidefiniteness across all time points

    num_elements = d * (d + 1) / 2;
    c = [];  % Initialize inequality constraints
    ceq = [];  % No equality constraints

    idx = 1;
    for t = 1:T
        % Extract and reconstruct the precision matrix for each time point
        Theta_upper_t = Theta_upper(idx:idx + num_elements - 1);
        Theta = zeros(d, d);
        Theta(triu(true(d))) = Theta_upper_t;
        Theta = Theta + triu(Theta, 1)';

        % Eigenvalues for positive semidefiniteness
        eigvals = eig(Theta);
        c = [c; -eigvals]; % Ensure all eigenvalues are >= 0
        idx = idx + num_elements;
    end
end
