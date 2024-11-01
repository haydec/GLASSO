% Example covariance matrix
S = [1.0, 0.5, 0.3;
     0.5, 1.0, 0.4;
     0.3, 0.4, 1.0];  % 3x3 covariance matrix as an example

% Call the custom Graphical fitting function without Lasso regularization
Theta_optimal = fitGraphicalModel(S);

% Display the result
disp('Optimal Precision Matrix (Theta):');
disp(Theta_optimal);

% Main function to fit Graphical model without Lasso
function Theta_optimal = fitGraphicalModel(S)
    % Fit a graphical model without Lasso regularization using fmincon in MATLAB.
    % Input:
    % - S: Empirical covariance matrix
    % Output:
    % - Theta_optimal: Estimated precision matrix

    d = size(S, 1);  % Dimension of the covariance matrix
    
    % Initial guess: use only upper triangular elements
    Theta0 = eye(d);
    Theta_upper0 = Theta0(triu(true(d))); % Extract upper triangular part

    % Set up the objective function
    objective = @(Theta_upper) graphicalModelObjective(Theta_upper, S, d);

    % Optimization options
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter');

    % Solve the optimization
    [Theta_upper_optimal, ~] = fmincon(objective, Theta_upper0, [], [], [], [], [], [], ...
                                       @(Theta_upper) positiveSemidefiniteConstraint(Theta_upper, d), options);

    % Reconstruct the symmetric matrix from the optimized upper triangular part
    Theta_optimal = zeros(d, d);
    Theta_optimal(triu(true(d))) = Theta_upper_optimal;
    Theta_optimal = Theta_optimal + triu(Theta_optimal, 1)';  % Make it symmetric
end

% Objective function for Graphical model without Lasso
function f = graphicalModelObjective(Theta_upper, S, d)
    % Objective function for the Graphical model without Lasso
    % Input:
    % - Theta_upper: Upper triangular elements of the precision matrix
    % - S: Covariance matrix
    % - d: Dimension of Theta

    % Reconstruct the symmetric matrix from the upper triangular elements
    Theta = zeros(d, d);
    Theta(triu(true(d))) = Theta_upper;
    Theta = Theta + triu(Theta, 1)';

    % Objective: -log(det(Theta)) + trace(S*Theta)
    if det(Theta) <= 0
        f = inf; % Return a high value if Theta is not positive semidefinite
    else
        log_det_term = -log(det(Theta));
        trace_term = trace(S * Theta);

        % Objective function value
        f = log_det_term + trace_term;
    end
end

% Convex Relaxation: Positive Semidefinite Constraint
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
