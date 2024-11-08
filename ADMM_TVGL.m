function Theta = ADMM_TVGL(S, n, rho, lambda, beta, penalty_type, max_iter, tol)
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
    [p, ~, T] = size(S);         % p: dimension of each matrix, T: number of time points
    Theta = zeros(p, p, T);       % Initialize Theta
    Z0 = zeros(p, p, T);          % Initialize Z0 (sparsity)
    Z1 = zeros(p, p, T-1);        % Initialize Z1 (temporal consistency)
    Z2 = zeros(p, p, T-1);        % Initialize Z2 (temporal consistency)
    U0 = zeros(p, p, T);          % Initialize U0 (dual variable for sparsity)
    U1 = zeros(p, p, T-1);        % Initialize U1 (dual variable for Z1)
    U2 = zeros(p, p, T-1);        % Initialize U2 (dual variable for Z2)

    % ADMM iteration loop
    for k = 1:max_iter
        % Step (a): Update Theta using the theta_update function
        Theta = theta_update(Z0, Z1, Z2, U0, U1, U2, S, n, rho);

        % Step (b): Update Z using the z_update function
        [Z0, Z1, Z2] = z_update(Theta, U0, U1, U2, lambda, beta, rho, penalty_type);

        % Step (c): Update dual variables U using the u_update function
        [U0, U1, U2] = u_update(U0, U1, U2, Theta, Z0, Z1, Z2);

        % Compute primal residual norm
        primal_residual = 0;
        for t = 1:T
            primal_residual = primal_residual + norm(Theta(:,:,t) - Z0(:,:,t), 'fro')^2;
        end
        for t = 2:T
            primal_residual = primal_residual + norm(Theta(:,:,t-1) - Z1(:,:,t-1), 'fro')^2 + norm(Theta(:,:,t) - Z2(:,:,t-1), 'fro')^2;
        end
        primal_residual = sqrt(primal_residual);

        % Compute dual residual norm
        dual_residual = 0;
        for t = 1:T
            dual_residual = dual_residual + norm(rho * (Z0(:,:,t) - Theta(:,:,t)), 'fro')^2;
        end
        for t = 2:T
            dual_residual = dual_residual + norm(rho * (Z1(:,:,t-1) - Theta(:,:,t-1)), 'fro')^2 + norm(rho * (Z2(:,:,t-1) - Theta(:,:,t)), 'fro')^2;
        end
        dual_residual = sqrt(dual_residual);

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
    end
end
