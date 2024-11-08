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
    
    %% PART 1  Z_{i,0} Update: Element-wise soft-thresholding for sparsity

    for i = 1:T
        A = Theta(:,:,i) + U0(:,:,i);
        Z0(:,:,i) = elementwise_soft_threshold(A, lambda / rho);
    end

    %% Part 2 (Z_{i-1,1}, Z_{i,2}) Update: Proximal operator for temporal consistency
    for i = 2:T
        A = Theta(:,:,i-1) + Theta(:,:,i) + U1(:,:,i-1) + U2(:,:,i);
        Adiff = Theta(:,:,i-1) - Theta(:,:,i) + U1(:,:,i-1) - U2(:,:,i); 
        nu = (2*beta)/rho;
        switch penalty_type
            case 'l1'
                % Element-wise L1 penalty (soft-threshold each element)
                E = elementwise_soft_threshold(Adiff, nu);

            case 'l2'
                % Group Lasso L2 Penalty (block-wise soft thresholding)
                E = group_lasso_threshold(Adiff, nu);

            case 'laplacian'
                % Laplacian penalty (element-wise scaling)
                E = laplacian_threshold(Adiff, nu);

            case 'l_inf'
                % L_inf Penalty (requires iterative bisection)
                E = l_inf_threshold(Adiff, nu);

            otherwise
                error('Unknown penalty type');
        end
    
    Z1(:,:,i-1) = 0.5*A - 0.5*E;
    Z2(:,:,i)   = 0.5*A + 0.5*E;
     

    end

    
    function Z = elementwise_soft_threshold(A, threshold)
        % Element-wise soft thresholding
        Z = sign(A) .* max(abs(A) - threshold, 0);
        Z = (Z + Z')/2;
    end
    
    function Z = group_lasso_threshold(A, threshold)
        % Group Lasso (L2) proximal operator - column-wise thresholding
        norms = sqrt(sum(A.^2, 1));  % L2 norm of each column
        scaling = max(0, 1 - threshold ./ norms);
        Z = A .* scaling;
    end
    
    function Z = laplacian_threshold(A, threshold)
        % Laplacian penalty (L2^2) proximal operator
        Z = A / (1 + 2 * threshold);
    end
    
    function E = l_inf_threshold(A, threshold)
        % L_inf proximal operator for the given matrix A and threshold eta.
        % Parameters:
        % A - Input matrix to threshold
        % eta - Threshold parameter for the L_inf norm
    
        % Initialize E as zeros (default case if ||A||_1 <= eta)
        E = zeros(size(A));
        
        % Check the 1-norm of A
        if norm(A(:), 1) <= threshold
            % If ||A||_1 <= threshold, we can directly return zero matrix E
            return;
        end
    
        % Set up bisection parameters for solving sigma
        low = 0;
        high = max(abs(A(:))) / threshold;
        tol = 1e-6; % Tolerance for bisection
    
        % Bisection loop to solve for sigma
        while (high - low) > tol
            % Midpoint for current sigma
            sigma = (low + high) / 2;
    
            % Compute the sum for the constraint
            sum_constraint = sum(max(abs(A(:)) / threshold - sigma, 0));
    
            if sum_constraint > 1
                low = sigma;  % Increase sigma
            else
                high = sigma; % Decrease sigma
            end
        end
    
        % Final sigma after bisection
        sigma = (low + high) / 2;
    
        % Compute the final E using the proximal operator formula
        E = A - sigma * sign(A) .* max(abs(A) - threshold * sigma, 0);
    end

end