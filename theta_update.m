function Theta = theta_update(Z0, Z1, Z2, U0, U1, U2, S, n, rho)
    % Parameters:
    % Z0, Z1, Z2 - Consensus variables for each constraint
    % U0, U1, U2 - Dual variables for each constraint
    % S - Empirical covariance matrices
    % n - Number of samples
    % rho - ADMM penalty parameter

    % Constants
    eta = n / (3 * rho);  % Adjusted penalty parameter for the proximal operator
    num_timestamps = size(S, 3); % Number of time points (assuming S is [p, p, T])

    % Initialize Theta
    Theta = zeros(size(S, 1), size(S, 2), num_timestamps);


    % Loop over each intermediate time point (2 to T-1)
    for i = 1:1:num_timestamps
        % Calculate A for the current time step
        A = ( Z0(:,:,i) + Z1(:,:,i) + Z2(:,:,i) - U0(:,:,i) - U1(:,:,i) - U2(:,:,i) ) / 3;
        A = (A + A') / 2;  % Ensure symmetry of A

        % Calculate the proximal operator for Theta update
        M = (eta^-1) * A - S(:,:,i);
        Msym = (M + M') / 2;
        [Q, D] = eig(Msym); % Eigen decomposition of the symmetric matrix M
        D_updated =  D + sqrt( D^2 + (4*eta^-1)*eye(size(D)) );
        Theta(:,:,i) = (1 / ( 2*eta^-1 ) ) *  Q * D_updated * Q';
        % Check for complex numbers in the matrix
        if any(imag(Theta(:)) ~= 0)
            error('Complex numbers detected in the matrix Theta.');
        end
    end

end