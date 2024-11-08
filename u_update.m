function [U0, U1, U2] = u_update(U0, U1, U2, Theta, Z0, Z1, Z2)
    % Parameters:
    % U0 - Dual variable associated with Z0 (sparsity constraint)
    % U1 - Dual variable associated with Z1 (temporal consistency with Z2)
    % U2 - Dual variable associated with Z2 (temporal consistency with Z1)
    % Theta - Current values of Theta matrices for each time step
    % Z0, Z1, Z2 - Consensus variables for each constraint
    
    % Update U0 for the sparsity constraint
    for i = 1:size(Theta, 3)
        U0(:,:,i) = U0(:,:,i) + (Theta(:,:,i) - Z0(:,:,i));
    end

    % Update U1 and U2 for the temporal consistency constraint
    for i = 1:size(Theta, 3)-1
        U1(:,:,i) = U1(:,:,i) + (Theta(:,:,i) - Z1(:,:,i));
        U2(:,:,i+1) = U2(:,:,i+1) + (Theta(:,:,i+1) - Z2(:,:,i+1));     
    end

end