close All
clear
clc

rho = 0.01;
lambda = 0.001; % Sparsity Penalty 
beta = 0; % Time Varying Penalty
MAX_ITER = 1000;
alpha = 1; % alpha is the over-relaxation parameter




disp("------------Generate Data------------------")
n_samples = 1000;
sections = 10;
n = 5;
mean_value = 0;
%std_dev = repelem(0.1,sections);
std_dev = linspace(0.1,1,sections);
%std_dev = randi([1 100],1,sections)/100

RandMatrixSeed = randn(n,n);
[D,pre_tensor,cov_tensor] = GenerateSamples( n_samples,sections,n,mean_value,std_dev,RandMatrixSeed);
disp("D Size: " + num2str(size(D)))
NumberOfObservations = size(D,1);
NumberOfVariables = size(D,2);
NumberOfSections = size(D,3);


% Efficient concatenation without a loop
Dplot = reshape(permute(D, [1, 3, 2]), [], size(D, 2));

% Plot the concatenated data



S = zeros(NumberOfVariables,NumberOfVariables,NumberOfSections);
%Data preprocessing
for t = 1:1:NumberOfSections
    S(:,:,t) = cov(D(:,:,t)); 
end


Theta_hat = fitTVGL(S,lambda,beta,n_samples,rho,MAX_ITER,alpha);

 printResult = true;
if printResult == true
    fprintf('iter\t primal_norm\t primal_tol\t dual_norm\t dual_tol\t Obj\n')
    disp("THETA HAT 1st TimeStep")
    idx = randi(size(Theta_hat,3),1,1);
    disp("At Index: " + idx)
    disp(Theta_hat(:,:,idx))
    disp("Pre Tensor 1st TimeStep")
    disp(pre_tensor(:,:,idx))
end


% Calculate Frobenius Norm
T = size(Theta_hat,3); % Number Of Timesteps
ThetaNormF= zeros(1,T-1);
PreNormF = zeros(1,T-1);
for t = 2:1:T
    Theta_Diff = Theta_hat(:,:,t) - Theta_hat(:,:,t-1);
    ThetaNormF(t-1) = norm(Theta_Diff,"fro");
    Pre_Diff = pre_tensor(:,:,t) - pre_tensor(:,:,t-1);
    PreNormF(t-1) = norm(Pre_Diff,"fro");
end


figure
subplot(1,2,1)
plot(Dplot)
title("Timeseries")
subplot(1,2,2)
plot(log(PreNormF))
hold on
plot(log(ThetaNormF))
hold off
title("Log Frobenius Norm: \Theta_{T}-\Theta_{T-1} ")
xlabel("T")

function Theta_hat = fitTVGL(S,lambda,beta,n_samples,rho,MAX_ITER,alpha)
    
    n = size(S,1); % Number Of Variables
    T = size(S,3); % Number Of Timesteps

    eta = n_samples / (3 * rho);  % Adjusted penalty parameter for the proximal operator

    BetaConsesus = 2*(beta > 0);
    LambdaConsensus = (lambda > 0);
    NumConsensus = LambdaConsensus + BetaConsesus;
    disp("Number Of Consesus Variables: " + num2str(NumConsensus))
    
    if NumConsensus < 1
        disp("TVGL via ADMM not defined for no constraints" )
        return
    end
        
    % Precision Variables
    Theta_hat = zeros(size(S,1),size(S,2),size(S,3));
    
    % Consensus Variables
    Z0 = zeros(size(S,1),size(S,2),size(S,3));
    Z1 = zeros(size(S,1),size(S,2),size(S,3));
    Z2 = zeros(size(S,1),size(S,2),size(S,3));
    
    % Dual Variables
    U0 = zeros(size(S,1),size(S,2),size(S,3));
    U1 = zeros(size(S,1),size(S,2),size(S,3));
    U2 = zeros(size(S,1),size(S,2),size(S,3));


    t_start = tic;
    %Global constants and defaults
    QUIET    = 0;
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;


    for k = 1:1:MAX_ITER
            
        Z0prev = Z0; % For Diagnostics (Check Convergence)
        Z1prev = Z1; % For Diagnostics (Check Convergence)
        Z2prev = Z2; % For Diagnostics (Check Convergence)
        U0prev = U0; % For Diagnostics (Check Convergence)
        U1prev = U1; % For Diagnostics (Check Convergence)
        U2prev = U2; % For Diagnostics (Check Convergence)
        
        % Update Precision Matrix
        Theta_hat = update_theta(Z0,Z1,Z2,U0,U1,U2,S,eta,lambda,beta,alpha);
    
        % z-update with relaxation
        Z0 =  update_z0(Theta_hat,U0,lambda,rho); 
        [Z1,Z2] = update_z1z2(Theta_hat,Z1,Z2,U1,U2,beta,rho);
           
    
        % u-update (Dual Variable) 
        U0 = update_u0(Theta_hat,U0,Z0,lambda);
        [U1,U2] = update_u1u2(Theta_hat,Z1,Z2,U1,U2,beta);
    
    
      
    
        
        % diagnostics, reporting, termination checks
       
        history.objval(k)  = objective(S, Theta_hat, Z0,Z1,Z2, lambda,beta,T);
        
        r0 = 0;
        for t = 1:1:T      
            r0 = r0 + norm(Theta_hat(:,:,t) - Z0(:,:,t), 'fro');
        end
        r1 = 0;
        r2 = 0;
    
        if beta > 0
            for t = 2:1:T
                r1 = r1 + norm(Z1(:,:,t) - Z1(:,:,t-1), 'fro');
                r2 = r2 + norm(Theta_hat(:,:,t) - Z2(:,:,t-1), 'fro');
            end
        end
        
        d0 = 0;
        for t = 1:1:T      
            d0 = d0 + norm(Z0(:,:,t) - Z0prev(:,:,t), 'fro');
        end
        
        d1 = 0;
        d2 = 0;
        if beta > 0
            for t = 2:1:T
                d1 = d1 + norm(Z1(:,:,t) - Z1prev(:,:,t-1), 'fro');
                d2 = d2 + norm(Z2(:,:,t) - Z2prev(:,:,t-1), 'fro');
            end
        end
    
    
        history.r_norm(k)  = sqrt(r0 + r1 + r2);
        history.s_norm(k)  = rho*sqrt(d0 + d1 + d2);
    
        history.eps_pri(k) = sqrt(3*n)*ABSTOL + RELTOL*max([norm(Theta_hat,'fro'), norm(Z0,'fro'), norm(Z1,'fro'), norm(Z2,'fro')]);
        history.eps_dual(k)= sqrt(3*n)*ABSTOL + RELTOL*rho*sqrt( norm(U0,'fro') + norm(U1,'fro') + norm(U2,'fro') );
    
    
        if ~QUIET
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
                history.r_norm(k), history.eps_pri(k), ...
                history.s_norm(k), history.eps_dual(k), history.objval(k));
        end
    
        if (history.r_norm(k) < history.eps_pri(k) && ...
           history.s_norm(k) < history.eps_dual(k))
             break;
        end
     
   
    
    end

end


function Theta_hat = update_theta(Z0,Z1,Z2,U0,U1,U2,S,eta,lambda,beta,alpha)

        Theta_hat = zeros(size(S));
        T = size(S,3);
        for t = 1:1:T
            %disp("------------------------------")
            A = ( ( Z0(:,:,t) + Z1(:,:,t) +  Z2(:,:,t) ) - ( U0(:,:,t) + U1(:,:,t) +  U2(:,:,t) ) )/( (lambda>0) + 2*(beta>0)  );
            %A = round(A,5);
        
            % Theta Update
            
            Asym = (A + A')/2;
            M = (1/eta)*(Asym) - S(:,:,t); % (rho)*(Asym) - S;
            [Q,L] = eigs(M);
            es = diag(L);
            xi = (es + sqrt( es.^2 + 4*(1/eta) )) ./( 2*( 1/eta )); % (es + sqrt(es.^2 + 4*rho))./(2*rho);
            Theta =  Q*diag(xi)*Q';
    
    
            
            Z0old = Z0(:,:,t);
            Theta_hat(:,:,t) = alpha*Theta + (1 - alpha)*Z0old;
        end

end


function Z0 =  update_z0(Theta_hat,U0,lambda,rho)

        Z0 = zeros(size(Theta_hat));
        if lambda > 0
            T = size(Theta_hat,3);
            for t = 1:1:T
                Z0(:,:,t) = soft_threshold_odd(Theta_hat(:,:,t) + U0(:,:,t), lambda, rho);
            end
        end


end


function U0 = update_u0(Theta_hat,U0,Z0,lambda)
        
        
        if lambda > 0
            T = size(Theta_hat,3);
            for t = 1:1:T
                U0(:,:,t) = U0(:,:,t) + ( Theta_hat(:,:,t) - Z0(:,:,t) );
            end
        end
    
end


function [Z1,Z2] = update_z1z2(Theta_hat,Z1,Z2,U1,U2,beta,rho)
        
        if beta > 0
            T = size(Theta_hat,3);
            for t = 2:1:T
                Adiff = Theta_hat(:,:,t) - Theta_hat(:,:,t-1) + U2(:,:,t)  - U1(:,:,t-1) ; 
                
                E = L1_element_wise(Adiff, beta, rho);
                
                Asum = Theta_hat(:,:,t-1) + Theta_hat(:,:,t)  + U2(:,:,t) + U1(:,:,t-1);
                Z1(:,:,t-1) = 0.5*(Asum - E);
                Z2(:,:,t)   = 0.5*(Asum + E);
            end
        end
        

end



function [U1,U2] = update_u1u2(Theta_hat,Z1,Z2,U1,U2,beta)

        if beta > 0
            T = size(Theta_hat,3);
            for t = 2:1:T   
                U1(:,:,t-1) = U1(:,:,t-1) + ( Theta_hat(:,:,t-1) - Z1(:,:,t-1) );
                U2(:,:,t) = U2(:,:,t) + ( Theta_hat(:,:,t) - Z2(:,:,t) );
            end
        end
    

end


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


function obj = objective(S, Theta, Z0, Z1, Z2, lambda,beta,T)

    obj1 = zeros(1,T);
    for t = 1:1:T
        obj1(t) = trace(S(:,:,t)*Theta(:,:,t)) - log(det(Theta(:,:,t))) + L1_odd(Z0(:,:,t), lambda);
    end

    obj2 = zeros(1,T);
    for t = 2:1:T       
        obj2(t) =  L1_odd(Z2(:,:,t) - Z1(:,:,t), beta);
    end

    obj = sum(obj1 + obj2);


end


function off_diag_l1_norm = L1_odd(A, lambda)

    % Computes the Z0 update with the off-diagonal soft-threshold operator
    
    % Get the dimension of the matrix
    % Create a copy of the matrix with zeros on the diagonal
    A_off_diag = A - diag(diag(A));
    
    % Calculate the off-diagonal L1 norm (sum of absolute values of off-diagonal elements)
    off_diag_l1_norm = lambda*norm(A_off_diag,1);

    
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


function [samples,pre_tensor,cov_tensor] = GenerateSamples( n_samples,sections,n,mean_value,std_dev,RandMatrixSeed)
    
    assert(length(std_dev) == sections, "Standard Devivation array must equal the number of Sections")


    [pre_tensor, cov_tensor] = dispersion_tensor(sections,n,mean_value,std_dev,RandMatrixSeed);
    
    
    
    % Number of samples to generate
    samples = GenerateRandomData(n,cov_tensor,n_samples,sections);
    
    
    

    %[numObservations, numVariables] = size(data);
    %numFullSamples = floor(numObservations / n_samples);
    
    % Extract only the data that fits into full samples
    %sampledData = data(1:numFullSamples * n_samples, :);
    
    % Reshape into samples with dimensions [sampleSize, numVariables, numFullSamples]
    %samples = reshape(sampledData, n_samples, numVariables, numFullSamples);



    function [pre_tensor, cov_tensor] = dispersion_tensor(sections,n,mean_value,std_dev,RandMatrixSeed)
        
        pre_tensor = zeros(n,n,sections);
        cov_tensor = zeros(n,n,sections);

                
        for s = 1:1:sections
            [pre_matrix, cov_matrix] = dispersion_matrix(mean_value,std_dev(s),RandMatrixSeed);
            pre_tensor(:,:,s) = pre_matrix;
            cov_tensor(:,:,s) = cov_matrix;
        end
    
    end

    function [pre_matrix, cov_matrix] = dispersion_matrix(mean_value,std_dev,RandMatrix)
        % Generate random numbers from the specified normal distribution
        matrix = mean_value + std_dev*RandMatrix;
        
        % Construct the positive definite covariance matrix
        cov_matrix = matrix' * matrix;
        
        % Scale the matrix to have variances close to 1 on the diagonal (optional)
        %cov_matrix = cov_matrix ./ max(abs(diag(cov_matrix)));
        %isPositiveDef(cov_matrix)
        
        % precision matrix
        pre_matrix = inv(cov_matrix);
        isPositiveDef(pre_matrix)
    end
    
    
    function data = GenerateRandomData(num_variables,true_covariance,n_samples,sections)
        
        data = zeros(n_samples,num_variables,sections);
        for s = 1:1:sections
        
            %rng(0);  % Set random seed for reproducibility
            data(:,:,s) = mvnrnd(zeros(1, num_variables), true_covariance(:,:,s), n_samples);

        end
    
    end

    
    function isPositiveDef(X)
        % Check that true_precision is positive definite
        eigvals = eig(X);
        if any(eigvals <= 0)
            disp('matrix is not positive definite');
        end
    end

end