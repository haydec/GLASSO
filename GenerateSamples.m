function [samples,pre_tensor,cov_tensor] = GenerateSamples( n_samples,sections,n,mean_value,std_dev)
    
    assert(length(std_dev) == sections, "Standard Devivation array must equal the number of Sections")


    [pre_tensor, cov_tensor] = dispersion_tensor(sections,n,mean_value,std_dev);
    
    
    
    % Number of samples to generate
    data = GenerateRandomData(n,cov_tensor,n_samples,sections);
    
    plot(data)
    

    [numObservations, numVariables] = size(data);
    numFullSamples = floor(numObservations / n_samples);
    
    % Extract only the data that fits into full samples
    sampledData = data(1:numFullSamples * n_samples, :);
    
    % Reshape into samples with dimensions [sampleSize, numVariables, numFullSamples]
    samples = reshape(sampledData, n_samples, numVariables, numFullSamples);



    function [pre_tensor, cov_tensor] = dispersion_tensor(sections,n,mean_value,std_dev)
        
        pre_tensor = zeros(n,n,sections);
        cov_tensor = zeros(n,n,sections);
        
        for s = 1:1:sections
            [pre_matrix, cov_matrix] = dispersion_matrix(n,mean_value,std_dev(s));
            pre_tensor(:,:,s) = pre_matrix;
            cov_tensor(:,:,s) = cov_matrix;
        end
    
    end

    function [pre_matrix, cov_matrix] = dispersion_matrix(n,mean_value,std_dev)
        % Generate random numbers from the specified normal distribution
        matrix = mean_value + std_dev * randn(n, n);
        
        % Construct the positive definite covariance matrix
        cov_matrix = matrix' * matrix;
        
        % Scale the matrix to have variances close to 1 on the diagonal (optional)
        cov_matrix = cov_matrix ./ max(abs(diag(cov_matrix)));
        isPositiveDef(cov_matrix)
        
        % precision matrix
        pre_matrix = inv(cov_matrix);
        isPositiveDef(pre_matrix)
    end
    
    
    function data = GenerateRandomData(num_variables,true_covariance,n_samples,sections)
        
        data = zeros(n_samples*sections,num_variables);
        Start = 1;
        End = n_samples;
        for s = 1:1:sections
        
            rng(0);  % Set random seed for reproducibility
            data(Start:End,:) = mvnrnd(zeros(1, num_variables), true_covariance(:,:,s), n_samples);
            Start = Start + n_samples;
            End = End + n_samples;
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