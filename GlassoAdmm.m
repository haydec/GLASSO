clear
clc
% https://web.stanford.edu/class/ee364b/lectures/admm_slides.pdf
% Define a known precision matrix (inverse covariance) with a known sparsity pattern
lowVariance = false;
if lowVariance
    true_precision = [ 1.0, -0.5,  0.0,  0.0,  0.0;
                      -0.5,  1.0, -0.4,  0.0,  0.0;
                       0.0, -0.4,  1.0, -0.3,  0.0;
                       0.0,  0.0, -0.3,  1.0, -0.2;
                       0.0,  0.0,  0.0, -0.2,  1.0];
else
    true_precision = [ 10.0, -1.8, -1.5, -1.2, -0.9;
                  -1.8,  10.0, -1.6, -1.3, -1.0;
                  -1.5, -1.6,  10.0, -1.4, -1.1;
                  -1.2, -1.3, -1.4,  10.0, -1.2;
                  -0.9, -1.0, -1.1, -1.2,  10.0];
end

% Ensure the matrix is symmetric and positive definite
true_precision = (true_precision + true_precision') / 2;
eigvals = eig(true_precision);
if any(eigvals <= 0)
    error('Precision matrix is not positive definite');
end

% Compute the true covariance matrix as the inverse of the precision matrix
true_covariance = inv(true_precision);

% Number of samples to generate
n_samples = 10000;

% Generate data from a multivariate normal distribution
rng(0);  % Set random seed for reproducibility
data = mvnrnd(zeros(1, 5), true_covariance, n_samples);

%plot(data)

% Define the sample size and reshape data into segments
% ============================================================================
sampleSize = 10000;  % Define sample size (number of time steps in each segment)
% ============================================================================
[numObservations, numVariables] = size(data);
numFullSamples = floor(numObservations / sampleSize);

% Extract only the data that fits into full samples
sampledData = data(1:numFullSamples * sampleSize, :);

% Reshape into samples with dimensions [sampleSize, numVariables, numFullSamples]
samples = reshape(sampledData, sampleSize, numVariables, numFullSamples);


% Calculate the empirical covariance matrices for each segment
S = zeros(numVariables, numVariables, numFullSamples);  % Initialize covariance matrix storage
for t = 1:numFullSamples
    S(:, :, t) = cov(squeeze(samples(:, :, t)));  % Calculate covariance for each segment
end

disp("Empirical Matrix")
disp(S)


A = Z - U