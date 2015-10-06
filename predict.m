unction [y, E_rms] = predict(X, T, w, means, stdDev, M)

 %design matrix
    [N,D] = size(X);
    phi = zeros(N,(M-1)* D + 1);
    phi(1:end, 1) = 1;

for i = 1:N
    for j = 1:(M-1)
        for k = 1:D
                phi(i, j * k + 1) = exp(-1 * ((X(i,k) - means(j))^2) / (2 * (stdDev(j)^2)));
        end
    end
end

    % Predict output
    y = phi * w;

    % Compute sum of squared errors
    E_d = 0.5 * sum( (T - y).^2 );

    % Compute Root Mean Square Error
    N = size(X,1);
    E_rms = sqrt( 2 * E_d / N );
end
