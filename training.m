[X,T]=loadMatrix();
trainingsize=uint32(0.4 * size(X,1));
validationsize=uint32(0.1 * size(X,1));
testsize=uint32(size(X,1))-(trainingsize+validationsize);

trainX=X(1:trainingsize,:);
trainT=T(1:trainingsize,:);
valX=X((trainingsize+1):(trainingsize+validationsize),:);
valT=T((trainingsize+1):(trainingsize+validationsize),:);
testX=X((trainingsize+validationsize+1):end,:);
testT=T((trainingsize+validationsize+1):end,:);

maxM = 25;
lambdaSet=[30,50,100,300,500];
lambdaSetSize = size(lambdaSet,2);

trainErrors=zeros(1,maxM);
valErrors=zeros(1,maxM);
bestLambda=0;
valErrorsLambda=zeros(1,maxM);



for M = 1:maxM
    %means and standard deviation
    
    means = zeros(M-1,1);
    stdDev = zeros(M-1,1);
    
if M ~= 1
    partition = M-1;
    begin = 1;
    

    partLimit = floor(size(trainX,1)/partition);
    last = partLimit;
    for k=1:partition
        subX=trainX(begin:last,:);
        begin = last+1;
        last = begin + partLimit;
        if last>=size(trainX,1)
            last=size(trainX,1);
        end
        means(k)= mean(mean(subX));
        stdDev(k)= std(mean(subX));
    end
end

%design matrix
    [N,D] = size(trainX);
    phi = zeros(N,(M-1)* D + 1);
    phi(1:end, 1) = 1;

for i = 1:N
    for j = 1:(M-1)
        for k = 1:D
                phi(i, j * k + 1) = exp(-1 * ((trainX(i,k) - means(j))^2) / (2 * (stdDev(j)^2)));
        end
    end
end
    
%calculate regularized weights for given model
wsize=size((phi.' * phi +  lambdaSet(4)* eye((M-1)* D + 1)) \ phi.' * trainT);
w=zeros(wsize);
min_t_E_Rms=1;

for lIndex=1:lambdaSetSize
w = (phi.' * phi +  lambdaSet(lIndex)* eye((M-1)* D + 1)) \ phi.' * trainT;


% COMPUTE TRAINING ERROR
    % Predict output
    y = phi * w;

    % Compute sum of squared errors
    E_D = 0.5 * sum( (trainT - y).^2 );

    % Compute Root Mean Square Error
    E_rms = sqrt( 2 * E_D / N );
    if(E_rms<min_t_E_Rms)
        min_t_E_Rms=E_rms;
    end
    

    % Store E_RMS value for current M, lambda combination
    trainErrors(M) = min_t_E_Rms;
end   
 

% COMPUTE VALIDATION ERROR

%design matrix for validation
    [Nval,Dval] = size(valX);
    phi = zeros(Nval,(M-1)* Dval + 1);
    phi(1:end, 1) = 1;
    for i = 1:Nval
        for j = 1:(M-1)
            for k = 1:Dval
                phi(i, j * k + 1) = exp(-1 * ((valX(i,k) - means(j))^2) / (2 * (stdDev(j)^2)));
            end
        end
    end
    
w=zeros(wsize);
min_v_E_Rms=1;
%disp('--------');
for lIndex1=1:lambdaSetSize
    %disp(lambdaSet(lIndex1));
w = (phi.' * phi +  lambdaSet(lIndex1)* eye((M-1)* D + 1)) \ phi.' * valT;

    % Predict output
    y = phi * w;

    % Compute sum of squared errors
    E_D = 0.5 * sum( (valT - y).^2 );

    % Compute ERMS
    E_rms = sqrt( 2 * E_D / Nval );
    %disp('calc erms');
   % disp(E_rms);
    
    if(E_rms<min_v_E_Rms)
        min_v_E_Rms=E_rms;
        bestLambda=lambdaSet(lIndex1);
       % disp(min_v_E_Rms);
       % disp(lIndex1);
        
    end
    %disp('min erms')
    %disp(min_v_E_Rms);

    % Set of E_RMS
    valErrors(M) = min_v_E_Rms;
    valErrorsLambda(M)=bestLambda;
end
%disp('--------');


end

% determine best model for test
% Find minimum E_RMS value observed
Minimum_E_Rms = min(valErrors(:));

% Get the indices of the minimum E_RMS to get best M
M = find(valErrors == Minimum_E_Rms);
bestLmodel = valErrorsLambda(M);

partition = M-1;
    begin = 1;
    

    partLimit = floor(size(testX,1)/partition);
    last = partLimit;
    for k=1:partition
        subX=testX(begin:last,:);
        begin = last+1;
        last = begin + partLimit;
        if last>=size(testX,1)
            last=size(testX,1);
        end
        means(k)= mean(mean(subX));
        stdDev(k)= std(mean(subX));
    end


%design matrix
    [N,D] = size(testX);
    phi = zeros(N,(M-1)* D + 1);
    phi(1:end, 1) = 1;

for i = 1:N
    for j = 1:(M-1)
        for k = 1:D
                phi(i, j * k + 1) = exp(-1 * ((testX(i,k) - means(j))^2) / (2 * (stdDev(j)^2)));
        end
    end
end
    
%calculate regularized weights for given model
wsize=size((phi.' * phi +  bestLmodel* eye((M-1)* D + 1)) \ phi.' * testT);
w=zeros(wsize);
min_t_E_Rms=1;


w = (phi.' * phi +  bestLmodel* eye((M-1)* D + 1)) \ phi.' * testT;
%predict for test set
[y, E_rms] = predict(testX, testT, w, means, stdDev, M);


rms_nn=0.2693;
sprintf('the model complexity M for the linear regression model is %d', M);
sprintf('the regularization parameters lambda for the linear regression model is %f', bestLmodel);
sprintf('the root mean square error for the linear regression model is %f', E_rms);
sprintf('the root mean square error for the neural network model is %f', rms_nn);
        
