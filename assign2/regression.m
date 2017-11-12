%Take the appropriate matrix from project 1
remove_outliers_pro2



%How to deal with 1-on-K coding here ??? do a forward selection with 46 features ?

%% First regression with all features
% Create holdout crossvalidation partition
M=M2_data;
lines=1:size(M,1);
N = size(M,1);
K=10;
CV = cvpartition(N, 'Holdout', K);

% Extract training and test set
M_train= M(CV.training, :);
M_test= M(CV.test,:);
area2_train= area2(CV.training,:);
area2_test= area2(CV.test,:);

%Check if funLinreg  works

funLinreg = @(X_train, y_train, X_test, y_test) ...
    sum((y_test-glmval(glmfit(X_train, y_train), ...
    X_test, 'identity')).^2);


Err2 = funLinreg(M_train, area2_train, M_test, area2_test);

%Manually
w_est=glmfit(M_train,area2_train);
area_train_estim = glmval(w_est, M_train, 'identity');
area_test_estim = glmval(w_est,M_test,'identity');
Error_gene= sum((area_test_estim - area2_test).^2);
Error_trn=sum((area_train_estim - area2_train).^2);
area2_estim = [area_train_estim;area_test_estim ];
area3=[area2_train;area2_test];


mfig('Area estimated barea2 regression and True Area'); clf;
plot(lines, area2_estim, '+',lines,area3,'o');
xlabel('data');
ylabel('area estimated');
legend('estimation', 'true data');
Error_gene

mfig('Area burnt'); clf;
plot(lines, area2,'o');
xlabel('data');
ylabel('area estimated');

%% forward attribute selection
colonne=1:size(M,2);
%Double layer
K=5;
m=length(colonne);
% Initialize variables
Features = nan(K,m);
Histo = nan(K,m);

Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_fs = nan(K,1);
Error_test_fs = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
CV = cvpartition(N, 'Kfold',K);

for k =1:K
     fprintf('Crossvalidation fold %d/%d\n', k, K);
      % Extract the training and test set
    M2_data_train = M2_data(CV.training(k), :);
    area2_train = area2(CV.training(k));
    M2_data_test = M2_data(CV.test(k), :);
    area2_test = area2(CV.test(k));

    % Sequential feature selection with sequentialfs: try all the
    % attributes until no improvement in the error (funlinreg)
    % The first layer of cross-validation is then already done here...
    [F, H] = sequentialfs(funLinreg, M2_data_train, area2_train);
    
    % Save the selected features
    Features(k,:) = F;  
    % Error with all features at each K fold
    Error_test(k) = funLinreg(M2_data_train, area2_train, M2_data_test, area2_test);
    % Compute squared error with feature subset selection: only for the
    % final selection, the appropriate model.
    Error_train_fs(k) = funLinreg(M2_data_train(:,F), area2_train, M2_data_train(:,F), area2_train);
    Error_test_fs(k) = funLinreg(M2_data_train(:,F), area2_train, M2_data_test(:,F), area2_test);            
   
    % Show variable selection history
    mfig(sprintf('(%d) Feature selection',k));
    I = size(H.In,1); % Number of iterations    
    subplot(1,2,1); % Plot error criterion
    plot(H.Crit);
    xlabel('Iteration');
    ylabel('Squared error (crossvalidation)');
    title('Value of error criterion');
    xlim([0 I]);
    subplot(1,2,2); % Plot feature selection sequence
    bmplot(attributeNames_M2, 1:I, H.In');
    title('Attributes selected');
    xlabel('Iteration');
     
end
fprintf('\n');
fprintf('Linear regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Gene error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
Error_gene = sum(Error_test)/sum(CV.TestSize)
% Show the selected features
mfig('Attributes'); clf;
bmplot(attributeNames_M2, 1:K, Features');
xlabel('Crossvalidation fold');
ylabel('Attribute');
title('Attributes selected');
%% Plot the residuals vs attributes and regression with the Matrix
% Inspect selected feature coefficients effect on the entire dataset and
% plot the fitted modeld residual error as function of each attribute to
% inspect for systematic structure in the residual
 % cross-validation fold to inspect
ff=find(sum(Features,1)>2);
Forward_selec = M2_data(:,ff);
attributeNames_Forward= attributeNames_M2(ff);
M=Forward_selec;
%Initialization cross-validation for regression
K=10;
CV = cvpartition(N, 'Kfold',K);
w=nan(size(M,2)+1,K);
Error_train=nan(K,1);
Error_test=nan(K,1);
w_temp=nan(size(M,2)+1);
 %Cross validation of the regression
 for k = 1:K % For each crossvalidation fold
     fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
 
     % Extract training and test set
     X_train = M(CV.training(k), :);
     y_train = area2(CV.training(k));
     X_test = M(CV.test(k), :);
     y_test = area2(CV.test(k));
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est=glmval(w_temp,X_test,'identity');
    
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
 end
 Error_gene_feat = sum(Error_train)/sum(CV.TrainSize) ;
 Error_train_feat =  sum(Error_test)/sum(CV.TestSize);
  fprintf('- Generalized error with attributes selected:     %8.2f\n', Error_gene_feat)    
  fprintf('- average Train error:     %8.2f\n', Error_train_feat)   
  
%Let's take the last fold to plot the regression
y_est=[y_train_est;y_test_est]; %the last one per default

mfig(['Regression on attributes selected'])
plot(lines,y_est,lines,area3);
xlabel('Observations','FontWeight','bold');
ylabel('Area');
legend('Estimations', 'Real data')


residual=area2-y_est;
mfig(['Residual error vs. Attributes for features selected in cross-validation fold' num2str(k)]); clf;
for k=1:length(ff)
   subplot(2,ceil(length(ff)/2),k);
   plot(Forward_selec(:,k),residual,'.');
   xlabel(attributeNames_Forward(k),'FontWeight','bold');
   ylabel('residual error','FontWeight','bold');
end

%% Linear regression with MET and FWI
K=5;
M=zscore(MET);
% Create holdout crossvalidation partition
CV = cvpartition(N, 'Kfold',K);
w=nan(size(M,2)+1,K);
Error_train=nan(K,1);
Error_test=nan(K,1);
w_temp=nan(size(M,2)+1);
 %Cross validation of the regression
 for k = 1:K % For each crossvalidation fold
     fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
 
     % Extract training and test set
     X_train = M(CV.training(k), :);
     y_train = area2(CV.training(k));
     X_test = M(CV.test(k), :);
     y_test = area2(CV.test(k));
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est=glmval(w_temp,X_test,'identity');
    
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
 end
 Error_gene_MET = sum(Error_train)/sum(CV.TrainSize) ;
 Error_train_MET =  sum(Error_test)/sum(CV.TestSize);
 w_mean_MET = mean(w,2)
  fprintf('- Generalized error with MET attributes:     %8.2f\n', Error_gene_MET)    
  fprintf('- average Train error:     %8.2f\n', Error_train_MET)   
 
%Let's take the last fold to plot the regression
y_est=[y_train_est;y_test_est]; %the last one per default
area3=[area2_train;area2_test];
residual=area3-y_est;
mfig(['Regression on attributes selected'])
plot(lines,y_est,lines,area3);
xlabel('Sample','FontWeight','bold');
ylabel('Area');
legend('Estimations', 'Real data')
axis([0 500 1 3])

mfig(['Residual error vs. Attributes for features selected in cross-validation fold' num2str(k)]); clf;
for k=1:size(M,2)
   subplot(2,ceil(size(M,2)/2),k);
   plot(M(:,k),residual,'.');
   xlabel(attributeNames_met(k),'FontWeight','bold');
   ylabel('residual error','FontWeight','bold');
end

%% Linear regression for FWI
M=FWI;

% Create Kfold crossvalidation partition
CV = cvpartition(N, 'Kfold',K);
w=nan(size(M,2)+1,K);
Error_train=nan(K,1);
Error_test=nan(K,1);
w_temp=nan(size(M,2)+1);
 %Cross validation of the regression
 for k = 1:K % For each crossvalidation fold
     fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
 
     % Extract training and test set
     X_train = M(CV.training(k), :);
     y_train = area2(CV.training(k));
     X_test = M(CV.test(k), :);
     y_test = area2(CV.test(k));
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est=glmval(w_temp,X_test,'identity');
    
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
 end
 Error_gene_FWI = sum(Error_train)/sum(CV.TrainSize) ;
 Error_train_FWI =  sum(Error_test)/sum(CV.TestSize);
  fprintf('- Generalized error with FWI attributes:     %8.2f\n', Error_gene_FWI)    
  fprintf('- average Train error FWI:     %8.2f\n', Error_train_FWI)   
   w_mean_FWI = mean(w,2)

%Let's take the last fold to plot the regression

y_est=[y_train_est;y_test_est]; %the last one per default
area3=[area2_train;area2_test];

mfig(['Regression on attributes FWI'])
plot(lines,y_est,lines,area3);
xlabel('Sample','FontWeight','bold');
ylabel('Area');
legend('Estimations', 'Real data')

%% STFWI:using spatial, temporal and the four FWI
M=STFWI;
% Create Kfold crossvalidation partition
CV = cvpartition(N, 'Kfold',K);
w=nan(size(M,2)+1,K);
Error_train=nan(K,1);
Error_test=nan(K,1);
w_temp=nan(size(M,2)+1);
 %Cross validation of the regression
 for k = 1:K % For each crossvalidation fold
     fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
 
     % Extract training and test set
     X_train = M(CV.training(k), :);
     y_train = area2(CV.training(k));
     X_test = M(CV.test(k), :);
     y_test = area2(CV.test(k));
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est=glmval(w_temp,X_test,'identity');
    
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
 end
 Error_gene_STFWI = sum(Error_train)/sum(CV.TrainSize) ;
 Error_train_STFWI =  sum(Error_test)/sum(CV.TestSize);
  fprintf('- Generalized error with STFWI attributes:     %8.2f\n', Error_gene_STFWI)    
  fprintf('- average Train error STFWI:     %8.2f\n', Error_train_STFWI)   
  w_mean_STFWI = mean(w,2)
%Let's take the last fold to plot the regression
y_est=[y_train_est;y_test_est]; %the last one per default
area3=[area2_train;area2_test];

mfig(['Regression on attributes STFWI'])
plot(lines,y_est,lines,area3);
xlabel('Sample','FontWeight','bold');
ylabel('Area');
legend('Estimations', 'Real data')
%% STM with the spatial, temporal and four weather variables; 
M=STM;

% Create Kfold crossvalidation partition
CV = cvpartition(N, 'Kfold',K);
w=nan(size(M,2)+1,K);
Error_train=nan(K,1);
Error_test=nan(K,1);
w_temp=nan(size(M,2)+1);
 %Cross validation of the regression
 for k = 1:K % For each crossvalidation fold
     fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
 
     % Extract training and test set
     X_train = M(CV.training(k), :);
     y_train = area2(CV.training(k));
     X_test = M(CV.test(k), :);
     y_test = area2(CV.test(k));
    w(:,k)=glmfit(X_train, y_train) ;
    w_temp=w(:,k);
    y_train_est= glmval(w_temp,X_train,'identity'); %Regression based on the features selected matrix
    y_test_est=glmval(w_temp,X_test,'identity');
    
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
 end
 Error_gene_STM = sum(Error_train)/sum(CV.TrainSize) ;
 Error_train_STM =  sum(Error_test)/sum(CV.TestSize);
  fprintf('- Generalized error with STM attributes:     %8.2f\n', Error_gene_STM)    
  fprintf('- average Train error FWI:     %8.2f\n', Error_train_STM)   
  
%Let's take the last fold to plot the regression
y_est=[y_train_est;y_test_est]; %the last one per default
area3=[area2_train;area2_test];

mfig(['Regression on attributes selected'])
plot(lines,y_est,lines,area3);
xlabel('Sample','FontWeight','bold');
ylabel('Area');
legend('Estimations', 'Real data')


%% Wrap-up  model selection
fprintf('Test error select attrib: %8.2f\n', Error_gene_feat)    
fprintf('Test error MET %8.2f\n', Error_gene_MET)
fprintf('Test error FWI  %8.2f\n', Error_gene_FWI)
fprintf('Test error STWFI %8.2f\n', Error_gene_STFWI)
fprintf('Test error STM  %8.2f\n', Error_gene_STM)

Error_gene_all = [Error_gene,Error_gene_feat,Error_gene_STFWI, Error_gene_STM,Error_gene_MET,Error_gene_FWI];
Error_train_all=[Error_trn,Error_train_feat,Error_train_STFWI, Error_train_STM,Error_train_MET,Error_train_FWI];
Model={'All data','Selec feat','MET','FWI','STWFI','STM'}; %attention triche
mfig(['Error generalized and trained per model'])
subplot(1,2,1)
plot(1:6,Error_gene_all);
xlabel('Model','FontWeight','bold');
ylabel('Error');
xticks([1:6]);
xticklabels(Model);
axis([1 6 1  3])
title('Generalized error');
subplot(1,2,2)
plot(1:6,Error_train_all)
xlabel('Model','FontWeight','bold');
ylabel('Error');
axis([1 6 1  3])
xticks([1:6]);
xticklabels(Model);
title('Trained error');