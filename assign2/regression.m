%Take the appropriate matrix from project 1
remove_outliers_pro2



%How to deal with 1-on-K coding here ??? do a forward selection with 46 features ?



%% First regression with all features
% Create holdout crossvalidation partition
lines=1:size(M2_data,1);
K=10;
CV = cvpartition(lines, 'Holdout', K);
M=M2_data;
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
area2_estim = [area_train_estim;area_test_estim ];
area3=[area2_train;area2_test];


mfig('Area estimated barea2 regression and True Area'); clf;
plot(lines, area2_estim, '+',lines,area3,'o');
axis([0 500 -10 200])
xlabel('data');
ylabel('area estimated');
legend('estimation', 'true data');



%% forward attribute selection
colonne=1:size(M2_data,2);
%Simple layer
K=10;
M=length(colonne);
% Initialize variables
Features = nan(K,M);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_fs = nan(K,1);
Error_test_fs = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
CV = cvpartition(lines, 'Kfold',K);

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
    Histo(k,:) = H;
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
    bmplot(colonne, 1:I, H.In');
    title('Attributes selected');
    xlabel('Iteration');
    drawnow;    
end

Error_gene = 1/K*sum(Error_test_fs(k));
% Show the selected features
mfig('Attributes'); clf;
bmplot(colonne, 1:K, Features');
xlabel('Crossvalidation fold');
ylabel('Attribute');
title('Attributes selected');
%% Plot the residuals vs attributes
% Inspect selected feature coefficients effect on the entire dataset and
% plot the fitted modeld residual error as function of each attribute to
% inspect for systematic structure in the residual
k=1; % cross-validation fold to inspect
ff=find(Features(k,:));
w=glmfit(M2_data(:,ff), area2) ;

y_est= glmval(w,M2_data(:,ff),'identity');
residual=area2-y_est;
mfig(['Residual error vs. Attributes for features selected in cross-validation fold' num2str(k)]); clf;
for k=1:length(ff)
   subplot(2,ceil(length(ff)/2),k);
   plot(M2_data(:,ff(k)),residual,'.');
   xlabel({colonne(ff(k))},'FontWeight','bold');
   ylabel('residual error','FontWeight','bold');
end

%% Attributes chosen to carry on 
% 1-5,19-26, 33-35, 37, 39, 42-45

%% Linear regression with MET and FWI
M=MET;
% Create holdout crossvalidation partition
lines=1:size(M,1);
K=10;
CV = cvpartition(lines, 'Holdout', K);
% Extract training and test set
M_train= M(CV.training, :);
M_test= M(CV.test,:);
area2_train= area2(CV.training,:);
area2_test= area2(CV.test,:);

%Manually
w_est=glmfit(M_train,area2_train);
area_train_estim = glmval(w_est, M_train, 'identity');
area_test_estim = glmval(w_est,M_test,'identity');
Error_MET= sum((area_test_estim - area2_test).^2)
area2_estim = [area_train_estim;area_test_estim ];
area3=[area2_train;area2_test];
mfig('Area estimated by meteorogical attributes'); clf;
plot(lines, area2_estim, '+',lines,area3,'o');
axis([0 500 -10 200])
xlabel('data');
ylabel('area estimated');
legend('estimation', 'true data');
fprintf('Test error MET', Error_MET)
%% Linear regression for FWI
M=FWI;

% Create holdout crossvalidation partition
lines=1:size(M,1);
K=10;
CV = cvpartition(lines, 'Holdout', K);
% Extract training and test set
M_train= M(CV.training, :);
M_test= M(CV.test,:);
area2_train= area2(CV.training,:);
area2_test= area2(CV.test,:);

%Manually
w_est=glmfit(M_train,area2_train);
area_train_estim = glmval(w_est, M_train, 'identity');
area_test_estim = glmval(w_est,M_test,'identity');
Error_FWI= sum((area_test_estim - area2_test).^2)
area2_estim = [area_train_estim;area_test_estim ];
area3=[area2_train;area2_test];
mfig('Area burnt estimated by FWI attributes'); clf;
plot(lines, area2_estim, '+',lines,area3,'o');
axis([0 500 -10 200])
xlabel('data');
ylabel('area estimated');
legend('estimation', 'true data');
fprintf('Test error FWI', Error_FWI)


