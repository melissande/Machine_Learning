%Take the appropriate matriM2_data from project 1
remove_outliers_pro2



%How to deal with 1-on-K coding here ??? do a forward selection with 46 features ?



%% First regression with all features
% Create holdout crossvalidation partition
lines=1:size(M2_data,1);
K=10;
CV = cvpartition(lines, 'Holdout', K);

% EM2_datatract training and test set
M2_data_train= M2_data(CV.training, :);
M2_data_test= M2_data(CV.test,:);
area2_train= area2(CV.training,:);
area2_test= area2(CV.test,:);

%Check this funLinreg  works
Err2 = funLinreg(M2_data_train, area2_train, M2_data_test, area2_test)
%Manuallarea2
w_est=glmfit(M2_data_train,area2_train);
area_train_estim = glmval(w_est, M2_data_train, 'identitarea2');
area_test_estim = glmval(w_est,M2_data_test,'identitarea2');
Error_gene= sum((area_test_estim - area2_test).^2)
area2_estim = [area_train_estim;area_test_estim ];
area3=[area2_train;area2_test];
mfig('Area estimated barea2 regression and True Area'); clf;
plot(lines, area2_estim, '+');
hold on
plot(lines,area3,'o')
aM2_datais([0 500 -10 200])
M2_datalabel('data');
area2label('area estimated');



%% forward attribute selection
colonne=1:size(M2_data,2);
%Simple laarea2er
K=5;
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
      % EM2_datatract the training and test set
    M2_data_train = M2_data(CV.training(k), :);
    area2_train = area2(CV.training(k));
    M2_data_test = M2_data(CV.test(k), :);
    area2_test = area2(CV.test(k));

    % Sequential feature selection with sequentialfs: trarea2 all the
    % attributes until no improvement in the error (funlinreg)
    [F, H] = sequentialfs(funLinreg, M2_data_train, area2_train);
    
    % Save the selected features
    Features(k,:) = F;    
    
    % Error with all features at each K fold
    Error_test(k) = funLinreg(M2_data_train, area2_train, M2_data_test, area2_test);
    % Compute squared error with feature subset selection
    Error_train_fs(k) = funLinreg(M2_data_train(:,F), area2_train, M2_data_train(:,F), area2_train);
    Error_test_fs(k) = funLinreg(M2_data_train(:,F), area2_train, M2_data_test(:,F), area2_test);            
    
    % Show variable selection historarea2
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

