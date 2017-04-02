%% Classif perf 
close all;
clear all;
clc;
run main_project1.m
run remove_outliers_pro2.m

%% definition of class names and preparation
area3= M_data(:,end);
classNames={'small','medium','large'}';

range_S_area = [min(area3), median(area3)] 
range_M_area = [prctile(area3,50), prctile(area3,75)]
range_L_area = [prctile(area3,75) prctile(area3,100)]

y=area3;
y((min(range_S_area)<=area3) & (area3<=max(range_S_area)))=0;
y((min(range_M_area)<=area3) & (area3<=max(range_M_area)))=1;
y((min(range_L_area)<=area3) & (area3<=max(range_L_area)))=2;

%% Full dataset - Method comparison


X= M2_data;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);


% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune_max =15;
prune=1:prune_max;

% Parameters for K neighbor classifier
Distance = 'euclidean'; % Distance measure
Lmax = 40; % Maximum number of neighbors

% Variable for performance errors

perf_bayes=nan(K1,3);

perf_dt=nan(K1,3);
dep_dt=nan(K1,1);

perf_knn=nan(K1,3);
neigh_knn=nan(K1,1);

perf_largest_cl=nan(K1,3); %col1=error, col2=precision, col3=recall


for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    %BAYES
   
    
     % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
  
    
    % Performance
    perf_bayes(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_bayes(k1,2) =(sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_bayes(k1,3) =(sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
    %%DECISION TREE
    Error_dt=nan(prune_max,1);
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_M2, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:prune_max % For each pruning level
        Error_dt(n,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
    [val_Min,prune_fin]=min(Error_dt);
    dep_dt(k1,1)=prune_fin;
  
        
    
      % Performance
    perf_dt(k1,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)))/length(y_test); % Error
    perf_dt(k1,2) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test_est==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test_est==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_dt(k1,3) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test==2))/3; % Recall
    
    %KNN 
    Error_knn=nan(Lmax,1);
     for l = 1:Lmax % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error_knn(l,1) = sum(y_test~=y_test_est); % Count the number of errors
     end
    [val_Min,L_fin]=min(Error_knn);
    neigh_knn(k1,1)=L_fin;
    
    %performance
    perf_knn(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_knn(k1,2) = (sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_knn(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
    %%Largest Class
    
    [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV1.TestSize(1,k1),1);
    perf_largest_cl(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_largest_cl(k1,2) = sum(y_test==y_test_est & y_test==larg_class)/sum(y_test_est==larg_class); % Precision
    perf_largest_cl(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
end



figure(1)
subplot(1,3,1)
plot(1:K1,perf_bayes(:,1),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,1),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,1),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,1),'y.','MarkerSize',12);
hold off;

legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Error')
title('Error')


subplot(1,3,2)
plot(1:K1,perf_bayes(:,2),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,2),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,2),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,2),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Precision')
title('Precision')


subplot(1,3,3)
plot(1:K1,perf_bayes(:,3),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,3),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,3),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,3),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Recall')
title('Recall')


%% FWI - Method comparison


X= FWI;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);
% K-fold crossvalidation inner loop
    K2 = 10;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune_max =15;
prune=1:prune_max;

% Parameters for K neighbor classifier
Distance = 'euclidean'; % Distance measure
Lmax = 40; % Maximum number of neighbors

% Variable for performance errors

perf_bayes=nan(K1,3);

perf_dt=nan(K1,3);
dep_dt=nan(K1,1);

perf_knn=nan(K1,3);
neigh_knn=nan(K1,1);

perf_largest_cl=nan(K1,3); %col1=error, col2=precision, col3=recall


for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    %BAYES
   
    
     % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
   
   
    % Performance
    perf_bayes(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_bayes(k1,2) =(sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_bayes(k1,3) =(sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
    %%DECISION TREE
    Error_dt=nan(prune_max,1);
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_fwi, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:prune_max % For each pruning level
        Error_dt(n,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
    [val_Min,prune_fin]=min(Error_dt);
    dep_dt(k1,1)=prune_fin;
  
        
    
      % Performance
    perf_dt(k1,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)))/length(y_test); % Error
    perf_dt(k1,2) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test_est==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test_est==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_dt(k1,3) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test==2))/3; % Recall
    
    %KNN 
    Error_knn=nan(Lmax,1);
     for l = 1:Lmax % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error_knn(l,1) = sum(y_test~=y_test_est); % Count the number of errors
     end
    [val_Min,L_fin]=min(Error_knn);
    neigh_knn(k1,1)=L_fin;
    
    %performance
    perf_knn(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_knn(k1,2) = (sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_knn(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    %%Largest Class
    
    [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV1.TestSize(1,k1),1);
    perf_largest_cl(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_largest_cl(k1,2) = sum(y_test==y_test_est & y_test==larg_class)/sum(y_test_est==larg_class); % Precision
    perf_largest_cl(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall

end


figure(2)
subplot(1,3,1)
plot(1:K1,perf_bayes(:,1),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,1),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,1),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,1),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Error')
title('Error')


subplot(1,3,2)
plot(1:K1,perf_bayes(:,2),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,2),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,2),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,2),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Precision')
title('Precision')


subplot(1,3,3)
plot(1:K1,perf_bayes(:,3),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,3),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,3),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,3),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Recall')
title('Recall')


%% STM - Method comparison


X= STM;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);
% K-fold crossvalidation inner loop
    K2 = 10;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune_max =15;
prune=1:prune_max;

% Parameters for K neighbor classifier
Distance = 'euclidean'; % Distance measure
Lmax = 40; % Maximum number of neighbors

% Variable for performance errors

perf_bayes=nan(K1,3);

perf_dt=nan(K1,3);
dep_dt=nan(K1,1);

perf_knn=nan(K1,3);
neigh_knn=nan(K1,1);

perf_largest_cl=nan(K1,3); %col1=error, col2=precision, col3=recall


for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    %BAYES
   
    
     % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
   
   
    % Performance
    perf_bayes(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_bayes(k1,2) =(sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_bayes(k1,3) =(sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
    %%DECISION TREE
    Error_dt=nan(prune_max,1);
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_stm, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:prune_max % For each pruning level
        Error_dt(n,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
    [val_Min,prune_fin]=min(Error_dt);
    dep_dt(k1,1)=prune_fin;
  
        
    
      % Performance
    perf_dt(k1,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)))/length(y_test); % Error
    perf_dt(k1,2) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test_est==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test_est==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_dt(k1,3) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test==2))/3; % Recall
    
    %KNN 
    Error_knn=nan(Lmax,1);
     for l = 1:Lmax % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error_knn(l,1) = sum(y_test~=y_test_est); % Count the number of errors
     end
    [val_Min,L_fin]=min(Error_knn);
    neigh_knn(k1,1)=L_fin;
    
    %performance
    perf_knn(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_knn(k1,2) = (sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_knn(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
%%Largest Class
    
    [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV1.TestSize(1,k1),1);
    perf_largest_cl(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_largest_cl(k1,2) = sum(y_test==y_test_est & y_test==larg_class)/sum(y_test_est==larg_class); % Precision
    perf_largest_cl(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
end


figure(3)
subplot(1,3,1)
plot(1:K1,perf_bayes(:,1),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,1),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,1),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,1),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Error')
title('Error')


subplot(1,3,2)
plot(1:K1,perf_bayes(:,2),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,2),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,2),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,2),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Precision')
title('Precision')


subplot(1,3,3)
plot(1:K1,perf_bayes(:,3),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,3),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,3),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,3),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Recall')
title('Recall')


%% STFWI - Method comparison


X= STFWI;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);
% K-fold crossvalidation inner loop
    K2 = 10;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune_max =15;
prune=1:prune_max;

% Parameters for K neighbor classifier
Distance = 'euclidean'; % Distance measure
Lmax = 40; % Maximum number of neighbors

% Variable for performance errors

perf_bayes=nan(K1,3);

perf_dt=nan(K1,3);
dep_dt=nan(K1,1);

perf_knn=nan(K1,3);
neigh_knn=nan(K1,1);

perf_largest_cl=nan(K1,3); %col1=error, col2=precision, col3=recall


for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    %BAYES
   
    
     % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
   
   
    % Performance
    perf_bayes(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_bayes(k1,2) =(sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_bayes(k1,3) =(sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
    %%DECISION TREE
    Error_dt=nan(prune_max,1);
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_stfwi, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:prune_max % For each pruning level
        Error_dt(n,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
    [val_Min,prune_fin]=min(Error_dt);
    dep_dt(k1,1)=prune_fin;
  
        
    
      % Performance
    perf_dt(k1,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)))/length(y_test); % Error
    perf_dt(k1,2) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test_est==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test_est==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_dt(k1,3) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test==2))/3; % Recall
    
    %KNN 
    Error_knn=nan(Lmax,1);
     for l = 1:Lmax % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error_knn(l,1) = sum(y_test~=y_test_est); % Count the number of errors
     end
    [val_Min,L_fin]=min(Error_knn);
    neigh_knn(k1,1)=L_fin;
    
    %performance
    perf_knn(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_knn(k1,2) = (sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_knn(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
%%Largest Class
    
    [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV1.TestSize(1,k1),1);
    perf_largest_cl(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_largest_cl(k1,2) = sum(y_test==y_test_est & y_test==larg_class)/sum(y_test_est==larg_class); % Precision
    perf_largest_cl(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
end


figure(4)
subplot(1,3,1)
plot(1:K1,perf_bayes(:,1),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,1),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,1),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,1),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Error')
title('Error')


subplot(1,3,2)
plot(1:K1,perf_bayes(:,2),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,2),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,2),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,2),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Precision')
title('Precision')


subplot(1,3,3)
plot(1:K1,perf_bayes(:,3),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,3),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,3),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,3),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Recall')
title('Recall')



%% MET - Method comparison


X= MET;


% K-fold crossvalidation out loop
K1 = 10;
CV1 = cvpartition(y, 'Kfold', K1);
% K-fold crossvalidation inner loop
    K2 = 10;

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'empirical';

% Parameters for Decision Tree classifier
prune_max =15;
prune=1:prune_max;

% Parameters for K neighbor classifier
Distance = 'euclidean'; % Distance measure
Lmax = 40; % Maximum number of neighbors

% Variable for performance errors

perf_bayes=nan(K1,3);

perf_dt=nan(K1,3);
dep_dt=nan(K1,1);

perf_knn=nan(K1,3);
neigh_knn=nan(K1,1);

perf_largest_cl=nan(K1,3); %col1=error, col2=precision, col3=recall


for k1 = 1:K1 % For each crossvalidation fold
   % fprintf('Crossvalidation fold %d/%d\n', k1, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV1.training(k1), :);
    y_train = y(CV1.training(k1));
    X_test = X(CV1.test(k1), :);
    y_test = y(CV1.test(k1));
    
    %BAYES
   
    
     % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
   
   
    % Performance
    perf_bayes(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_bayes(k1,2) =(sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_bayes(k1,3) =(sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
    %%DECISION TREE
    Error_dt=nan(prune_max,1);
    T = classregtree(X_train, classNames(y_train+1), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames_met, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:prune_max % For each pruning level
        Error_dt(n,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune(n))));
    end    
    [val_Min,prune_fin]=min(Error_dt);
    dep_dt(k1,1)=prune_fin;
  
        
    
      % Performance
    perf_dt(k1,1) = sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)))/length(y_test); % Error
    perf_dt(k1,2) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test_est==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test_est==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_dt(k1,3) = (sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==0)/sum(y_test==0)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==1)/sum(y_test==1)+...
        sum(~strcmp(classNames(y_test+1), eval(T, X_test, prune_fin)) & y_test==2)/sum(y_test==2))/3; % Recall
    
    %KNN 
    Error_knn=nan(Lmax,1);
     for l = 1:Lmax % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error_knn(l,1) = sum(y_test~=y_test_est); % Count the number of errors
     end
    [val_Min,L_fin]=min(Error_knn);
    neigh_knn(k1,1)=L_fin;
    
    %performance
    perf_knn(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_knn(k1,2) = (sum(y_test==y_test_est & y_test==0)/sum(y_test_est==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test_est==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test_est==2))/3; % Precision
    perf_knn(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
    
%%Largest Class
    
    [larg_class,nb]=mode(y_train);
    y_test_est=larg_class*ones(CV1.TestSize(1,k1),1);
    perf_largest_cl(k1,1) = sum(y_test~=y_test_est)/length(y_test); % Error
    perf_largest_cl(k1,2) = sum(y_test==y_test_est & y_test==larg_class)/sum(y_test_est==larg_class); % Precision
    perf_largest_cl(k1,3) = (sum(y_test==y_test_est & y_test==0)/sum(y_test==0)+...
        sum(y_test==y_test_est & y_test==1)/sum(y_test==1)+...
        sum(y_test==y_test_est & y_test==2)/sum(y_test==2))/3; % Recall
end


figure(5)
subplot(1,3,1)
plot(1:K1,perf_bayes(:,1),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,1),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,1),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,1),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,1),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Error')
title('Error')


subplot(1,3,2)
plot(1:K1,perf_bayes(:,2),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,2),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,2),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,2),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,2),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Precision')
title('Precision')


subplot(1,3,3)
plot(1:K1,perf_bayes(:,3),'r.','MarkerSize',12);
hold on;
plot(1:K1,perf_dt(:,3),'g.','MarkerSize',12);
str_prec_dt = [num2str(dep_dt)];
text(1:K1',perf_dt(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_knn(:,3),'b.','MarkerSize',12);
str_prec_dt = [num2str(neigh_knn)];
text(1:K1',perf_knn(:,3),str_prec_dt,'HorizontalAlignment','left');
hold on;
plot(1:K1,perf_largest_cl(:,3),'y.','MarkerSize',12);
hold off;


legend('Naive Bayes', 'DT','KNN','Largest Class')
xlabel('Folds')
ylabel('Recall')
title('Recall')