%% Load Data
clear all
close all
clc

run main_project1.m

%% %%%%%%%%%%%%%%%%%%%%%%% Clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run remove_outliers_pro2.m   %on devrait d�s le d�but clusteriser sur le dataset entier
%% Normalization 
zscore(MET);
zscore(FWI);
zscore(M2_data);
zscore(STM);
zscore(STFWI);
%% Generate labels
area3= log(M_data(:,end)+1);
% 2 classes


classNames2={'small','large'}';

range_S_area = [min(area3), median(area3)] ;
range_L_area = [prctile(area3,50) prctile(area3,100)];

y_2=area3;
y_2((min(range_S_area)<=area3) & (area3<=max(range_S_area)))=0;
y_2((min(range_M_area)<=area3) & (area3<=max(range_M_area)))=1;
y_2((min(range_L_area)<=area3) & (area3<=max(range_L_area)))=2;

% 3 classes



classNames3={'small','medium','large'}';

range_S_area = [min(area3), median(area3)] ;
range_M_area = [prctile(area3,50), prctile(area3,75)];
range_L_area = [prctile(area3,75) prctile(area3,100)];

y_3=area3;
y_3((min(range_S_area)<=area3) & (area3<=max(range_S_area)))=0;
y_3((min(range_M_area)<=area3) & (area3<=max(range_M_area)))=1;
y_3((min(range_L_area)<=area3) & (area3<=max(range_L_area)))=2;



%% GMM
%% Full Dataset
Kmax=5;
GMM_k_sel(M2_data,'Full Dataset',Kmax);
%Read the graphs and select K

K_M2=3;
[i_M2,X_c_M2,Sigma_c_M2  ] = GMM_perform( K_M2,M2_data,'Full Dataset',y_2,classNames2);
%% FWI 
Kmax=10;
GMM_k_sel((FWI),'FWI',Kmax);

%Read the graphs and select K
K_FWI=9;
[i_FWI,X_c_FWI,Sigma_c_FWI] = GMM_perform( K_FWI,FWI,'FWI',y_3,classNames3);

% We have got a result
%% STFWI
Kmax=5;
GMM_k_sel(STFWI,'STFWI',Kmax);
%Read the graphs and select K
%OR
K_STFWI=size(X_c,2);
K_STFWI=2;
[i_STFWI,X_c_STFWI,Sigma_c_STFWI  ] = GMM_perform( K_STFWI,STFWI,'STFWI',y_2,classNames2);


%% STM
Kmax=5;
GMM_k_sel(STM,'STM',Kmax);
%Read the graphs and select K
K_STM=2;
[i_STM,X_c_STM,Sigma_c_STM  ] = GMM_perform( K_STM,STM,'STM',y_2,classNames2);


%% MET
Kmax=8;
GMM_k_sel(MET,'MET',Kmax);
%Read the graphs and select K
K_MET=3;
[i_MET,X_c_MET,Sigma_c_MET ] = GMM_perform( K_MET,MET,'MET',y_2,classNames2);
%% Hierarchical Clustering
%% Full Dataset
Kmax=5;
[Z_M2,i_M2 ] = hierarch_clust( Kmax,M2_data,'Full Dataset' );

K_co=K_M2;
% Plot data
i_M2_co = cluster(Z_M2, 'Maxclust', K_co);
mfig('Full Dataset Hierarchical'); clf; 
clusterplot(M2_data, y_2, i_M2_co);
legend(classNames2)
%% FWI
Kmax=5;
[Z_FWI,i_FWI ] = hierarch_clust( Kmax,FWI,'FWI' );

K_co=K_FWI;
% Plot data
i_FWI_co = cluster(Z_FWI, 'Maxclust', K_co);
mfig('FWI Hierarchical'); clf; 
clusterplot(FWI, y_3, i_FWI_co);
legend(classNames3)

%% STFWI
Kmax=5;
[Z_STFWI,i_STFWI ] = hierarch_clust( Kmax,STFWI,'STFWI' );

K_co=K_STFWI;
% Plot data
i_STFWI_co = cluster(Z_STFWI, 'Maxclust', K_co);
mfig('STFWI Hierarchical'); clf; 
clusterplot(STFWI, y_2, i_STFWI_co);
legend(classNames2)

%% STM
Kmax=5;
[Z_STM,i_STM ] = hierarch_clust( Kmax,STM,'STM' );

K_co=K_STM;
% Plot data
i_STM_co = cluster(Z_STM, 'Maxclust', K_co);
mfig('STM Hierarchical'); clf; 
clusterplot(STM, y_2, i_STM_co);
legend(classNames2)

%% MET
Kmax=5;
[Z_MET,i_MET ] = hierarch_clust( Kmax,MET,'MET' );

K_co=K_MET;
% Plot data
i_MET_co = cluster(Z_MET, 'Maxclust', K_co);
mfig('MET Hierarchical'); clf; 
clusterplot(MET, y_2, i_MET_co);
legend(classNames2)

%% %%%%%%%%%%%%%%%%%%%%%% Outlier detection %%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Data
clear all
close all
clc

run main_project1.m

%we don't run outlier remove as we want to discover the outliers



%% Full Dataset

%Kernel Density Estimator
kde_perform( M2_data,'Full Dataset' );

%KNN density estimator
K=12;
knn_density_perform(M2_data,K,'Full Dataset' );

%KNN average relative density
K=12;
knn_ard(M2_data,K,'Full Dataset' );


%% %%%%%%%%%%%%%%%%%%%%% Association mining %%%%%%%%%%%%%%%%%%%%%%%%%
%M�li... Je trouve que c'est un endroit appropri� pour te d�clarer ma
%flamme... Qu'elle puisse br�ler � jamais

%% MET: work du tonnerre !
X=MET;
attributeNames=attributeNames_met;
sup = 30;
conf= 60;
[asso_met,freq_met]=a_priori(X,attributeNames,sup,conf);
%% Full dataset: out of time
X=M2_data;
attributeNames=attributeNames_M2;
sup = 30;
conf= 60;
[asso_M,freq_M]=a_priori(X,attributeNames,sup,conf);

%% STM
X=STM;
attributeNames=attributeNames_stm;
sup = 30;
conf= 60;
[asso_stm,freq_stm]=a_priori(X,attributeNames,sup,conf);
%% FWI
X=FWI;
attributeNames=attributeNames_fwi;
sup = 30;
conf= 60;
[asso_fwi,freq_fwi]=a_priori(X,attributeNames,sup,conf);

