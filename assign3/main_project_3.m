%% Load Data
clear all
close all
clc

run main_project1.m

%% %%%%%%%%%%%%%%%%%%%%%%% Clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run remove_outliers_pro2.m   %on devrait dès le début clusteriser sur le dataset entier
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
%Méli... Je trouve que c'est un endroit approprié pour te déclarer ma
%flamme... Qu'elle puisse brûler à jamais  

%In order to do an association mining with the full dataset, first MET and
%FWI have to be binarized and then associated with ST, it is not possible
%to use binarize.m if there are attributes already binary apparently.
%% MET: work du tonnerre !
X=MET;
attributeNames=attributeNames_met;
sup = 30;
conf= 60;
%That's why I take back the binary matrix generated in order to reuse it
[asso_met,freq_met,X_bin_met]=a_priori(X,attributeNames,sup,conf);
%% MET+area !
X=[MET,area1];
attributeNames=[attributeNames_met,'area'];
sup = 30;
conf= 60;
[asso_met_ar,freq_met_ar,X_bin_met_ar]=a_priori(X,attributeNames,sup,conf);
%% FWI
X=FWI;
attributeNames=attributeNames_fwi;
sup = 30;
conf= 60;
[asso_fwi,freq_fwi,X_bin_fwi]=a_priori(X,attributeNames,sup,conf);
%% Full dataset with area: works baby
M_binary=[X_bin,Y_bin,day_bin,month_bin,X_bin_fwi,X_bin_met_ar];
sup = 48;
conf= 60;
writeAprioriFile(M_binary,'m_full_binary.txt');

%Apriori is used here 
[asso_M,freq_M]=apriori('m_full_binary.txt',sup,conf);
%item 52: Few rain should be erased because is polluting the results, being
%always here, always frequent and confident with the rest.
%Otherwise 38<->46; 46<->49; 44<->38


%% STM: better without FWIs which are difficult to interpret
STM_binary=[X_bin,Y_bin,day_bin,month_bin,X_bin_met_ar];
sup = 50;
conf= 70;
writeAprioriFile(STM_binary,'stm_binary.txt');

%Apriori is used here 
[asso_STM,freq_STM]=apriori('stm_binary.txt',sup,conf);


