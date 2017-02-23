%% Building Matrix
close all;
clear all;
run main_project1.m

M_data=[X,Y,month1_var,day1_var,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area1];
nb_obs=length(X);
attributeNames = {'X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind'...
    'rain','area'};
n_att=length(attributeNames);

FWI = [FFMC,DMC,DC,ISI,temp,RH,wind,rain];
attributeNames_fwi = {'FFMC','DMC','DC','ISI','temp','RH','wind'...
    'rain'};
n_att_fwi=length(attributeNames_fwi);


M2_data=[X,Y,month1_var,day1_var,FFMC,DMC,DC,ISI,temp,RH,wind,rain];
attributeNames_2 = {'X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind'...
    'rain'};
n_att_2=length(attributeNames_2);

%% OBERVATIONS OF THE DATASETS TO DETERMINE WHICH ATTRIBUTES HAVE OUTLIERS
%% Box plots and histogram Full Dataset

mfig('Boxplots for each attribute and full dataset'); clf;
boxplot(M_data, attributeNames);
mfig('Boxplot for attribute area'); clf;
boxplot(area1);
mfig('Boxplots for each attribute and full dataset (standardized)');
boxplot(zscore(M_data), attributeNames, 'LabelOrientation', 'inline');

mfig('Histogram full dataset'); clf;
for m = 1:n_att
    u = floor(sqrt(n_att)); v = ceil(n_att/u);
    subplot(u,v,m);
	hist(M_data(:,m));
	xlabel(attributeNames{m});      
	axis square;
end

%Outliers to remove: area, rain, FFMC + ISI? (confirmed on boxplots and histograms)
%

%% Box plots FWI Dataset

mfig('Boxplots for each attribute and FWI dataset'); clf;
boxplot(FWI, attributeNames_fwi);
mfig('Boxplots for each attribute and FWI dataset (standardized)');
boxplot(zscore(FWI), attributeNames_fwi, 'LabelOrientation', 'inline');

mfig('Histogram full dataset'); clf;
for m = 1:n_att_fwi
    u = floor(sqrt(n_att_fwi)); v = ceil(n_att_fwi/u);
    subplot(u,v,m);
	hist(FWI(:,m));
	xlabel(attributeNames_fwi{m});      
	axis square;
end

%Outliers to remove: rain, FFMC + ISI?


%% Box plots  Dataset 2 without area

mfig('Boxplots for each attribute and dataset without area'); clf;
boxplot(M2_data, attributeNames_2);
mfig('Boxplots for each attribute and dataset without area (standardized)');
boxplot(zscore(M2_data), attributeNames_2, 'LabelOrientation', 'inline');


mfig('Histogram full dataset'); clf;
for m = 1:n_att_2
    u = floor(sqrt(n_att_2)); v = ceil(n_att_2/u);
    subplot(u,v,m);
	hist(M2_data(:,m));
	xlabel(attributeNames_2{m});      
	axis square;
end


%Outliers to remove: rain, FFMC +ISI?

%% REMOVE OUTLIERS TO THE DIFFERENT DATASETS


%% Remove outliers for attribute area 


% We observe that the median is 0.5, very close to 0. This means that most
% of the important data are outside the boxplot. To understand which are
% really outliers, it is interesting to divide the vector area_burnt in 3
% distinct vectors classifying the size of the fire.
range_S_area = [min(area1), median(area1)] 
range_M_area = [prctile(area1,50), prctile(area1,75)]
range_L_area = [prctile(area1,75) prctile(area1,100)]
S_area = area1(area1>min(area1) & area1< prctile(area1,50)) %small area burnt fires
M_area = area1(area1>range_M_area(1)& area1<range_M_area(2)) %medium area burnt fires
L_area = area1(area1>range_L_area(1)& area1<range_L_area(2)) % large area burnt fires


%By plotting the box of these vector, it is possible to define outliers

mfig('Boxplot area different size'); clf;
subplot (1,3,1,'replace')
boxplot(S_area)
subplot (1,3,2,'replace')
boxplot(M_area)
subplot (1,3,3,'replace')
boxplot(L_area)

%Remove the outliers according to the previous plot
idxoutlier = find(area1>100); % 100 is chosen regarding the L_area boxplot
% Finally we will remove these from the data set
M_data(idxoutlier,:) = []; %matrix with all the data without outliers
FWI(idxoutlier,:) = []; %matrix with only meteorological indexes without outliers
M2_data(idxoutlier,:)=[]; %matrix with all the attributes except area burnt
N = length(area1)-length(idxoutlier);
area2= area1;
area2(idxoutlier) = [];

%Plot again the box 

mfig('Boxplot Full Dataset for attribute area without outliers'); clf;
boxplot(M_data, attributeNames);
mfig('Boxplot Full Dataset for attribute area without outliers'); clf;
boxplot(area2);

%% Remove outliers for attribute rain for ALL DATASET
%Using histograms

rain2=M_data(:,12);
mfig('Boxplot and histogram Full Dataset for attribute rain'); clf;
subplot(1,2,1);
boxplot(rain2);
subplot(1,2,2);
hist(rain2,100);

%Remove the outliers according to the previous plot


% Above  1mm/m2 metter (so the first bin!) every thing is kind of
% considered as outlier because in this region it almost doesn't rain
% that's why we have such few quantity of rainfall. However the higher other
% values don't have to be considered as outleirs as it's is only reality
% and not outlier!BUT the highest value, 6.4 mm/m2 value seems quite crazy so we will remove
% it because it's super far away from the over values (the second higher one is 
%1.4 mm/m2)

%Remove the outliers according to the previous plot
idxoutlier = find(rain2>1.5); % (we need something over 1.4mm/m2)is chosen regarding the L_area boxplot
% Finally we will remove these from the data set
M_data(idxoutlier,:) = []; %matrix with all the data without outliers
FWI(idxoutlier,:) = []; %matrix with only meteorological indexes without outliers
M2_data(idxoutlier,:)=[]; %matrix with all the attributes except area burnt
N = length(rain2)-length(idxoutlier);
rain3= rain2;
rain3(idxoutlier) = [];


mfig('Boxplot and histogram Full Dataset for attribute rain without outliers'); clf;
subplot(1,3,1);
boxplot(zscore(M_data), attributeNames);
subplot(1,3,2);
boxplot(rain3);
subplot(1,3,3);
histogram(rain3,100);




%%  Remove outliers for attribute FFMC for ALL DATASET

ffmc2=M_data(:,5);
mfig('Boxplot and histogram Full Dataset for attribute FFMC'); clf;
subplot(1,2,1);
boxplot(ffmc2);
subplot(1,2,2);
hist(ffmc2);

% We can consider that under 85, we have outliers

idxoutlier = find(ffmc2<85); % 85 is chosen 
M_data(idxoutlier,:) = []; %matrix with all the data without outliers
FWI(idxoutlier,:) = []; %matrix with only meteorological indexes without outliers
M2_data(idxoutlier,:)=[]; %matrix with all the attributes except area burnt
N = length(ffmc2)-length(idxoutlier);
ffmc3= ffmc2;
ffmc3(idxoutlier) = [];


mfig('Boxplot and histogram Full Dataset for attribute FFMC  without outliers'); clf;
subplot(1,3,1);
boxplot(zscore(M_data), attributeNames);
subplot(1,3,2);
boxplot(ffmc3);
subplot(1,3,3);
histogram(ffmc3,100);


%%  Remove outliers for attribute ISI for ALL DATASET

ISI2=M_data(:,8);
mfig('Boxplot and histogram Full Dataset for attribute ISI'); clf;
subplot(1,2,1);
boxplot(ISI2);
subplot(1,2,2);
hist(ISI2);

% There is only one value of ISI completely cazy (56.1 )
%We can consider that over 56, we have outliers

idxoutlier = find(ISI2>56); % 56 is chosen 
M_data(idxoutlier,:) = []; %matrix with all the data without outliers
FWI(idxoutlier,:) = []; %matrix with only meteorological indexes without outliers
M2_data(idxoutlier,:)=[]; %matrix with all the attributes except area burnt
N = length(ISI2)-length(idxoutlier);
ISI3= ISI2;
ISI3(idxoutlier) = [];


mfig('Boxplot and histogram Full Dataset for attribute ISI  without outliers'); clf;
subplot(1,3,1);
boxplot(zscore(M_data), attributeNames);
subplot(1,3,2);
boxplot(ISI3);
subplot(1,3,3);
histogram(ISI3,100);

%% Display histograms without outliers


mfig('Histogram full dataset'); clf;
for m = 1:n_att
    u = floor(sqrt(n_att)); v = ceil(n_att/u);
    subplot(u,v,m);
	hist(M_data(:,m));
	xlabel(attributeNames{m});      
	axis square;
end

%normal distributed: temp
% maybe also normal distributed but less beautiful: ffmc,isi,rh wind
