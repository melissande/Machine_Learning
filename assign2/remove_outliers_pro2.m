%% Building Matrix
close all;
clear all;
run main_project1.m

%% OBERVATIONS OF THE DATASETS TO DETERMINE WHICH ATTRIBUTES HAVE OUTLIERS
%% Box plots and histogram Full Dataset



%Outliers to remove: area, rain, FFMC + ISI? (confirmed on boxplots and histograms)
%

%% Box plots FWI Dataset

%Outliers to remove: rain, FFMC + ISI?


%% Box plots  Dataset 2 without area

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

area2=area1

%% Remove outliers for attribute rain for ALL DATASET
%Using histograms

rain2=M_data(:,12+Col);
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




%%  Remove outliers for attribute FFMC for ALL DATASET

ffmc2=M_data(:,5+Col);

% We can consider that under 85, we have outliers

idxoutlier = find(ffmc2<85); % 85 is chosen 
M_data(idxoutlier,:) = []; %matrix with all the data without outliers
FWI(idxoutlier,:) = []; %matrix with only meteorological indexes without outliers
M2_data(idxoutlier,:)=[]; %matrix with all the attributes except area burnt
N = length(ffmc2)-length(idxoutlier);
ffmc3= ffmc2;
ffmc3(idxoutlier) = [];

%%  Remove outliers for attribute ISI for ALL DATASET

ISI2=M_data(:,8+Col);
% There is only one value of ISI completely cazy (56.1 )
%We can consider that over 56, we have outliers

idxoutlier = find(ISI2>56); % 56 is chosen 
M_data(idxoutlier,:) = []; %matrix with all the data without outliers
FWI(idxoutlier,:) = []; %matrix with only meteorological indexes without outliers
M2_data(idxoutlier,:)=[]; %matrix with all the attributes except area burnt
N = length(ISI2)-length(idxoutlier);
ISI3= ISI2;
ISI3(idxoutlier) = [];


%% Display histograms without outliers

%normal distributed: temp,ffmc,isi,rh 
% maybe also normal distributed but less beautiful:wind and DMC 
