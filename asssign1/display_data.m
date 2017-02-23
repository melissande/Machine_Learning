%% Run main_project1.m
%% Run remove_outliers.m

%% Display some data without outliers

%% Plot burnt area depending on months and days

figure(1);
subplot(1,2,1);
month_wo=M_data(:,3);
area_wo=M_data(:,13);

months_un=unique(month_wo);
area_month=zeros(length(months_un),1);


for i=1:length(months_un)
    ind=find(month_wo==i);
    area_month(i,1)=sum(area_wo(ind,1));
    
end

figure(1);
plot(months_un,area_month);
title('Area burnt/month')
xlabel('Month from Jan to Dec')
ylabel('Summed Area in ha')

subplot(1,2,2);

day_wo=M_data(:,4);
day_un=unique(day_wo);
area_day=zeros(length(day_un),1);


for i=1:length(day_un)
    ind=find(day_wo==i);
    area_day(i,1)=sum(area_wo(ind,1));
    
end


plot(day_un,area_day);
title('Area burnt/day')
xlabel('Month from Mon to Sun')
ylabel('Summed Area in ha')

%% Display burnt area depending by location


mat_loc=M_data(:,1:2);
mat_loc_un=unique(mat_loc,'rows');
size_mat_loc=size(mat_loc_un);
area_loc=zeros(size_mat_loc(1,1),1);

for i=1:size_mat_loc(1,1)
    
  
     [i_s,j_s] = ind2sub(size(mat_loc),find(ismember(mat_loc,mat_loc_un(i,:))));
     size(i_s)
     area_loc(i,1)=sum(area_wo(i_s,1));
    
end


% 

[Xm,Ym] = meshgrid(mat_loc_un(:,1),mat_loc_un(:,2));
Fout = griddata(mat_loc_un(:,1),mat_loc_un(:,2),area_loc,Xm,Ym);


figure(3)
surf(Xm,Ym,Fout);
shading flat
title('Map of the Summed burned area')
colormap jet;
% % 


%% Scatter plots to see which data are correlated
% we remove spatio temporal data
% and area burnt so we use FWI dataset

FWI = [FFMC,DMC,DC,ISI,temp,RH,wind,rain];
attributeNames_fwi = {'FFMC','DMC','DC','ISI','temp','RH','wind'...
    'rain'};
n_att_fwi=length(attributeNames_fwi);
FWI_std=zscore(FWI);

nb=(n_att_fwi^2-n_att_fwi)/2;
mfig('Comparative scatter plots (standardized data)'); clf;


for m = 1:n_att_fwi-1

for k=1:m-1
subplot(n_att_fwi,n_att_fwi,n_att_fwi*(m-1)+k);
plot(FWI_std(:,m+1),FWI_std(:,k),'r.','MarkerSize',12)
d=attributeNames_fwi{m};
i=attributeNames_fwi{k};
str=strcat(d,' vs ',i);
title(str);
end


end