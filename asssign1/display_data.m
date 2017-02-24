%% Run main_project1.m
%% Run remove_outliers.m

%% Display some data without outliers

%% Plot burnt area depending on months and days


month_wo=M_data(:,3);
area_wo=M_data(:,13);

months_un=unique(month_wo);
area_month=zeros(length(months_un),1);


for i=1:length(months_un)
    ind=find(month_wo==i);
    area_month(i,1)=sum(area_wo(ind,1));
    
end

figure(1);
subplot(1,2,1);

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
xlabel('Day from Mon to Sun')
ylabel('Summed Area in ha')

figure(2)
subplot(1,2,1)
histogram2(month_wo,area_wo)
title('2D histograms for area and months')
xlabel('Months from Jan to Dec ')
ylabel('Area burnt in ha')
subplot(1,2,2)
histogram2(day_wo,area_wo)
title('2D histograms for area and days')
xlabel('Days from Mon to Sun ')
ylabel('Area burnt in ha')

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


figure(2)
surf(Xm,Ym,Fout);
shading flat
title('Map of the Summed burned area')
colormap jet;
% % 


%% Scatter plots to see which data are correlated
% we remove spatio temporal data
% and area burnt so we use FWI dataset


FWI_std=zscore(FWI);

nb=(n_att_fwi^2-n_att_fwi)/2;
% mfig('Comparative scatter plots (standardized data) for meteorigal indexes'); clf;


for m = 1:n_att_fwi-1

for k=1:m-1
% subplot(n_att_fwi,n_att_fwi,n_att_fwi*(m-1)+k);
figure;
plot(FWI_std(:,k),FWI_std(:,m+1),'r.','MarkerSize',12)
% set(ax,'XScale','log');
d=attributeNames_fwi{m+1};
i=attributeNames_fwi{k};
str=strcat(i,' vs ',d);
title(str);
xlabel(i);
ylabel(d);
end


end

%%The correlated ones appear to be:
% FFMC and ISI
% FFMC and temperature, DMC and temperature
% FFMC and RH a bit




%% Area burnt vs meteorological indexes

% mfig('Area burnt in ha vs meteorological indexes'); clf;



for m = 1:n_att_fwi
    
    
    
data_wo=FWI(:,m);


% data_un=unique(data_wo);
% area_wo_data=zeros(length(data_un),1);


% for i=1:length(data_un)
%     ind=find(data_wo==data_un(i,1));
%     area_wo_data(i,1)=sum(area_wo(ind,1));
%     
% end
 figure;
%subplot(1,n_att_fwi,m);

% plot(data_un,area_wo_data);
% str=strcat('Summed Area burnt/',attributeNames_fwi{m});
% title(str)
% xlabel(attributeNames_fwi{m})
% ylabel('Summed Area in ha')

histogram2(data_wo,area_wo)
str=strcat(' 2D histograms for Area burnt and ',attributeNames_fwi{m});
title(str)
xlabel(attributeNames_fwi{m})
ylabel(' Area in ha')

    
end

%We see that:
%-for rain: of course really large burnt area when there is no rain
%- for wind: from 3 to 5 km/h really large burnt area. Funny as the stronger values are not when 
% the wind is stronger
%- for RH (humidity): large area burnt around 20-60 percent and it 
% decreases as the humidity increases which makes sense so it 
%can be quite humid and still have big area burnt, surprising
%- for ISI: from 5 to 12, big area burnt, can't say a lot
%- for temp: after 15 degree bigger area burnt, so it doesn't need 
% to be that hot to launch a fire
%- for DMC: from 0 - 200, greater area burnt  but also lots of set to
% zero so it's not a compulsory factor but seems to help
%- for FFMC: from 90-94 greater area burnt but also lots of set to
% zero so it's not a compulsory factor but seems to help
%- for DC: quite spread around all the values, so we can't say anything

%

%% Month vs meteorological indexes

% mfig('Month  vs meteorological indexes'); clf;



for m = 1:n_att_fwi
    
    
    
data_wo=FWI(:,m);


% data_un=unique(data_wo);
% area_wo_data=zeros(length(data_un),1);


% for i=1:length(data_un)
%     ind=find(data_wo==data_un(i,1));
%     area_wo_data(i,1)=sum(area_wo(ind,1));
%     
% end
 figure;
%subplot(1,n_att_fwi,m);

% plot(data_un,area_wo_data);
% str=strcat('Summed Area burnt/',attributeNames_fwi{m});
% title(str)
% xlabel(attributeNames_fwi{m})
% ylabel('Summed Area in ha')

histogram2(data_wo,month_wo)
str=strcat(' 2D histograms for months and ',attributeNames_fwi{m});
title(str)
xlabel(attributeNames_fwi{m})
ylabel(' Months from Jan to Dec')

    
end

% - rain: Surprisingly,  it rains more during august and september, which is 
%supposed to be the period of time when we get more fires but it
% might be due to the data date acquisition.. weird doesn't make sens
% to use rain though? 
% - wind: It's way more windy from july to september but also in march !
% -temp:  high (15-30 corresponding to area burnt before) more in
% june-september)
% - ISI from 5-12 (when there is more burnt area considering the graphs before)
% from august to september meaning that the fire propagtes really
%fast at that period of time
%- FFMC values that lead to larger burnt area (90-94) (see graphs before
%) are again from august to september
% - DMC values that lead to larger burnt area (0-200) (see graphs before) are 
% again from august to september
% - DC: dryer (high values of DC) during august-september
% - RH: values that lead to larger burnt area (20-60 percent) (see graphs before) are 
% again from august to september

% Doens't make any sense to display in function of days of the week
% these meteorolical indexes


