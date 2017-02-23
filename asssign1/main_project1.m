%% Import data from text file.
% Script for importing data from the following text file:
%
%    /Users/melissande/Documents/MATLAB/02450_intro_ML/Td_1/forestfires.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2017/01/31 15:53:48

%% Initialize variables.
filename = 'forestfires.csv';
delimiter = ',';
startRow = 2;

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: text (%s)
%	column4: text (%s)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%s%s%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
X = dataArray{:, 1};
X=X(2:end,1);

Y = dataArray{:, 2};
Y=Y(2:end,1);

month1 = dataArray{:, 3};
month1=month1(2:end,1);

day1 = dataArray{:, 4};
day1=day1(2:end,1);

FFMC = dataArray{:, 5}; %Fine Fuel Moisture code= influences ignition and fire 
%spread - From surface layers of the ground
FFMC=FFMC(2:end,1);

DMC = dataArray{:, 6}; %Duff Moisture Code= influences fire intensity 
%spread - From deep layers of the ground
DMC=DMC(2:end,1);


DC = dataArray{:, 7}; %Drought Code= influences fire intensity 
%spread - From deep layers of the ground
DC=DC(2:end,1);

ISI = dataArray{:, 8};%Initial Spread Index= score that coorrelate with fire velocity
% spread
ISI=ISI(2:end,1);

temp = dataArray{:, 9};%temperature
temp=temp(2:end,1);


RH = dataArray{:, 10};%% relative humidity
RH=RH(2:end,1);

wind = dataArray{:, 11};
wind=wind(2:end,1);


rain = dataArray{:, 12};
rain=rain(2:end,1);

area1 = dataArray{:, 13};
area1=area1(2:end,1);




%% From str to Num

length_m1=length(month1);
month1_var=zeros(length_m1,1);

ind_j = find(cellfun('length',regexp(month1,'jan')) == 1);
month1_var(ind_j,1)=1;
ind_f = find(cellfun('length',regexp(month1,'feb')) == 1);
month1_var(ind_f,1)=2;
ind_m = find(cellfun('length',regexp(month1,'mar')) == 1);
month1_var(ind_m,1)=3;
ind_a = find(cellfun('length',regexp(month1,'apr')) == 1);
month1_var(ind_a,1)=4;
ind_ma = find(cellfun('length',regexp(month1,'may')) == 1);
month1_var(ind_ma,1)=5;
ind_ju = find(cellfun('length',regexp(month1,'jun')) == 1);
month1_var(ind_ju,1)=6;
ind_jul = find(cellfun('length',regexp(month1,'jul')) == 1);
month1_var(ind_jul,1)=7;
ind_au = find(cellfun('length',regexp(month1,'aug')) == 1);
month1_var(ind_au,1)=8;
ind_sep = find(cellfun('length',regexp(month1,'sep')) == 1);
month1_var(ind_sep,1)=9;
ind_oct = find(cellfun('length',regexp(month1,'oct')) == 1);
month1_var(ind_oct,1)=10;
ind_nov = find(cellfun('length',regexp(month1,'nov')) == 1);
month1_var(ind_nov,1)=11;
ind_dec = find(cellfun('length',regexp(month1,'dec')) == 1);
month1_var(ind_dec,1)=12;

length_d1=length(day1);
day1_var=zeros(length_d1,1);

mond = find(cellfun('length',regexp(day1,'mon')) == 1);
tues = find(cellfun('length',regexp(day1,'tue')) ==1);
wedn = find(cellfun('length',regexp(day1,'wed')) ==1);
thur = find(cellfun('length',regexp(day1,'thu')) ==1);
frid  = find(cellfun('length',regexp(day1,'fri')) ==1);
satu = find(cellfun('length',regexp(day1,'sat')) ==1);
sund = find(cellfun('length',regexp(day1,'sun')) ==1);
day1_var(mond,1)= 1;
day1_var(tues,1)= 2;
day1_var(wedn,1) = 3;
day1_var(thur,1) = 4;
day1_var(frid,1) = 5;
day1_var(satu,1) = 6;
day1_var(sund, 1) = 7;

%% Building Matrix

M_data=[X,Y,month1_var,day1_var,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area1];
nb_obs=length(X);
attributeNames = {'X','Y','month','day','FFMC','DMC','ISI','temp','RH','wind'...
    'RH','rain','area'};
n_att=length(attributeNames);

FWI = [FFMC,DMC,DC,ISI,temp,RH,wind,rain];
attributeNames_fwi = {'FFMC','DMC','ISI','temp','RH','wind'...
    'RH','rain'};
n_att_fwi=length(attributeNames_fwi);


M2_data=[X,Y,month1_var,day1_var,FFMC,DMC,DC,ISI,temp,RH,wind,rain];
attributeNames_2 = {'X','Y','month','day','FFMC','DMC','ISI','temp','RH','wind'...
    'RH','rain'};
n_att_2=length(attributeNames_2);
%% Test display of data

month_cat = categorical(month1,...
     {'jan' 'feb' 'mar' 'apr' 'may' 'jun' 'jul' 'aug' 'sep' 'oct' 'nov' 'dec'},'Ordinal',true);
day_cat = categorical(day1,...
    {'mon' 'tue' 'wed' 'thu' 'fri' 'sat' 'sun'},'Ordinal',true);

 summary(month_cat)
% figure;
% histogram(month_cat);
% title('Months');

summary(day_cat)
% figure;
% histogram(day_cat);
% title('Days');

%% Test 2 display of data
jan_days=day1(month_cat=='jan');
feb_days=day1(month_cat=='feb');
mar_days=day1(month_cat=='mar');
apr_days=day1(month_cat=='apr');
may_days=day1(month_cat=='may');
jun_days=day1(month_cat=='jun');
jul_days=day1(month_cat=='jul');
aug_days=day1(month_cat=='aug');
sep_days=day1(month_cat=='sep');
oct_days=day1(month_cat=='oct');
nov_days=day1(month_cat=='nov');
dec_days=day1(month_cat=='dec');




%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;