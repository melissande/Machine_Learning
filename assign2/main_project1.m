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
% - From deep layers of the ground
DMC=DMC(2:end,1);


DC = dataArray{:, 7}; %Drought Code= influences fire intensity 
%- From deep layers of the ground
DC=DC(2:end,1);

ISI = dataArray{:, 8};%Initial Spread Index= score that coorrelate with fire velocity
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

day_bin = zeros(numel(day1_var),max(day1_var));
R=1:numel(day1_var);
day_bin(sub2ind(size(day_bin),R',day1_var))=1;

month_bin = zeros(numel(month1_var),max(month1_var));
R2=1:numel(month1_var);
month_bin(sub2ind(size(month_bin),R2',month1_var))=1

X_bin = zeros(numel(X),max(X));
R=1:numel(X);
X_bin(sub2ind(size(X_bin),R',X))=1;
Y_bin = zeros(numel(Y),max(Y));
R=1:numel(X);
Y_bin(sub2ind(size(Y_bin),R',Y))=1;

%% Building data

 nb_obs=length(X);
%FWI ? using
%only the four FWI components; and M ? with the four weather conditions
%M_data_nom=[X,Y,month1_var,day1_var,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area1];
X_att={'X1','X2','X3','X4','X5','X6','X7','X8','X9'};
Y_att={'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9'};
day_att={'mon','tue','wed','thu','fri','sat','sun'};
month_att={'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'};

M_data = [X_bin,Y_bin,month_bin,day_bin,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area1];
attributeNames_M = [X_att,Y_att,month_att,day_att,{'FFMC','DMC','DC','ISI'...
    ,'temp','RH','wind','rain','area'}];
n_att_M=length(attributeNames_M);


M_data_nom = [X,Y,month1_var,day1_var,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area1];
attributeNames_M_nom = {'X','Y','month','day','FFMC','DMC','DC','ISI'...
    ,'temp','RH','wind','rain','area'};
n_att_M_nom=length(attributeNames_M_nom);

M2_data = [X_bin,Y_bin,month_bin,day_bin,FFMC,DMC,DC,ISI,temp,RH,wind,rain];
attributeNames_M2 = [X_att,Y_att,month_att,day_att,{'FFMC','DMC','DC','ISI'...
    ,'temp','RH','wind','rain'}];
n_att_M2=length(attributeNames_M2);


%STFWI:using spatial, temporal and the four FWI
STFWI=[X_bin,Y_bin,month_bin,day_bin,FFMC,DMC,DC,ISI];
attributeNames_stfwi = [X_att,Y_att,month_att,day_att,{'FFMC','DMC','DC','ISI'}];
n_att_stfwi=length(attributeNames_stfwi);

%STM ? with the spatial, temporal and four weather variables; 
STM=[X_bin,Y_bin,month_bin,day_bin,temp,RH,wind,rain];
attributeNames_stm = [X_att,Y_att,month_att,day_att,{'temp','RH','wind','rain'}];
n_att_stm=length(attributeNames_stm);

%FWI ? using only the four FWI components;
FWI = [FFMC,DMC,DC,ISI];
attributeNames_fwi = {'FFMC','DMC','DC','ISI'};
n_att_fwi=length(attributeNames_fwi);

%MET- with the four weather conditions;
MET = [temp,RH,wind,rain];
attributeNames_met = {'temp','RH','wind','rain'};
n_att_met=length(attributeNames_met);



Col = size(month_bin,2) + size(day_bin,2)+size(X_bin,2)+size(Y_bin,2)-4;


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