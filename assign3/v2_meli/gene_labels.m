function [ y ] = gene_labels( output,nb_cl )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

area=log(output+1);
range=zeros(nb_cl,2);
step=floor(50/(nb_cl-1));
y=area;
range(1,:)=[min(area),median(area)];
y((min(range(1,:))<=area)& (area<=max(range(1,:))))=0;
for i=2:nb_cl
    a1=prctile(area,50+(i-2)*step);
    a2=prctile(area,50+(i-1)*step);
    
        if i==nb_cl
        a2=max(area);
        end
    
    
    range(i,:)=[a1,a2];
    y((min(range(i,:))<=area)& (area<=max(range(i,:))))=i-1;
end

end

