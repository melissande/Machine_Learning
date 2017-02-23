
close all
sm = size(M_data);
col= sm(2);
lig = sm(1);
mean_Mdata = mean(M_data,1);
std_Mdata = std(M_data,1);
var_Mdata = std(M_data,1).^2;
cov_Mdata = cov(M_data);
median_Mdata = median(M_data,1);
range_Mdata = range(M_data,1);
cosine = zeros(col,col);


for i=1:col 
    for j = 1:col
cosine(i,j) = M_data(:,i)'*M_data(:,j)/(norm(M_data(:,i))*norm(M_data(:,j)));
    end
end
correl = zeros(col,col);
for i=1:col 
    for j = 1:col
        correl(i,j) = norm(cov_Mdata(i,j))/(std_Mdata(i)*std_Mdata(j))
    end
end

extjacq = zeros(col,col);
for i=1:col 
    for j = 1:col
        extjacq(i,j) = M_data(:,i)'*M_data(:,j)/(norm(M_data(:,i))^2+norm(M_data(:,j))^2-M_data(:,i)'*M_data(:,j))
    end
end

figure
imagesc(correl)
colorbar
title('correlation attributes')
figure
imagesc(cosine)
colorbar
title('cosine attr')
figure
imagesc(extjacq)
colorbar
title('extended jacquart')
