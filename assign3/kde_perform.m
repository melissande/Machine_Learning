function [] = kde_perform( X,dataset_Name )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%% Gausian Kernel density estimator
% cross-validate kernel width by leave-one-out-cross-validation
% automatically implemented in the script gausKernelDensity
widths=max(var(X))*(2.^[-10:2]); % evaluate for a range of kernel widths

for w=1:length(widths)
   [density,log_density]=gausKernelDensity(X,widths(w));
   logP(w)=sum(log_density);
end

[val,ind]=max(logP);
width=widths(ind);
display([dataset_Name,'Optimal kernel width is ' num2str(width)])
% evaluate density for estimated width
density=gausKernelDensity(X,width);

% Sort the densities
[y,i] = sort(density);

% Plot outlier scores
mfig([dataset_Name,'Gaussian Kernel Density: outlier score']); clf;
bar(y(1:20));



end

