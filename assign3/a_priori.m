
function [asso_m,freq_m]=a_priori(X,attributeNames,sup,conf)
M=size(X,2);
[Xbinary,attributeNamesBin]=binarize(X,[2*ones(1,M)],attributeNames);
writeAprioriFile(Xbinary,'m_binary.txt');

[asso_m,freq_m]=apriori('m_binary.txt',sup,conf);