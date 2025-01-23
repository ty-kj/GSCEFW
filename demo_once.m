% The code is written by Kun Jiang, 
% if you have any problems, please don't hesitate to contact me: jk_365@xaut.edu.cn 

clear all
clc
clear memory;
addpath(genpath('data4sc'));

% name = 'YaleB';
% name = 'ORL';
% name = 'COIL20';
name = 'lung';

load (name);

fea = fea';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
n = length(gnd);
nnClass = length(unique(gnd));  

options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
Z = constructW(fea',options);
Z = full(Z);
Z1 = Z-diag(diag(Z));         
Z = (Z1+Z1')/2;
DZ= diag(sum(Z));
LZ = DZ - Z;                
[F_ini, ~, evs]=eig1(LZ, nnClass, 0);
Z_ini = Z;
clear LZ DZ Z Z1 options

lambda1 = 10.000000; lambda2 = 0.000010; lambda3 = 0.000010;


miu = 1e-3;
rho = 1.1;
max_iter = 20;
% % if you only have cpu do this 
Ctg = inv(fea'*fea+eye(size(fea,2)));
% for max_iter=1:20
tic;


[Z,S,obj] = GSCEFW(fea,F_ini,Z_ini,nnClass,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);

addpath('Ncut_9');
Z_out = Z;
A = Z_out;
A = A - diag(diag(A));
A = abs(A);
A = (A+A')/2;  

for k=1:10
    [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
    result_label = zeros(size(fea,2),1);%vec2ind
    for j = 1:nnClass
        id = find(NcutDiscrete(:,j));
        result_label(id) = j;
    end
    result = ClusteringMeasure(gnd, result_label);
    % acc  = result(1)
    % nmi  = result(2) 

    acc(k)  = result(1);
    nmi(k)  = result(2);  
end % 10 experiments

fprintf('all the acc values are :%f\n',acc);
fprintf('mean acc is: %f and std is: %f\n\n', mean(acc), std(acc));
fprintf('all the nmi values are :%f\n',nmi);
fprintf('mean nmi is: %f and std is: %f\n\n', mean(nmi), std(nmi));

% acc_iter(max_iter)=mean(acc);
% end

                                                      
