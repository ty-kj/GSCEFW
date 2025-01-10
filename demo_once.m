% The code is written by Kun Jiang, 
% if you have any problems, please don't hesitate to contact me: jk_365@xaut.edu.cn 

clear all
clc
clear memory;
addpath(genpath('data4sc'));
addpath(genpath('utils'));

name = 'YaleB';
name = 'ORL';
% name = 'COIL20';
% name = 'COIL100';
% name = 'MNIST_6996';
% name = 'usps_random_1000';
% name = '3ring_data';
load (name);
% fea=X;
% gnd=y;

fea = fea';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
n = length(gnd);
nnClass = length(unique(gnd));  

options = [];
options.NeighborMode = 'KNN';
options.k = 10;
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

lambda1 = 0.010; lambda2 = 0.000100; lambda3 = 0.000010;
% lambda1 = 0.000010; lambda2 = 0.100000; lambda3 = 1.000000;
miu = 1e-3;
rho = 1.1;
% miu = 1.25;
% rho = 1.15;
max_iter = 20;
% % if you only have cpu do this 
Ctg = inv(fea'*fea+eye(size(fea,2)));
% for max_iter=1:20
for k=1:20
    [Z,S,obj] = GERAFW(fea,F_ini,Z_ini,nnClass,lambda1,lambda2,lambda3,k,Ctg,miu,rho);
    addpath('Ncut_9');
    Z_out = Z;
    A = Z_out;
    A = A - diag(diag(A));
    A = abs(A);
    A = (A+A')/2;  

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

                                                      