% The code is written by Kun Jiang

clear all
clc
clear memory;
addpath(genpath('data4sc'));
addpath(genpath('utils'));

name = 'COIL20';
% name = 'YaleB';
% name = 'ORL';
% name = 'COIL100';
% name = 'MNIST_6996';
% name = 'COIL20';
% name = 'usps_random_1000';
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

% % if you only have cpu do this 
Ctg = inv(fea'*fea+eye(size(fea,2)));

% % -------- if you have gpu you can accelerate the inverse operation as follows:  ---------- % %
% Xg = gpuArray(single(fea));
% Ctg = inv(Xg'*Xg+eye(n));
% Ctg = double(gather(Ctg));
% clear Xg;

lambda1 = 10;
lambda2 = 0.001;
lambda3 = 0.02;

%grid search
filename=['./results/',name,'_',datestr(now,30),'.txt'];
fileID = fopen(filename,'w');
lambdas=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
% lambdas=[1e-6, 1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5];
accs=[]; nmis=[];
for i =1 : 1
    lambda1 = lambdas(i);
     lambda1 = 10;
    for ii =1 : length(lambdas)
        lambda2 = lambdas(ii);
        for iii =1 : length(lambdas)
            lambda3 = lambdas(iii);
            
            fprintf(fileID,'lambda1 : %f, lambda2 : %f, lambda3 : %f\n', lambda1,lambda2,lambda3);
            experiments=10;
            acc=[]; nmi=[];
            
            miu = 1e-3;
            rho = 1.1;
            max_iter = 20;
            [Z,S,obj] = GERAFW(fea,F_ini,Z_ini,nnClass,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
            addpath('Ncut_9');
            Z_out = Z;
            A = Z_out;
            A = A - diag(diag(A));
            A = abs(A);
            A = (A+A')/2;  
            for k=1:experiments
                [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
                result_label = zeros(size(fea,2),1);%vec2ind
                for j = 1:nnClass
                    id = find(NcutDiscrete(:,j));
                    result_label(id) = j;
                end
                result = ClusteringMeasure(gnd, result_label);
                
                acc(k)  = result(1);
                nmi(k)  = result(2);  
            end % 10 experiments
            fprintf(fileID,'all the acc values are :%f\n',acc);
            fprintf(fileID,'mean acc is: %f and std is: %f\n\n', mean(acc), std(acc));
            fprintf(fileID,'all the nmi values are :%f\n',nmi);
            fprintf(fileID,'mean nmi is: %f and std is: %f\n\n', mean(nmi), std(nmi));
            accs(ii,iii)=mean(acc);
            nmis(ii,iii)=mean(nmi);
        end
       
    end
end
fclose(fileID);                                                       