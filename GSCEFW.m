function [Z,S,obj] = GERAFW(X,F_ini,Z_ini,c,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho)
% The code is written by Kun Jiang, 
% if you have any problems, please don't hesitate to contact me: jk_365@xaut.edu.cn 

max_miu = 1e8;
tol  = 1e-6;
tol2 = 1e-2;
C1 = zeros(size(X));
C2 = zeros(size(Z_ini));

for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        U = Z_ini;
        E = X-X*Z;
        F = F_ini;
        clear Z_ini F_ini
    end
    Z_old = Z;
    E_old = E;
    U_old = U;
 
    % ------------ S ------------- %
    S_temp = -(E.^2)/lambda1;
    S = zeros(size(S_temp));
    for j = 1:size(S,2)
        for i = 1:size(S,1)
            S(i,j) = exp(S_temp(i,j));
        end
        S(:,j) = S(:,j)./sum(S(:,j));
    end
%     ind=(S<1e-3);
%     S(ind)=0;

   
    % --------- E -------- %
    N = X-X*Z+C1/miu;
    E = (miu*N)./(miu+2*S);
    
    % -------- Z ------------ %
    M1 = X-E+C1/miu;
    M2 = U-C2/miu;
    D = L2_distance_1(F',F');
    Z = Ctg*(X'*M1+M2-lambda3*D/miu);
    Z = Z - diag(diag(Z));
    for ii = 1:size(Z,2)
        idx = 1:size(Z,2);
        idx(ii) = [];
        Z(ii,idx) = EProjSimplex_new(Z(ii,idx));
    end
    % ------------ F ------------ %
    LU = (Z+Z')/2;
    L = diag(sum(LU)) - LU;
    [F, ~, ev] = eig1(L, c, 0);
    
    % ------------ U ------------ %
    A = Z.*(F*F');
    tempU = Z+C2/miu;
    U = (miu*tempU+2*lambda2*A)/(miu+2*lambda2); 
    
%     % ------ C1 C2 miu ---------- %
    L1 = X-X*Z-E;
    L2 = Z-U; 
    C1 = C1+miu*L1;
    C2 = C2+miu*L2;
    
    LL1 = norm(Z-Z_old,'fro');
    LL2 = norm(U-U_old,'fro');
    LL3 = norm(E-E_old,'fro');
    SLSL= max(max(LL1,LL2),LL3)/norm(X,'fro');
    if miu*SLSL < tol2
        miu = min(rho*miu,max_miu); 
    end
    stopC = (norm(LL1,'fro')+norm(LL2,'fro')+norm(LL3,'fro'))/norm(X,'fro');
    if stopC < tol
    %         iter
        break;
    end
    obj(iter) = stopC;    
end
end




    
