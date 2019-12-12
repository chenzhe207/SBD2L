function [W, M] = classifier_learning(A, c, Htr, gama)


[K, n] = size(A);
M = ones(c, n);
W = zeros(c, K);
maxIter = 100;

B = Htr;
for i = 1 : n
    for j = 1 : c
        if B(j, i) == 0
            B(j, i) = -1;
        end
    end
end

%% DSLRR
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    Mk = M;
    Wk = W;
   
    %update W
    W = Htr .* Mk * A' * inv(A * A' + gama * eye(K)); 
    
    %update M
    Mtemp = 2 * W * A - Mk;
    M = max(B .* Mtemp, 0);
 
end

 W = normc(W);