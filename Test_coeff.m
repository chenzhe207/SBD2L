function [A, B] = DL(Y, D, alpha, beta)


tol = 1e-6; 
maxIter = 1e3;
rho = 1.1;
max_mu = 1e8;
mu = 1e-5;

[d, m] = size(Y);
[d, K] = size(D);
%% initialize
A = zeros(K, m);
J = zeros(K, m);
B = sparse(d, m);
T1 = zeros(K, m);


%% Release_BD
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    Ak = A; 
    Jk = J;
    Bk = B;

    
    %update A
    A = inv(D' * D + mu * eye(K)) * (D' * (Y - Bk) + mu * Jk - T1);

    %update J
    Jtemp = A + mu \ T1;
    [U,sigma,V] = svd(Jtemp, 'econ');
    sigma = diag(sigma);
    svp = length(find(sigma > alpha / mu));
    if svp>=1
      sigma = sigma(1:svp) - alpha / mu;
    else
      svp = 1;
      sigma = 0;
    end
    J = U(:, 1 : svp) * diag(sigma) * V(:, 1 : svp)';
    
     
          
    %update B sample-wisely
    Btemp = Y - D * A;
    B = max(0, Btemp - beta) + min(0, Btemp + beta);
 


     %% convergence check   
    leq1 = A - J;
    stopC = max(max(abs(leq1)));

    if stopC < tol || iter >= maxIter
        break;
    else
        T1 = T1 + mu * leq1;
        mu = min(max_mu, mu * rho);
    end
    if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e') ]);
    end
end
   
       
end

% Solving L_{21} norm minimization
function [E] = solve_l1l2(W,lambda)
n = size(W,1);
E = W;
for i=1:n
    E(i,:) = solve_l2(W(i,:),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end