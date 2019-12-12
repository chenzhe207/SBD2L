function [A, B, D, R, value] = DL(X, Dinit, train_num, c, alpha, beta, lambda1, lambda2)


tol = 1e-6; 
rho = 1.1;
max_mu = 1e8;
mu = 1e-5;
maxIter = 1e3;


[d, n] = size(X);
[d, K] = size(Dinit);
k = K / c;

%% initialize
A = zeros(K, n);
J = zeros(K, n);
B = sparse(d, n);
R = zeros(K, n);
D = Dinit;
T1 = zeros(K, n);


Q = []; %¿é¶Ô½Ç¾ØÕó
for i = 1 : c
    t = ones(k, train_num);
    Q = blkdiag(Q, t);
    clear t
end

%% Release_BD
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    Ak = A; 
    Jk = J;
    Bk = B;
    Rk = R;
    Dk = D;
    
    %update A
    A = inv(Dk' * Dk + (lambda1 + mu) * eye(K)) * (Dk' * (X - Bk) + lambda1 * (Q + R) + mu * Jk - T1);
    
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
    Btemp = X - D * A;
    B = max(0, Btemp - beta) + min(0, Btemp + beta);
 
    %update R 
    R = [];
    for i = 1 : c
      Ai = A(:, (i - 1) * train_num + 1 : i * train_num);
      Qi = Q(:, (i - 1) * train_num + 1 : i * train_num);
      Ritemp = Ai - Qi;
      Ri = solve_l1l2(Ritemp, lambda2 / lambda1);
      R = [R, Ri];
      clear Ai Qi Aitemp Ri
    end
   
    %update D
    D = (X - B) * A' * inv(A * A' + 0.0001 * eye(K));
    
%     value1 = 2 \ (norm((X - D * A - B), 'fro')^2);
%     value2 = alpha * sum(svd(A));
%     value3 = beta * sum(sum(abs(B)));
%     value4 = 2 \ (lambda1 * (norm((A - Q - R), 'fro')^2));
%     value5 = 0;
%     for i = 1 : c
%        Ri = R(:, (i - 1) * train_num + 1 : i * train_num); 
%        temp = sqrt(Ri .* Ri);
%        value5 = value5 + sum(sum(temp, 2));
%     end
%     value5 = lambda2 * value5;
%     value(iter) = value1 + value2 + value3 + value4 + value5;
    
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