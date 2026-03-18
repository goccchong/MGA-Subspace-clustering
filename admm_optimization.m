function U = admm_optimization(U_init, distants, m, lambda, L, max_admm_iter, rho)
% ADMM optimization for updating membership matrix U

[N, K] = size(U_init);

U = U_init;
Z = U;
Y = zeros(N, K);

for iter = 1:max_admm_iter
    
    % === Step 1: Update U ===
    U = (distants + rho*(Z - Y)) ./ (1 + rho);
    

    U = max(U, eps);
    
    % === Step 2: Update Z (projection: sum=1 constraint) ===
    Z_old = Z;
    
    Z = U + Y;
    Z = Z ./ sum(Z, 2);  
    
    % === Step 3: Update dual variable ===
    Y = Y + (U - Z);
    
    if norm(Z - Z_old, 'fro') < 1e-5
        break;
    end
end

U = Z;
end
