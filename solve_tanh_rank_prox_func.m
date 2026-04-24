function S = solve_tanh_rank_prox_func(W, rho, step_size)
%SOLVE_TANH_RANK_PROX_FUNC Proximal step for hyperbolic tangent rank (Eq. 30-33)
%   Gradient descent on singular values using tanh rank gradient

    [Bs, Cs_diag, Ds] = svd(W, 'econ');
    sigma_vals = diag(Cs_diag);
    f_len = length(sigma_vals);

    % Eq. (33): eta(i) = (4*sigma(i)/rho^2) / (exp(sigma^2/(2*rho^2)) + exp(-sigma^2/(2*rho^2)))^2
    eta = zeros(f_len, 1);
    for i = 1:f_len
        sv = sigma_vals(i);
        arg = sv^2 / (2 * rho^2);
        exp_pos = exp(arg);
        exp_neg = exp(-arg);
        denom = (exp_pos + exp_neg)^2;
        eta(i) = (4 * sv / rho^2) / (denom + eps);
    end

    sigma_new = max(sigma_vals - step_size * eta, 0);
    S = Bs * diag(sigma_new) * Ds';
end