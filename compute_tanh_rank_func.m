function val = compute_tanh_rank_func(S, rho)
%COMPUTE_TANH_RANK_FUNC Hyperbolic tangent rank value
%   ||S||_FHT = (1/n) * sum_i tanh(sigma_i^2 / (2*rho^2))
%
%   Inputs:
%     S   - n x ls matrix
%     rho - parameter (>0)


    sigma_vals = svd(S, 'econ');
    n_sv = length(sigma_vals);
    val = 0;
    for i = 1:n_sv
        val = val + tanh(sigma_vals(i)^2 / (2 * rho^2));
    end
    val = val / n_sv;
end