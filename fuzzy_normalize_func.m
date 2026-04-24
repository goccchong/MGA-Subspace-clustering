function U = fuzzy_normalize_func(U_hat, k)
%FUZZY_NORMALIZE_FUNC Fuzzy normalization tau (Eq. 10)
%   tau(u_hat) = (k * u_hat) / (1 + exp(-u_hat))
%   Then row-normalize to get valid membership
%
%   Inputs:
%     U_hat - n x ls enhanced membership (values in (0,2))
%     k     - gradient adjustment factor
%   Outputs:
%     U - n x ls fuzzy membership (values in (0,1))

    sigmoid_val = 1 ./ (1 + exp(-U_hat));
    U = k * U_hat .* sigmoid_val;
    U = max(min(U, 0.999), 0.001);
    row_sums = sum(U, 2) + eps;
    U = U ./ row_sums;
end