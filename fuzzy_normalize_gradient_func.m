function grad = fuzzy_normalize_gradient_func(H, k)
%FUZZY_NORMALIZE_GRADIENT_FUNC Gradient of fuzzy normalization
%   nabla_H = k * (1 + exp(-h) + h*exp(-h)) / (1 + exp(-h))^2
%
%   Inputs:
%     H - n x ls matrix
%     k - gradient adjustment factor
%   Outputs:
%     grad - n x ls gradient

    exp_neg = exp(-H);
    denom = (1 + exp_neg).^2;
    numer = 1 + exp_neg + H .* exp_neg;
    grad = k * numer ./ (denom + eps);
end