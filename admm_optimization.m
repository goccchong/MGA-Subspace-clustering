function [Us, Ud, Z, S, H, M, E, A, Y1, Y2, Y3, Y4] = admm_optimization(X, Us, Ud, U, A, Lh, params)

alpha = params.alpha;
beta = params.beta;
gamma = params.gamma;
theta = params.theta;

eps1 = params.eps1;
eps2 = params.eps2;
eps3 = params.eps3;
eps4 = params.eps4;

max_iter = params.max_iter;

[n, ~] = size(Us);

S = Us;
Z = Ud;
H = U;

M = zeros(size(X,1));
E = zeros(size(X));

Y1 = zeros(size(Us));
Y2 = zeros(size(Ud));
Y3 = zeros(size(U));
Y4 = zeros(size(X));

I_s = eye(size(Us,2));
I_d = eye(size(Ud,2));
I_x = eye(size(X,1));

for iter = 1:max_iter

    R1 = Y1 / eps1;
    R2 = Y2 / eps2;
    R3 = Y3 / eps3;
    R4 = Y4 / eps4;

    Us = (eps1*S - R1 + eps4*A'*X + eps4*A'*M*X + eps4*A'*E - A'*R4) * ((eps1 + eps4)*I_s)^(-1);

    Q = X - A*(Us)' - M*X - E + R4;
    for i = 1:size(Q,2)
        qi = Q(:,i);
        norm_q = norm(qi,2);
        if norm_q > beta/eps4
            E(:,i) = (norm_q - beta/eps4)/norm_q * qi;
        else
            E(:,i) = 0;
        end
    end

    M = (eps4*X - eps4*A*(Us)' - eps4*E + R4) * X' * ((alpha*I_x + eps4*(X*X'))^(-1));

    Q_A = X - M*X - E + R4;
    [Omega, ~, Gamma] = svd(Q_A*Us, 'econ');
    A = Omega * Gamma';

    Ud = Z + (1/eps2)*R2;

    Z = (eps2*Ud - R2) * ((2*gamma*Lh + eps2*I_d)^(-1));

    [Bs, Cs, Ds] = svd(S, 'econ');
    sigma = diag(Cs);
    eta = (4*sigma/(params.rho^2)) .* ((exp(sigma.^2/(2*params.rho^2)) + exp(-sigma.^2/(2*params.rho^2))).^(-2));
    grad_S = Bs * diag(eta) * Ds';
    S = Us + R1 - grad_S/eps1;

    H = U + R3 - (theta/eps3) * (params.phi * (1 + exp(-H) + H.*exp(-H)) ./ (1 + exp(-H)).^2);

    Y1 = Y1 + eps1*(Us - S);
    Y2 = Y2 + eps2*(Ud - Z);
    Y3 = Y3 + eps3*(U - H);
    Y4 = Y4 + eps4*(X - A*(Us)' - M*X - E);

end

end
