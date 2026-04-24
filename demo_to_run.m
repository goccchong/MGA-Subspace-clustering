%% MADE-LFRC: Multi-granularity Anchor Embedded Discriminative Latent Low-rank
%% Fuzzy Representation Clustering for Color Image Segmentation
%%
%% CONFIRMED DIMENSIONS FROM PAPER (Eq.7, Eq.11, Eq.14-35):
%%   X   ∈ R^{d×n}    d=feat_dim=6,  n=N=pixels
%%   Us  ∈ R^{n×ls}   pixel-to-fine-anchor similarity
%%   Ud  ∈ R^{n×ld}   pixel-to-coarse-anchor similarity
%%   A   ∈ R^{d×ls}   orthonormal basis, A'A=I_{ls}  => use economy SVD trick
%%   M   ∈ R^{d×d}
%%   E   ∈ R^{d×n}
%%   S   ∈ R^{n×ls}   auxiliary for Us
%%   Z   ∈ R^{n×ld}   auxiliary for Ud   (Eq.11,29)
%%   H   ∈ R^{n×ls}   auxiliary for U
%%   Lh  ∈ R^{ld×ld}  Laplacian on coarse anchors
%%   Zp  ∈ R^{ld×ls}  fixed projection matrix (Eq.9)
%%
%% J2 = γ Tr(Z·Lh·Z') — uses trace identity to avoid N×N matrix
%%
%% Since d=6 < ls=245, A'A=I_{ls} is impossible for a real d×ls matrix.
%% SOLUTION: A is d×d square orthogonal (A'A=I_d), and Us is projected
%% to d dims (Us_p: n×d) for the LatLRR reconstruction only.
%% Full Us (n×ls) is kept for J2/fuzzy/assignment terms.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; warning off; close all;

%% ===================== Parameters =====================
phi_param   = 500;
l=9; s_nl=15; g_nl=10; sigma_nl=4;
cluster_num = 4;

alpha_lr = 0.6;   % ||M||_F^2
beta_lr  = 1.0;   % ||E||_{2,1}
gamma_lr = 0.7;   % Tr(Z Lh Z')
theta_lr = 0.4;   % fuzzy norm
rho_ht   = 2.0;   % hyperbolic tangent
delta_mg = 5;     % fine-grained multiplicity
k_grad   = 0.5;   % gradient adjustment

error_thresh = 1e-4;
max_iter     = 30;
eps1=1.0; eps2=1.0; eps3=1.0; eps4=1.0;
penalty_grow = 0.1;
penalty_max  = 1e6;

Color_Map = [0.960012728938293, 0.867750283531604, 0.539798806341941;
    0.00620528530061783, 0.456645636880323, 0.807705802687295;
    0.16111134052427435, 0.0634483119173452, 0.321794465524281;
    0.394969976441930,   0.662682208014249, 0.628325568854317;
    0.0945269059837998,  0.794795142853933, 0.473068960772494];

%% ===================== Input =====================
f_uint8 = imread('./s11.jpg');
f = double(f_uint8);
figure, imshow(f_uint8), title('Original image');
[row, col, depth] = size(f);
N = row * col;

truth_mask = [];
try
    load('C:\Users\23747\Desktop\image_result\classnote\class167062.mat');
    truth_mask = double(class167062);
catch ME
    warning('Ground truth not found: %s', ME.message);
end

%% ===================== Preprocessing =====================
fprintf('=== Preprocessing ===\n');
f_gpu = gpuArray(f);
non_local_info = non_local_information(f_gpu, l, s_nl, g_nl, sigma_nl);
local_var      = local_variance(non_local_info, phi_param);

all_pixels    = gather(reshape(double(f_gpu),  N, depth));   % N x depth
all_pixels_xi = gather(reshape(double(local_var), N, depth));

k_svd = min(10, min(size(all_pixels)));
[Up,Sp,Vp] = svd(all_pixels,'econ');    Sp(k_svd+1:end,k_svd+1:end)=0; all_pixels    = Up*Sp*Vp';
[Ux,Sx,Vx] = svd(all_pixels_xi,'econ'); Sx(k_svd+1:end,k_svd+1:end)=0; all_pixels_xi = Ux*Sx*Vx';

combined_feat = [all_pixels, all_pixels_xi];  % N x feat_dim
feat_dim = size(combined_feat, 2);            % d = 2*depth = 6
X = combined_feat';                           % d x N
fprintf('  N=%d, d=%d\n', N, feat_dim);

%% ===================== Multi-granularity Anchors =====================
fprintf('=== Generating multi-granularity anchors ===\n');
target_anchors = max(cluster_num*15, 20);

if exist('superpixels','file')==2
    [coarse_labels, ld] = superpixels(f_uint8, target_anchors);
    coarse_labels = double(coarse_labels);
else
    gray_img = double(rgb2gray(f_uint8));
    [gx,gy]  = gradient(gray_img);
    grad_mag = sqrt(gx.^2+gy.^2);
    marker   = imerode(grad_mag, strel('disk',3));
    rsmooth  = imgaussfilt(imreconstruct(marker,grad_mag), 2);
    ws       = watershed(rsmooth);
    coarse_labels = double(ws); coarse_labels(coarse_labels==0)=1;
    old_ids = unique(coarse_labels(:));
    mp = zeros(max(old_ids),1);
    for ii=1:length(old_ids), mp(old_ids(ii))=ii; end
    coarse_labels = arrayfun(@(x)mp(x), coarse_labels);
    ld = length(old_ids);
end
fprintf('  Coarse anchors ld=%d\n', ld);

fine_labels = zeros(row,col); cur_lbl=0; img_d=double(f_uint8);
for r_idx=1:ld
    mask_r = (coarse_labels==r_idx);
    rsz = sum(mask_r(:));
    if rsz < delta_mg*2
        cur_lbl=cur_lbl+1; fine_labels(mask_r)=cur_lbl; continue;
    end
    [ri,ci] = find(mask_r);
    feats = zeros(length(ri), depth+2);
    for dd=1:depth, ch=img_d(:,:,dd); feats(:,dd)=ch(mask_r); end
    feats(:,depth+1)=ri/row; feats(:,depth+2)=ci/col;
    nsub = min(delta_mg, max(2,round(rsz/50)));
    try
        slbls = kmeans(feats, nsub, 'Replicates',3, 'MaxIter',50);
    catch
        slbls=ones(size(feats,1),1); nsub=1;
    end
    for ss=1:nsub
        cur_lbl=cur_lbl+1;
        idx_f = sub2ind([row,col], ri(slbls==ss), ci(slbls==ss));
        fine_labels(idx_f)=cur_lbl;
    end
end
if any(fine_labels(:)==0), cur_lbl=cur_lbl+1; fine_labels(fine_labels==0)=cur_lbl; end
ls = cur_lbl;
fprintf('  Fine anchors ls=%d\n', ls);

%% ===================== Anchor Centers & Similarities =====================
fprintf('=== Computing anchor features & similarities ===\n');
clv = coarse_labels(:); flv = fine_labels(:);

anc_d = zeros(ld, feat_dim);
for a=1:ld
    idx_a=(clv==a); if any(idx_a), anc_d(a,:)=mean(combined_feat(idx_a,:),1); end
end
anc_s = zeros(ls, feat_dim);
for a=1:ls
    idx_a=(flv==a); if any(idx_a), anc_s(a,:)=mean(combined_feat(idx_a,:),1); end
end

sigma_sim = mean(std(combined_feat,0,1))+eps;

% Ud: N x ld
Ud = zeros(N,ld);
for a=1:ld
    dff=combined_feat-anc_d(a,:);
    Ud(:,a)=exp(-sum(dff.^2,2)/(2*sigma_sim^2));
end
Ud = Ud./(sum(Ud,2)+eps);

% Us: N x ls
Us = zeros(N,ls);
for a=1:ls
    dff=combined_feat-anc_s(a,:);
    Us(:,a)=exp(-sum(dff.^2,2)/(2*sigma_sim^2));
end
Us = Us./(sum(Us,2)+eps);

fprintf('  Us:%dx%d  Ud:%dx%d\n', size(Us,1),size(Us,2),size(Ud,1),size(Ud,2));

%% ===================== Graph Laplacian (ld x ld) =====================
fprintf('=== Graph Laplacian on coarse anchors ===\n');
knn_k = min(5,ld-1);
Ddist = pdist2(anc_d,anc_d);
sig_w = mean(Ddist(:))+eps;
W_anc = zeros(ld);
for i=1:ld
    [~,si]=sort(Ddist(i,:)); nb=si(2:knn_k+1);
    W_anc(i,nb)=exp(-Ddist(i,nb).^2/(2*sig_w^2));
end
W_anc = (W_anc+W_anc')/2;
Lh = diag(sum(W_anc,2)) - W_anc;  % ld x ld
Lh = (Lh+Lh')/2;

%% ===================== Projection Zp: ld x ls =====================
% Eq.(9): U_hat = Us + Ud * Zp'  where Zp in R^{ld x ls}
% Ud*(Zp)' = (N x ld)*(ls x ld)' ... Zp': ls x ld  => (N x ld)*(ld x ls)... 
% From paper Eq.(9): "Z ∈ R^{ld x ls} is the projection matrix"
fprintf('=== Computing projection Zp (ls x ld for MATLAB) ===\n');
% Zp: ls x ld  such that  Ud * Zp' = N x ls
% Least-squares: minimize ||anc_s - Zp * anc_d||_F
% => Zp = anc_s * pinv(anc_d)  (ls x feat_dim)*(feat_dim x ld) = ls x ld 
Zp = anc_s * pinv(anc_d);  % ls x ld

test_hat = Ud * Zp';  % (N x ld)*(ld x ls) = N x ls 
fprintf('  Zp: %dx%d,  Ud*Zp'': %dx%d (should be %dx%d)\n',...
    size(Zp,1),size(Zp,2), size(test_hat,1),size(test_hat,2), N,ls);

%% ===================== Project Us -> Us_p for LatLRR =====================

fprintf('=== Projecting Us (%dx%d) -> Us_p (%dx%d) ===\n', N,ls,N,feat_dim);
[~, ~, Vus] = svd(Us, 'econ');         % Vus: ls x ls
Us_p = Us * Vus(:, 1:feat_dim);        % N x d
Vus_keep = Vus(:, 1:feat_dim);         % ls x d (for back-projection if needed)
fprintf('  Us_p: %dx%d\n', size(Us_p,1), size(Us_p,2));

%% ===================== ADMM Initialization =====================
fprintf('=== ADMM Initialization ===\n');
d = feat_dim;  

% A: d x d, orthonormal. Init via SVD of X * Us_p = (d x N)*(N x d) = d x d
[A, ~, Va] = svd(X * Us_p, 'econ');
A = A * Va';   % d x d, A'A = I_d


S_var = Us_p;    % N x d   (auxiliary for Us_p, tanh rank)
Z_var = Ud;      % N x ld  (auxiliary for Ud, graph Laplacian)

M_var = zeros(d, d);   % d x d
E_var = zeros(d, N);   % d x N

% Fuzzy membership 
U_hat = Us + Ud * Zp';            % N x ls  
U_hat = max(min(U_hat,1.999),0.001);
sig_h = 1./(1+exp(-U_hat));
H_var = k_grad * U_hat .* sig_h;
H_var = max(min(H_var,0.999),0.001);
H_var = H_var ./ (sum(H_var,2)+eps);

% Lagrange multipliers
Y1 = zeros(N, d);    % N x d
Y2 = zeros(N, ld);   % N x ld
Y3 = zeros(N, ls);   % N x ls
Y4 = zeros(d, N);    % d x N

J_obj = zeros(max_iter,1);

% Verify
recon_test = A * Us_p' + M_var * X + E_var;
fprintf('  A:%dx%d  Us_p:%dx%d  M:%dx%d  E:%dx%d\n',...
    size(A,1),size(A,2), size(Us_p,1),size(Us_p,2),...
    size(M_var,1),size(M_var,2), size(E_var,1),size(E_var,2));
fprintf('  A*Us_p'' + M*X + E = %dx%d (should be %dx%d) ',...
    size(recon_test,1),size(recon_test,2), d, N);
fprintf('  Z_var:%dx%d  Lh:%dx%d\n', size(Z_var,1),size(Z_var,2),size(Lh,1),size(Lh,2));
fprintf('  Tr(Z*Lh*Z'') = sum(sum((Z*Lh).*Z)) avoids N×N matrix ');

%% ===================== ADMM Main Loop =====================
fprintf('=== ADMM Optimization ===\n');
for iter = 1:max_iter

    %% Step 1: Us_p subproblem
    % X = A*Us_p' + M*X + E  =>  A'*(residual)' gives N x d
    Q4 = X - M_var*X - E_var + Y4/eps4;          % d x N
    Us_p = (eps1*S_var - Y1 + eps4*(A'*Q4)') / (eps1+eps4);  % N x d 

    %% Step 2: E subproblem
    Q_E = X - A*Us_p' - M_var*X + Y4/eps4;       % d x N 
    thr_E = beta_lr / eps4;
    col_n = sqrt(sum(Q_E.^2,1));                  % 1 x N
    scale = max(col_n - thr_E, 0) ./ (col_n+eps); % 1 x N
    E_var = Q_E .* scale;                          % d x N 

    %% Step 3: M subproblem
    T_M = X - A*Us_p' - E_var + Y4/eps4;          % d x N
    % M = eps4*T_M*X' * inv(2*alpha*I + eps4*X*X')
    M_var = (eps4*(T_M*X')) / (2*alpha_lr*eye(d) + eps4*(X*X'));  % d x d 

    %% Step 4: A subproblem
    Q_A = X - M_var*X - E_var + Y4/eps4;          % d x N
    % SVD of Q_A * Us_p = (d x N)*(N x d) = d x d
    [Om, ~, Gm] = svd(Q_A * Us_p, 'econ');
    A = Om * Gm';   % d x d, A'A = I_d 

    %% Step 5: Ud subproblem
    Ud = Z_var + Y2/eps2;    % N x ld 
    Ud = max(Ud, 0);

    %% Step 6: Z subproblem
    % Z = (eps2*Ud - Y2) * inv(2*gamma*Lh + eps2*I)
    % Z: N x ld, Lh: ld x ld => (N x ld)*(ld x ld)^{-1} = N x ld 
    Z_var = (eps2*Ud - Y2) / (2*gamma_lr*Lh + eps2*eye(ld));  % N x ld 
    Z_var = max(Z_var, 0);

    %% Step 7: S subproblem
    W_s = Us_p + Y1/eps1;                         % N x d
    [Bs, Cs_d, Ds] = svd(W_s, 'econ');            % Bs: N x d, Ds: d x d
    sv = diag(Cs_d);  f_len=length(sv);
    eta = zeros(f_len,1);
    for i=1:f_len
        svi=sv(i);
        arg=min(svi^2/(2*rho_ht^2), 500);
        ep=exp(arg); en=exp(-arg);
        eta(i)=(4*svi/rho_ht^2)/((ep+en)^2+eps);
    end
    sv_new = max(sv - eta/eps1, 0);
    S_var = Bs * diag(sv_new) * Ds';              % N x d 

    %% Step 8: H subproblem -- fuzzy normalization
    U_hat = Us + Ud*Zp';                          % N x ls   (uses full Us)
    U_hat = max(min(U_hat,1.999),0.001);
    sig_u = 1./(1+exp(-U_hat));
    U_fuz = k_grad * U_hat .* sig_u;
    U_fuz = max(min(U_fuz,0.999),0.001);
    U_fuz = U_fuz ./ (sum(U_fuz,2)+eps);          % N x ls

    en_H = exp(-H_var);
    grad_H = k_grad*(1+en_H+H_var.*en_H)./((1+en_H).^2+eps);  % N x ls
    H_var = U_fuz - (theta_lr/eps3)*grad_H;
    H_var = max(min(H_var,0.999),0.001);

    %% Step 9: Lagrange multiplier updates
    Y1 = Y1 + eps1*(Us_p  - S_var);                              % N x d  
    Y2 = Y2 + eps2*(Ud    - Z_var);                              % N x ld 
    Y3 = Y3 + eps3*(U_fuz - H_var);                              % N x ls 
    Y4 = Y4 + eps4*(X - A*Us_p' - M_var*X - E_var);             % d x N  

    %% Update penalties
    eps1=min(eps1*penalty_grow, penalty_max);
    eps2=min(eps2*penalty_grow, penalty_max);
    eps3=min(eps3*penalty_grow, penalty_max);
    eps4=min(eps4*penalty_grow, penalty_max);

    %% Objective (Eq.7)
    sv_S   = svd(S_var,'econ');
    J_tanh = sum(tanh(sv_S.^2/(2*rho_ht^2))) / length(sv_S);
    J1 = J_tanh + alpha_lr*norm(M_var,'fro')^2 + ...
         beta_lr*sum(sqrt(sum(E_var.^2,1)));

    %% J2 = gamma * Tr(Z * Lh * Z')
   
    ZLh = Z_var * Lh;                             % N x ld (cheap)
    J2 = gamma_lr * sum(ZLh(:) .* Z_var(:));      % scalar 

    J3 = theta_lr * norm(H_var,'fro')^2;
    J_obj(iter) = J1 + J2 + J3;

    recon_err = norm(X - A*Us_p' - M_var*X - E_var,'fro') / (norm(X,'fro')+eps);
    fprintf('Iter %3d | J=%.5f | J1=%.4f J2=%.4f J3=%.4f | Recon=%.6f\n',...
        iter, J_obj(iter), J1, J2, J3, recon_err);

    if iter>1 && abs(J_obj(iter)-J_obj(iter-1))/(abs(J_obj(iter-1))+eps) <= error_thresh
        fprintf('*** Converged at iter %d ***\n', iter); break;
    end
end
if iter==max_iter, fprintf('Max iterations reached.\n'); end

%% ===================== Final Segmentation =====================
fprintf('=== Final Segmentation ===\n');

aff_s = anc_s * anc_s';            % ls x ls
aff_s = (aff_s+aff_s')/2; aff_s=max(aff_s,0);
D_s = diag(sum(aff_s,1));
L_s = D_s - aff_s;

try
    [ev,~] = eigs(L_s+1e-6*eye(ls), D_s+1e-6*eye(ls), cluster_num,'smallestabs');
catch
    [ev,~] = eigs(sparse(L_s+1e-6*eye(ls)), sparse(D_s+1e-6*eye(ls)), cluster_num,'sm');
end
ev = real(ev);
ev = ev ./ (sqrt(sum(ev.^2,2))+eps);
anc_labels = kmeans(ev, cluster_num, 'Replicates',20, 'MaxIter',300);

[~, pix_anc] = max(Us,[],2);       % N x 1: strongest fine anchor per pixel
cluster_idx  = anc_labels(pix_anc);

%% ===================== Visualization =====================
cluster_mask = reshape(cluster_idx, [row,col]);
result_rgb   = label2rgb(cluster_mask, Color_Map);

U_cluster = zeros(N, cluster_num);
for c=1:cluster_num
    aidx=find(anc_labels==c);
    if ~isempty(aidx), U_cluster(:,c)=sum(H_var(:,aidx),2); end
end
U_cluster = U_cluster./(sum(U_cluster,2)+eps);
Vpc = sum(U_cluster.^2,'all')/N*100;
Vpe = -sum(U_cluster.*log(U_cluster+eps),'all')/N*100;
fprintf('Vpc=%.2f%%  Vpe=%.2f%%\n', Vpc, Vpe);

figure; imshow(result_rgb); title('MADE-LFRC Result');
save('segmentation_result.mat','cluster_mask');
imwrite(result_rgb,'cluster_mask.png');

%% ===================== Evaluation =====================
if ~isempty(truth_mask) && isequal(size(double(cluster_mask)),size(truth_mask))
    pred=double(cluster_mask(:)); truth=truth_mask(:);
    pl=perms(1:cluster_num); bSA=0; bperm=1:cluster_num;
    for p=1:size(pl,1)
        mp2=pred;
        for cc=1:cluster_num, mp2(pred==cc)=pl(p,cc); end
        sa_t=sum(mp2==truth)/length(truth)*100;
        if sa_t>bSA, bSA=sa_t; bperm=pl(p,:); end
    end
    ml=pred;
    for cc=1:cluster_num, ml(pred==cc)=bperm(cc); end
    CM=confusionmat(truth,ml);
    tp=diag(CM); fp=sum(CM,1)'-tp; fn=sum(CM,2)-tp;
    SA=sum(tp)/sum(CM(:))*100;
    pr=tp./(tp+fp+eps); re=tp./(tp+fn+eps);
    F1=mean(2*(pr.*re)./(pr+re+eps))*100;
    mi=calc_mi(truth,ml);
    NMI=mi/sqrt(calc_ent(truth)*calc_ent(ml)+eps)*100;
    fprintf('\n=== Metrics ===\nSA=%.2f%%  F1=%.2f%%  NMI=%.2f%%\nVpc=%.2f%%  Vpe=%.2f%%\n',...
        SA,F1,NMI,Vpc,Vpe);
end



%% ===================== Local Helpers =====================
function e = calc_ent(x)
    p=histcounts(x,'Normalization','probability');
    e=-sum(p(p>0).*log2(p(p>0)));
end
function mi = calc_mi(x,y)
    jp=histcounts2(x,y,'Normalization','probability');
    mx=sum(jp,2); my=sum(jp,1);
    mi=sum(jp.*log2((jp+eps)./(mx*my+eps)),'all');
end
