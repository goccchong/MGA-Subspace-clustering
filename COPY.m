% CUDA is required in this version. 
% However there is no need to install CUDA seperately since MATLAB has done all the work.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Intialization
clear
warning off
close all
%% Parameters
m = 2;
error = 0.0001;
max_iter = 100;
phi=500;
cluster_num=3;
%% color map
Color_Map=[0.960012728938293,0.867750283531604,0.539798806341941;
    0.00620528530061783,0.456645636880323,0.807705802687295;
    0.16111134052427435,0.0634483119173452,0.321794465524281;
    0.394969976441930,0.662682208014249,0.628325568854317;
    0.0945269059837998,0.794795142853933,0.473068960772494];
%% Input Image 
%f_uint8=imread('C:\data\Project\river_255.jpg');
f_uint8=imread('C:\Users\rs174.jpg');
%f_uint8=imread('C:\data\ILSVRC2012_test_00069249.jpg');
f=double(f_uint8);
figure,imshow(f_uint8),title('Original image');

try
    load('C:\Users\167062\class167062.mat'); 
    truth_mask = double(class167062);
catch ME
    error('参考真值文件加载失败: %s', ME.message);
end
rho = 1.0;
max_admm_iter = 20;

%% Adding mixing noise
%%f = f / 255;
%%f = imnoise(f,'gaussian',0,density);
%%f = imnoise(f,'salt & pepper',density);
%%f = imnoise(f,'speckle',density);
%%f=f*255;
%%figure,imshow(uint8(f)),title('Noise image');
%% Calculate size
[row,col,depth] = size (f);
N = row * col;
%% Computing non-local spatial information and local variance information
f = gpuArray(f);
non_local_infomation = non_local_information(f, l, s, g, sigma);
local_variance = local_variance(non_local_infomation,phi);
%% Pixel reorganization
all_pixels = gather(reshape(double(f), N, depth));
all_pixels_xi = gather(reshape(double(local_variance), N, depth));


k_p_max = min(size(all_pixels, 2), size(all_pixels, 1));
k_xi_max = min(size(all_pixels_xi, 2), size(all_pixels_xi, 1)); 


k_p = min(10, k_p_max);    
k_xi = min(10, k_xi_max);


[U_p, S_p, V_p] = svd(all_pixels, 'econ');
S_p_trunc = diag([diag(S_p(1:k_p, 1:k_p)); zeros(k_p_max - k_p, 1)]); 
all_pixels = U_p * S_p_trunc * V_p';


[U_xi, S_xi, V_xi] = svd(all_pixels_xi, 'econ');
S_xi_trunc = diag([diag(S_xi(1:k_xi, 1:k_xi)); zeros(k_xi_max - k_xi, 1)]); 
all_pixels_xi = U_xi * S_xi_trunc * V_xi';


%% Calculate difference
difference = 20*(mean(mean(all_pixels)-mean(all_pixels_xi))).^2 + eps;
alpha = 1 ./ difference;
beta = difference;



radius = 5;         
sigma = 4;          

[X, Y] = meshgrid(1:col, 1:row);
coordinates = [Y(:), X(:)];  

max_nonzero = N * (2*radius + 1)^2;  
rows = zeros(max_nonzero, 1);
cols = zeros(max_nonzero, 1);
values = zeros(max_nonzero, 1);
idx = 1;

for i = 1:N
    y1 = coordinates(i, 1);
    x1 = coordinates(i, 2);
    
    y_min = max(1, y1 - radius);
    y_max = min(row, y1 + radius);
    x_min = max(1, x1 - radius);
    x_max = min(col, x1 + radius);
    
    [yg, xg] = meshgrid(y_min:y_max, x_min:x_max);
    neighbors = [yg(:), xg(:)];
    
 
    dist = sqrt((y1 - neighbors(:,1)).^2 + (x1 - neighbors(:,2)).^2);
    
    w = exp(-dist.^2 / (2*sigma^2));

    neighbor_linear = sub2ind([row, col], neighbors(:,1), neighbors(:,2));
    
    num_neighbors = length(neighbor_linear);
    rows(idx:idx+num_neighbors-1) = i;
    cols(idx:idx+num_neighbors-1) = neighbor_linear;
    values(idx:idx+num_neighbors-1) = w;
    idx = idx + num_neighbors;
end

rows(idx:end) = [];
cols(idx:end) = [];
values(idx:end) = [];

W = sparse(rows, cols, values, N, N);
W = max(W, W'); 


D = spdiags(sum(W, 2), 0, N, N);
L = D - W;


if exist('gpuArray', 'file')
    L = gpuArray(L);


%% Allocate memory space
J=zeros(max_iter,1);
[N,depth]=size(all_pixels);
w=zeros(cluster_num, depth);
distants=zeros(N, cluster_num);
%% Initializing membership
U=rand(N,cluster_num);
U_row_sum=sum(U,2);
U=U./repmat(U_row_sum,[1 cluster_num]);
U_m=U.^m;
%%  Detailed membership degrees in a randomly collected 5x5 local area
U_cluster1_local_5x5=zeros(5,5);
U_cluster2_local_5x5=zeros(5,5);
%% Begin Clustering
for iter=1:max_iter
    % Update Clustering Center using Eq.(39)
    center=((U_m')*(all_pixels*alpha))+((U_m')*(all_pixels_xi*beta));  
    center=center./((sum(U_m))'*ones(1,depth)*(alpha+beta));
    % Update the weight matrix using Eq.(46)
    for k=1:cluster_num
        w(k, :)=sum(repmat(U_m(:,k), 1, depth).*(alpha*(all_pixels-repmat(center(k,:), N, 1)).^2)+repmat(U_m(:,k), 1, depth).*(beta*(all_pixels_xi-repmat(center(k,:), N, 1)).^2))+sigm.*ones(1, depth);  %1*D
    end
    w_up=w.^(-1/(tao-1));  
    w=w_up./repmat(sum(w_up,2), 1, depth); % weight

    membership_linking = repmat(log(sum(repmat(mean(U),[N 1])) + 1) .^ 2, [N 1]); 
    
  
    laplacian_term = lambda * (L * U); 
    
        for k=1:cluster_num
     
        dist_term = sum(repmat(w(k,:).^tao,N,1).*(alpha*(all_pixels-repmat(center(k,:),N,1)).^2 + ...
                      beta*(all_pixels_xi-repmat(center(k,:),N,1)).^2), 2);
        
     
        distants(:,k) = dist_term + laplacian_term(:,k); 
    end
    membership_linking = repmat(log(sum(repmat(mean(U),[N 1])) + 1) .^ 2, [N 1]);
    U_numerator=(distants./membership_linking).^(1/(m-1));
    U=U_numerator.*repmat(sum(1./U_numerator,2),[1,cluster_num]);
    U=1./U;
    U_m=U.^m;
    % Check local membership degrees
    U_reshape1 = reshape(U(:, 1), row, col);
    U_reshape2 = reshape(U(:, 2), row, col);
    U_cluster1_local_5x5(:, :, iter) = U_reshape1(116 :120 , 154 : 158); 
    U_cluster2_local_5x5(:, :, iter) = U_reshape2(116 :120 , 154 : 158);
    J(iter)=sum(sum((U_m.*distants)./membership_linking))+sigm*sum(sum(w.^tao));
    fprintf('Iter %d\n', iter);
    fprintf('%d object function   : %.4f\n', iter,J(iter));
    fid=fopen(['C:\data\Project\matlab\FSC_LNML-Algorithm-main\','A.txt'],'a');
        fprintf(fid,'%.4f\r',J(iter));  
    
    % Convergence condition
    if iter > 1 && abs(J(iter) - J(iter - 1)) <= error
        fprintf('Objective function is converged\n');
        break;
    end
    if iter > 1 && iter == max_iter && abs(J(iter) - J(iter - 1)) > error
        fprintf('Objective function is not converged. Max iteration reached\n');
        break;
    end
end

U = admm_optimization(U, distants, m, lambda, L, max_admm_iter, rho);

U_m = U.^m;







[~, cluster_indices] = max(U, [], 2);
cluster_mask = reshape(cluster_indices, [row, col]);
result_rgb = label2rgb(cluster_mask, Color_Map);


Vpc=sum(gather(U).^2,'all')/(row*col)*100;
Vpe=-sum(gather(U).*log(gather(U)+eps),'all')/(row*col)*100;

fprintf('Fuzzy partition coefficient Vpc = %.2f%%\n', Vpc);
fprintf('Fuzzy partition entropy Vpe = %.2f%%\n', Vpe);

figure;
imshow(result_rgb);
title('Clustering Result');


save('segmentation_result.mat', 'cluster_mask'); 
imwrite(result_rgb, 'cluster_mask.png');
imwrite(uint16(cluster_mask), 'class_mask.png');

loaded=load('segmentation_result.mat');
cluster_mask=double(cluster_mask);
cluster_mask=cluster_mask-1;
%cluster_mask(cluster_mask==0)==3;
%cluster_mask(cluster_mask==1)==0;
%cluster_mask(cluster_mask==3)==1;
%save('segmentation_result.mat', 'cluster_mask');


pred_labels = cluster_mask(:);
true_labels = truth_mask(:);

conf_mat = confusionmat(true_labels, pred_labels);

tp = diag(conf_mat);                    
fp = sum(conf_mat, 1)' - tp;           
fn = sum(conf_mat, 2) - tp;             
total = sum(conf_mat(:));

SA = sum(tp) / total * 100;


precision = tp ./ (tp + fp + eps);
recall = tp ./ (tp + fn + eps);
f1_scores = 2 * (precision .* recall) ./ (precision + recall + eps);
macro_f1 = mean(f1_scores) * 100;


mi = mutualinfo(true_labels, pred_labels);
h_true = entropy(true_labels);
h_pred = entropy(pred_labels);
NMI = mi / sqrt(h_true * h_pred) * 100;


intersection = diag(conf_mat);
union = sum(conf_mat, 1)' + sum(conf_mat, 2) - intersection;
iou = intersection ./ (union + eps);
mIoU = mean(iou) * 100;




function e = entropy(x)
    p = histcounts(x, 'Normalization', 'probability');
    e = -sum(p(p>0) .* log2(p(p>0)));
end

function mi = mutualinfo(x, y)
    joint_p = histcounts2(x, y, 'Normalization', 'probability');
    marginal_x = sum(joint_p, 2);
    marginal_y = sum(joint_p, 1);
    
    mi = sum(joint_p .* log2((joint_p + eps) ./ (marginal_x * marginal_y + eps)), 'all');
end

