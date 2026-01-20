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
tao=2;
sigm=1e-5;
%density = 0.05;
% Parameters of non-local spatial information
l = 9;   %7
s = 15;
g=10;
sigma = 4;
cluster_num=3;
%% color map
Color_Map=[0.960012728938293,0.867750283531604,0.539798806341941;
    0.00620528530061783,0.456645636880323,0.807705802687295;
    0.16111134052427435,0.0634483119173452,0.321794465524281;
    0.394969976441930,0.662682208014249,0.628325568854317;
    0.0945269059837998,0.794795142853933,0.473068960772494];
%% Input Image 
%f_uint8=imread('C:\data\Project\AID_dataset\AID\River\river_255.jpg');
f_uint8=imread('C:\Users\23747\Desktop\work2\image_result\rs_image\rs174\rs174.jpg');
%f_uint8=imread('C:\data\Project\matlab\FSC_LNML-Algorithm-main\FSC_LNML-Algorithm-main\image\ILSVRC2012_test_00069249.jpg');
f=double(f_uint8);
figure,imshow(f_uint8),title('Original image');

%% ------------------------- 加载参考真值图像 -------------------------
try
    load('C:\Users\167062\class167062.mat'); 
    truth_mask = double(class167062);
catch ME
    error('参考真值文件加载失败: %s', ME.message);
end

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

% 动态计算允许的最大奇异值数量
k_p_max = min(size(all_pixels, 2), size(all_pixels, 1));  % all_pixels 是 N×depth 矩阵
k_xi_max = min(size(all_pixels_xi, 2), size(all_pixels_xi, 1)); 

% 设置保留的奇异值数量（不超过最大允许值）
k_p = min(10, k_p_max);    % 例如保留前10个奇异值（但不超过维度限制）
k_xi = min(10, k_xi_max);

% 分解原始像素矩阵 all_pixels
[U_p, S_p, V_p] = svd(all_pixels, 'econ');
S_p_trunc = diag([diag(S_p(1:k_p, 1:k_p)); zeros(k_p_max - k_p, 1)]);  % 安全截断
all_pixels = U_p * S_p_trunc * V_p';

% 分解局部方差特征矩阵 all_pixels_xi
[U_xi, S_xi, V_xi] = svd(all_pixels_xi, 'econ');
S_xi_trunc = diag([diag(S_xi(1:k_xi, 1:k_xi)); zeros(k_xi_max - k_xi, 1)]); 
all_pixels_xi = U_xi * S_xi_trunc * V_xi';
%% 低秩分解结束 

%% Calculate difference
difference = 20*(mean(mean(all_pixels)-mean(all_pixels_xi))).^2 + eps;
alpha = 1 ./ difference;
beta = difference;

%%拉普拉斯矩阵构建
% 参数设置
radius = 5;          % 邻域半径
sigma = 4;           % 高斯核标准差
lambda = 0.1;        % 正则化系数

[X, Y] = meshgrid(1:col, 1:row);
coordinates = [Y(:), X(:)];  

max_nonzero = N * (2*radius + 1)^2;  % 估计非零元素数量
rows = zeros(max_nonzero, 1);
cols = zeros(max_nonzero, 1);
values = zeros(max_nonzero, 1);
idx = 1;

for i = 1:N
    % 当前像素坐标 (y1, x1)
    y1 = coordinates(i, 1);
    x1 = coordinates(i, 2);
    
    y_min = max(1, y1 - radius);
    y_max = min(row, y1 + radius);
    x_min = max(1, x1 - radius);
    x_max = min(col, x1 + radius);
    
    [yg, xg] = meshgrid(y_min:y_max, x_min:x_max);
    neighbors = [yg(:), xg(:)];
    
    % 计算与邻居的欧式距离
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

% 计算拉普拉斯矩阵
D = spdiags(sum(W, 2), 0, N, N);
L = D - W;

% 上传到 GPU（可选）
if exist('gpuArray', 'file')
    L = gpuArray(L);
end
%% ==========结束 ==========

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

    %% 拉普拉斯约束项 
    % 隶属度链接项
    membership_linking = repmat(log(sum(repmat(mean(U),[N 1])) + 1) .^ 2, [N 1]); 
    
    % 拉普拉斯正则项梯度 (lambda * trace(U'*L*U))
    laplacian_term = lambda * (L * U); 
    
        for k=1:cluster_num
     
        dist_term = sum(repmat(w(k,:).^tao,N,1).*(alpha*(all_pixels-repmat(center(k,:),N,1)).^2 + ...
                      beta*(all_pixels_xi-repmat(center(k,:),N,1)).^2), 2);
        
        % 拉普拉斯约束项
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
        fprintf(fid,'%.4f\r',J(iter));   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj))
    
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

fprintf('成功生成 %d 个聚类隶属度图文件\n', cluster_num);






%% ------------------------- 聚类结果可视化与保存 -------------------------
[~, cluster_indices] = max(U, [], 2);
cluster_mask = reshape(cluster_indices, [row, col]);
result_rgb = label2rgb(cluster_mask, Color_Map);


Vpc=sum(gather(U).^2,'all')/(row*col)*100;
Vpe=-sum(gather(U).*log(gather(U)+eps),'all')/(row*col)*100;

fprintf('Fuzzy partition coefficient Vpc = %.2f%%\n', Vpc);
fprintf('Fuzzy partition entropy Vpe = %.2f%%\n', Vpe);
% 显示并保存结果
figure;
imshow(result_rgb);
title('Clustering Result');

% 保存分割结果为MAT文件
save('segmentation_result.mat', 'cluster_mask');  % 新增.mat保存
imwrite(result_rgb, 'cluster_mask.png');
imwrite(uint16(cluster_mask), 'class_mask.png');

loaded=load('segmentation_result.mat');
cluster_mask=double(cluster_mask);
cluster_mask=cluster_mask-1;
%cluster_mask(cluster_mask==0)==3;
%cluster_mask(cluster_mask==1)==0;
%cluster_mask(cluster_mask==3)==1;
%save('segmentation_result.mat', 'cluster_mask');


%% ------------------------- 指标 -------------------------
% 数据验证
if ~isequal(size(cluster_mask), size(truth_mask))
    error('聚类结果与真值图像尺寸不匹配');
end

% 转换为向量便于计算
pred_labels = cluster_mask(:);
true_labels = truth_mask(:);

% 计算混淆矩阵
conf_mat = confusionmat(true_labels, pred_labels);

% 基础参数计算
tp = diag(conf_mat);                     % 各类别真阳性
fp = sum(conf_mat, 1)' - tp;             % 各类别假阳性
fn = sum(conf_mat, 2) - tp;              % 各类别假阴性
total = sum(conf_mat(:));

SA = sum(tp) / total * 100;

% F1分数
precision = tp ./ (tp + fp + eps);
recall = tp ./ (tp + fn + eps);
f1_scores = 2 * (precision .* recall) ./ (precision + recall + eps);
macro_f1 = mean(f1_scores) * 100;

% 归一化互信息 (NMI)
mi = mutualinfo(true_labels, pred_labels);
h_true = entropy(true_labels);
h_pred = entropy(pred_labels);
NMI = mi / sqrt(h_true * h_pred) * 100;

% 平均交并比 (mIoU)
intersection = diag(conf_mat);
union = sum(conf_mat, 1)' + sum(conf_mat, 2) - intersection;
iou = intersection ./ (union + eps);
mIoU = mean(iou) * 100;

%% ------------------------- 结果输出 -------------------------
fprintf('[综合性能指标]\n');
fprintf('分割准确率 (SA)   : %.2f%%\n', SA);
fprintf('宏平均F1分数      : %.2f%%\n', macro_f1);
fprintf('归一化互信息 (NMI): %.2f%%\n', NMI);
fprintf('平均交并比 (mIoU) : %.2f%%\n', mIoU);

%% ------------------------- 辅助函数定义 -------------------------
function e = entropy(x)
    % 计算熵值
    p = histcounts(x, 'Normalization', 'probability');
    e = -sum(p(p>0) .* log2(p(p>0)));
end

function mi = mutualinfo(x, y)
    % 计算互信息
    joint_p = histcounts2(x, y, 'Normalization', 'probability');
    marginal_x = sum(joint_p, 2);
    marginal_y = sum(joint_p, 1);
    
    mi = sum(joint_p .* log2((joint_p + eps) ./ (marginal_x * marginal_y + eps)), 'all');
end
