function [labels, num_anchors] = generate_coarse_anchors(img, cluster_num)
% Generate coarse-grained anchors via superpixel decomposition
% GMMR (Gradient Momentum Morphology Reconstruction)
%
% Inputs:
%   img         - uint8 color image
%   cluster_num - number of desired clusters (guides anchor count)
%
% Outputs:
%   labels      - row x col label map
%   num_anchors - number of coarse anchors generated

[row, col, ~] = size(img);
N = row * col;

% Target: coarse anchors ~ 10-30x cluster_num
target_anchors = max(cluster_num * 15, 20);


if exist('superpixels', 'file')
    [labels, num_anchors] = superpixels(img, target_anchors);
else

    fprintf('Using grid-based coarse anchor generation...\n');
    
    gray = double(rgb2gray(img));
    

    [gx, gy] = gradient(gray);
    grad_mag = sqrt(gx.^2 + gy.^2);
    

    se = strel('disk', 3);
    marker = imerode(grad_mag, se);
    recon = imreconstruct(marker, grad_mag);

    recon_smooth = imgaussfilt(recon, 2);
    ws = watershed(recon_smooth);
    

    labels = ws;
    region_ids = unique(labels(:));
    region_ids(region_ids == 0) = [];
    

    new_label = 0;
    label_map = zeros(max(region_ids), 1);
    for i = 1:length(region_ids)
        new_label = new_label + 1;
        label_map(region_ids(i)) = new_label;
    end
    
    for i = 1:row
        for j = 1:col
            if labels(i,j) > 0
                labels(i,j) = label_map(labels(i,j));
            else
                labels(i,j) = 1;  
            end
        end
    end
    
    num_anchors = new_label;
    
    if num_anchors > target_anchors * 3

        min_size = round(N / (target_anchors * 2));
        labels = merge_small_regions(labels, min_size, img);
        num_anchors = max(labels(:));
    end
end

fprintf('Coarse anchors generated: %d\n', num_anchors);
end

function labels = merge_small_regions(labels, min_size, img)
% Merge regions smaller than min_size into nearest neighbor
[row, col, ~] = size(img);
gray = double(rgb2gray(img));

num_regions = max(labels(:));
region_sizes = zeros(num_regions, 1);
region_means = zeros(num_regions, 1);

for r = 1:num_regions
    mask = (labels == r);
    region_sizes(r) = sum(mask(:));
    region_means(r) = mean(gray(mask));
end

changed = true;
while changed
    changed = false;
    num_regions = max(labels(:));
    for r = 1:num_regions
        mask = (labels == r);
        if sum(mask(:)) < min_size && sum(mask(:)) > 0
            % Find neighboring regions
            dilated = imdilate(mask, strel('square', 3));
            border = dilated & ~mask;
            neighbor_labels = unique(labels(border));
            neighbor_labels(neighbor_labels == r) = [];
            neighbor_labels(neighbor_labels == 0) = [];
            
            if ~isempty(neighbor_labels)
                % Merge with most similar neighbor
                curr_mean = mean(gray(mask));
                best_diff = inf;
                best_nb = neighbor_labels(1);
                for nb = neighbor_labels'
                    nb_mask = (labels == nb);
                    nb_mean = mean(gray(nb_mask));
                    if abs(curr_mean - nb_mean) < best_diff
                        best_diff = abs(curr_mean - nb_mean);
                        best_nb = nb;
                    end
                end
                labels(mask) = best_nb;
                changed = true;
            end
        end
    end
end

% Relabel consecutively
old_labels = unique(labels(:));
new_labels = zeros(max(old_labels), 1);
for i = 1:length(old_labels)
    new_labels(old_labels(i)) = i;
end
for i = 1:row
    for j = 1:col
        labels(i,j) = new_labels(labels(i,j));
    end
end
end