function [labels, num_anchors] = generate_coarse_anchors_func(img, cluster_num)
%GENERATE_COARSE_ANCHORS_FUNC Generate coarse-grained anchors via superpixels
%   Based on GMMR multi-granularity decomposition Stage 1
%
%   Inputs:
%     img         - uint8 color image (H x W x 3)
%     cluster_num - number of desired clusters
%   Outputs:
%     labels      - H x W label map (integer labels 1..num_anchors)
%     num_anchors - number of coarse anchors

    [row, col, ~] = size(img);
    N_pixels = row * col;
    target_anchors = max(cluster_num * 15, 20);


    if exist('superpixels', 'file') == 2
        [labels, num_anchors] = superpixels(img, target_anchors);
        labels = double(labels);
    else
        gray = double(rgb2gray(img));
        [gx, gy] = gradient(gray);
        grad_mag = sqrt(gx.^2 + gy.^2);

        se = strel('disk', 3);
        marker = imerode(grad_mag, se);
        recon = imreconstruct(marker, grad_mag);
        recon_smooth = imgaussfilt(recon, 2);
        ws = watershed(recon_smooth);

        labels = double(ws);
        labels(labels == 0) = 1;  

        old_ids = unique(labels(:));
        new_map = zeros(max(old_ids), 1);
        for ii = 1:length(old_ids)
            new_map(old_ids(ii)) = ii;
        end
        labels = arrayfun(@(x) new_map(x), labels);
        num_anchors = length(old_ids);

        if num_anchors > target_anchors * 3
            min_size = round(N_pixels / (target_anchors * 2));
            labels = merge_small_func(labels, min_size, img);
            num_anchors = max(labels(:));
        end
    end

    fprintf('  Coarse anchors: %d\n', num_anchors);
end

function labels = merge_small_func(labels, min_size, img)
    [row, col, ~] = size(img);
    gray = double(rgb2gray(img));
    changed = true;
    while changed
        changed = false;
        num_r = max(labels(:));
        for r = 1:num_r
            mask_r = (labels == r);
            if sum(mask_r(:)) < min_size && sum(mask_r(:)) > 0
                dilated = imdilate(mask_r, strel('square', 3));
                border = dilated & ~mask_r;
                nb_labels = unique(labels(border));
                nb_labels(nb_labels == r) = [];
                nb_labels(nb_labels == 0) = [];
                if ~isempty(nb_labels)
                    curr_mean = mean(gray(mask_r));
                    best_diff = inf; best_nb = nb_labels(1);
                    for nb = nb_labels'
                        nb_mean = mean(gray(labels == nb));
                        dd = abs(curr_mean - nb_mean);
                        if dd < best_diff
                            best_diff = dd; best_nb = nb;
                        end
                    end
                    labels(mask_r) = best_nb;
                    changed = true;
                end
            end
        end
    end
    % Relabel
    old_ids = unique(labels(:));
    new_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for ii = 1:length(old_ids)
        new_map(old_ids(ii)) = ii;
    end
    for ii = 1:row
        for jj = 1:col
            labels(ii, jj) = new_map(labels(ii, jj));
        end
    end
end