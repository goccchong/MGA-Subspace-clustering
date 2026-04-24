function [fine_labels, ls] = generate_fine_anchors(img, coarse_labels, delta)
% Generate fine-grained anchors by refining coarse anchors
% Each coarse anchor is split into delta fine anchors
% GMMR multi-stage decomposition (Eq. 5-6 in paper)
%
% Inputs:
%   img           - uint8 color image
%   coarse_labels - row x col coarse label map
%   delta         - multiplicity factor (ls = delta * ld)
%
% Outputs:
%   fine_labels   - row x col fine label map
%   ls            - number of fine anchors

[row, col, depth] = size(img);
img_d = double(img);
ld = max(coarse_labels(:));
ls_target = delta * ld;

fine_labels = zeros(row, col);
current_label = 0;

for r = 1:ld
    mask = (coarse_labels == r);
    region_size = sum(mask(:));
    
    if region_size < delta * 2
        % Too small to split further
        current_label = current_label + 1;
        fine_labels(mask) = current_label;
        continue;
    end
    
   
    [ri, ci] = find(mask);
    features = zeros(length(ri), depth + 2); 
    for d = 1:depth
        ch = img_d(:,:,d);
        features(:, d) = ch(mask);
    end

    features(:, depth+1) = ri / row;
    features(:, depth+2) = ci / col;
    
    num_sub = min(delta, max(2, round(region_size / 50)));
    
    if num_sub <= 1
        current_label = current_label + 1;
        fine_labels(mask) = current_label;
    else
        try
            sub_labels = kmeans(features, num_sub, 'Replicates', 3, 'MaxIter', 50);
        catch
            sub_labels = ones(size(features, 1), 1);
            num_sub = 1;
        end
        
        for s = 1:num_sub
            current_label = current_label + 1;
            sub_mask = (sub_labels == s);
            idx = sub2ind([row, col], ri(sub_mask), ci(sub_mask));
            fine_labels(idx) = current_label;
        end
    end
end

unassigned = (fine_labels == 0);
if any(unassigned(:))
    current_label = current_label + 1;
    fine_labels(unassigned) = current_label;
end

ls = current_label;
fprintf('Fine anchors generated: %d\n', ls);
end