function W = build_anchor_adjacency_func(anchor_features, knn_k)
%BUILD_ANCHOR_ADJACENCY_FUNC KNN-based adjacency for anchor graph Laplacian

    num_a = size(anchor_features, 1);
    knn_k = min(knn_k, num_a - 1);

    D_dist = pdist2(anchor_features, anchor_features);
    sigma_w = mean(D_dist(:)) + eps;

    W = zeros(num_a);
    for i = 1:num_a
        [~, idx_sorted] = sort(D_dist(i, :));
        neighbors = idx_sorted(2:knn_k+1);
        for j = neighbors
            W(i, j) = exp(-D_dist(i, j)^2 / (2 * sigma_w^2));
        end
    end
    W = (W + W') / 2;
end