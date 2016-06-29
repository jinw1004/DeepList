function [triplet002, triplet012, triplet001] = sampleTriplet(dist, sim)
%This function samples triplets per epoch. The sampling scheme should be
%fine tuned. There are several sampling schemes. Here, we first use the
%simplest one. Triplet001 are all used, triplet 002 and triplet 012 are
%smapled with hard negatives
% dist: probe-by-gallery (316*4-by-316*4)
triplet002 = []; triplet012 = []; triplet001 = [];
[~, ind] = sort(dist, 2, 'ascend');
for p = 1:size(dist, 1)
    % first, sample triplet002
    dist_p = dist(p,:);
    sim_p = sim(p,:);
    
    level0 = find(sim_p == 0);
    level1 = find(sim_p == 1);
    level2 = find(sim_p == 2);
    
%     dist_n = dist_p(level2);
%     [~, index] = sort(dist_n, 2, 'ascend');
    index = setdiff(ind(p,:), [level0, level1]);
    level2_sampled = index(1:50); %这个数字很重要，每个probe采样100个negative
    
    [X, Y, Z] = meshgrid(p, setdiff(level0, p), level2_sampled);
    triplet002 = cat(1, triplet002, [X(:), Y(:), Z(:)]);
    
    % sample triplet012
    [X, Y, Z] = meshgrid(p, level1, level2_sampled);
    triplet012 = cat(1, triplet012, [X(:), Y(:), Z(:)]);
    
    % sample triplet 001
    [X, Y, Z] = meshgrid(p, setdiff(level0, p), level1);
    triplet001 = cat(1, triplet001, [X(:), Y(:), Z(:)]);
end 