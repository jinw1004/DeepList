function list = genList(id_batch)
% id_batch是当前feature batch 内各个feature对应的id
list = [];
for i=1:numel(id_batch)
    id_prb = id_batch(i);
    idx_prb = i;
    idx_pos = setdiff(find(id_prb == id_batch), i);
    idx_neg = setdiff(1:numel(id_batch), [i, idx_pos]);
    for j=1:numel(idx_pos)
        single_pos_idx = idx_pos(j);
        single_list = [idx_prb, single_pos_idx, idx_neg];
        list = cat(1, list, single_list);
    end
end
        