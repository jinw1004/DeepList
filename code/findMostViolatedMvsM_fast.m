function [slack, y_star] = findMostViolatedMvsM_fast(Sim_pos, Sim_neg, y_gt, type)
% find the most violated constraint. Actually, it solves the following
% optimization problem:
% y_star = arg_max F(x, yi)-F(x, y_hat)+delta(yi, y_hat) and F(x, y_hat) is
% fixed
% seperation oracle
% here, the y_hat is always [1 1 1 .. 1],这里只给出negative内的排序
% Sim is the similarity vector
switch type
    case 'auc'
        a = 1/numel(Sim_neg)*ones(1, numel(Sim_neg));
        phi_k = cumsum(a);
        phi_k = [0, phi_k];
    case '2_order'
        x = 1:numel(Sim_neg);
        a = 1./x.^2;
        phi_k = cumsum(a)/sum(a);
        phi_k = [0, phi_k];
    case '1_order'
        x = 1:numel(Sim_neg);
        a = 1./x;
        phi_k = cumsum(a)/sum(a);
        phi_k = [0, phi_k];
    otherwise
        error('unknow loss type');
end

y_star = ones(size(y_gt));
slack_all = zeros(numel(Sim_pos), 1);
for i=1:size(y_gt, 1)
    s_p = Sim_pos(i);
    s_neg = Sim_neg;
    s_p = repmat(s_p, [1, numel(s_neg)]);
    delta_sim = s_p - s_neg;
    [value, sInd] = sort(delta_sim, 2, 'ascend');
    ksi_k = zeros(size(value));
    for j=1:numel(value)
        ksi_k(j) = -1*sum(value(1:j)) + sum(value(j+1:end));
    end
    ksi_k = ksi_k/numel(s_neg);
    ksi_k = [sum(value)/numel(s_neg), ksi_k];
    
    H_k = phi_k + ksi_k - sum(value)/numel(s_neg);
    [slack_all(i), index] = max(H_k);
    % index前的全为-1
    y_star(i,sInd(1:index-1)) = -1;
end
slack = mean(slack_all(:));

    
    
    


