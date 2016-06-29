function [y, norm] = jk_cnn_l2norm(x, dzdy)
% the L2-normalization layer
% x : [d n], d is the feature dimension
% dzdy: [d n]
if nargin == 1
    x_norm = sqrt(sum(x.^2, 1)); % [1 n]
    y = bsxfun(@rdivide, x, x_norm);
    norm = x_norm;
else
%     x_norm = sqrt(sum(x.^2, 1));
    [d,n] = size(x);
    y = zeros(size(x));
    for i=1:n
        x_i = x(:,i);
        dzdy_i = dzdy(:,i);
        norm_x_i = sqrt(sum(x_i.^2, 1));
        dy_idx_i = (1/norm_x_i)*eye(d) - (1/(norm_x_i^1.5))*(x_i*x_i');
        dzdx_i = dzdy_i'*dy_idx_i;
        y(:,i) = dzdx_i';
    end
end

%=========== bypass the L2-normalization layer ===================
% if nargin == 1
%     x_norm = sqrt(sum(x.^2, 1)); % [1 n]
%     y = x;
%     norm = x_norm;
% else
% %     x_norm = sqrt(sum(x.^2, 1));
%     [d,n] = size(x);
%     y = zeros(size(x));
%     y = dzdy;
% end