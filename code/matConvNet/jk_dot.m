function [y, dzdw, dzdb] = jk_dot(x, weight, bias, dzdy)
% x: 1-by-n(i)
% weight: n(i)-by-n(i+1)
% bias: 1-by-n(i+1)
% dzdy: 1-by-n(i+1)
if nargin < 4
    % forward
    y = bsxfun(@plus, x*weight, bias);
else
    % backward
    num = size(x, 1);
    y = dzdy*weight'; %dzdx
    dzdw = x'* dzdy;
    dzdb = sum(dzdy, 1);
end