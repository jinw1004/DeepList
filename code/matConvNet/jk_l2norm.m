function y = jk_l2norm(x, dzdy)
% x: [1 d]
% dydz = y(1-y)

if nargin <= 1
  y = x./sqrt(sum(x.*x, 2));
  %y = x .* (x > single(0)) ;
else
  y = dzdy ; % [1 d]
  %y(x <= 0) = 0 ;
  x_norm = sqrt(sum(x.*x, 2));
  d = size(x, 2);
  dydx = (1/x_norm)*eye(d) - (1/(x_norm^1.5))*(x'*x);
%   dydx = (1/x_norm)*eye(n); % 近似，等价于原来的做法
  y = y * dydx ;
end