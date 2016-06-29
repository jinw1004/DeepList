function y = vl_nntanh(x, dzdy)
%实现tanh激活函数，范围在(-1 1)之间
%前向：y = tanh(x)
%后向：y = dzdy * (1-tanh^2(x)),后向的y应该对应dzdx
%dzdx = dzdy*dydx, dydx = 1-tanh^2(x)
%注意，这里的x, dzdy应该都是四维的
if nargin <= 1
    y = tanh(x);
else 
    output = tanh(x);
    dydx = 1 - output.*output;
    y = dzdy .* dydx;
end