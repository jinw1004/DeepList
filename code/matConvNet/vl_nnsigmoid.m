function y = vl_nnsigmoid(x, dzdy)
if nargin <= 1
    y = 1./(exp(-x)+1);
else 
    output = 1./(exp(-x)+1);
    dydx = output.*(1-output);
    y = dzdy .* dydx;
end