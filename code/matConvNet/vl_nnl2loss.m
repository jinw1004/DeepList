function Y = vl_nnl2loss(X,c,dzdy)
%计算预测输出的L2损失loss = sum((x-c).^2)
%X是输入，c是ground truth，对于损失层而言，dzdy应该是和X同维的1向量
%前向时，Y是计算得到的loss；后向时，Y是计算得到的梯度dzdx
%前向：Y = vl_nnl2loss(X,c)，X为列向量D*N，c为同维的列向量，Y为实数
%后向：Y = vl_nnl2loss(X,c,dzdy),X为列向量，c为同维的列向量，dzdy应该是1
b = squeeze(X);% X是[1 1 28 100]大小的输出
%b(c==0) = 0;%这样做相当于对于那些没看到的关节就不计算它们的损失,c是[28 100]大小的输出
b(c==-0.5) = -0.5;
b = reshape(b,[1,1,size(b,1),size(b,2)]);
c = reshape(c, [1,1,size(c,1),size(c,2)]);
if nargin <= 2
    % forward, calculate the loss
    Y = sum((b - c).^2, 3);
    Y = sum(Y,4); %返回的是一个batch的所有loss，因此把这些loss要加起来
    Y = squeeze(Y);
else
    Y = 2*(b - c)*dzdy;%其实这里乘以dzdy没啥用
end

