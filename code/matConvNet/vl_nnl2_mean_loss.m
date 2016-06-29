function Y = vl_nnl2_mean_loss(X,c,dzdy)
%计算预测输出的L2损失loss = sum((x-c).^2)
%X是输入，c是ground truth，对于损失层而言，dzdy应该是和X同维的1向量
%前向时，Y是计算得到的loss；后向时，Y是计算得到的梯度dzdx
%前向：Y = vl_nnl2loss(X,c)，X为列向量D*N，c为同维的列向量，Y为实数
%后向：Y = vl_nnl2loss(X,c,dzdy),X为列向量，c为同维的列向量，dzdy应该是1
b = squeeze(X);% X是[1 1 28 100]大小的输出
%b(c==0) = 0;%这样做相当于对于那些没看到的关节就不计算它们的损失,c是[28 100]大小的输出
%计算每幅图像invisible joint的个数
count = (c~=-0.5);
nc = sum(count,1)/2;
b(c==-0.5) = -0.5;
lamda = 1;
sqrt_lamda = sqrt(lamda);
PX = b(1:14,:); PY = b(15:28,:);
LX = c(1:14,:); LY = c(15:28,:);
tb = [sqrt_lamda*PX; PY]; tc = [sqrt_lamda*LX; LY];
% tb = [PX; PY]; tc = [LX; LY];
tb = reshape(tb,[1,1,size(tb,1),size(tb,2)]);
tc = reshape(tc, [1,1,size(tc,1),size(tc,2)]);
if nargin <= 2
    % forward, calculate the loss
    t_nc = reshape(nc, [1,1,1,size(nc,2)]);% [1 1 1 128]
    Y = sum((tb - tc).^2, 3);
    Y = Y./t_nc; %求每个关节的平均误差
    Y = sum(Y,4); %返回的是一个batch的所有loss，因此把这些loss要加起来
    Y = squeeze(Y);
else
    t_nc = repmat(nc, [28 1]);%[28 100]
    t_nc = reshape(t_nc, [1 1 28 size(nc,2)]);
    Y = 2*(tb - tc)*dzdy;%其实这里乘以dzdy没啥用,Y是[1 1 28 100]
    Y = Y./t_nc;
end

