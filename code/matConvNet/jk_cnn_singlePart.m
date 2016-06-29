function res = jk_cnn_singlePart(net, res_input, dzdy, varargin)
% 网络结构1的一个子部分，即针对单个part的部分。由于对于arch1而言，它是由三个相同的
%子part并联构成的，最后一层也是每一个子part所提取出来的特征拼起来得到的，所以有了这个
%子part就可以了。在这个函数中，我们完成这个子part的前向和后向的计算过程。
if nargin<=2 || isempty(dzdy)
    forward = 1;
else
    forward = 0;
end

opts.res = [] ;
opts.conserveMemory = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.mode = 'train'; %train or test, bnorm layer will use different mean and standard deviation
opts = vl_argparse(opts, varargin);

res = res_input;
n = numel(net.layers);
% forward or backward
if forward
    for i=1:n
        layer = net.layers{i};
        switch layer.type
            case 'conv'
                res(i+1).x = vl_nnconv(res(i).x, layer.filters, layer.biases, 'pad', layer.pad, 'stride', layer.stride) ;
            case 'pool'
                res(i+1).x = vl_nnpool(res(i).x, layer.pool, 'pad', layer.pad, 'stride', layer.stride, 'method', layer.method) ;
            case 'relu'
                res(i+1).x = vl_nnrelu(res(i).x) ;
            case 'tanh'
                res(i+1).x = vl_nntanh(res(i).x);  
            case 'dropout'
                if opts.disableDropout
                    res(i+1).x = res(i).x ;
                elseif opts.freezeDropout
                    [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', layer.rate, 'mask', res(i+1).aux) ;
                else
                    [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', layer.rate) ;
                end
            case 'bnorm'
                if strcmp(opts.mode, 'test')
                    res(i+1).x = vl_nnbnorm(res(i).x, layer.filters, layer.biases, 'moments', layer.moments) ;
                else
                    res(i+1).x = vl_nnbnorm(res(i).x, layer.filters, layer.biases) ;
                end
            case 'l2norm'
                FP_unnorm = squeeze(double(gather(res(i).x)));% 变成2维矩阵
                [res(i+1).x, res(i+1).aux] = jk_cnn_l2norm(FP_unnorm);
                
            case 'softmaxloss'
                res(i+1).x = vl_nnsoftmaxloss(res(i).x, layer.class) ;    
            otherwise
                error('Unknown layer type %s', layer.type) ;
        end
    end
else
    % backward
    res(n+1).dzdx = dzdy;
    for i=n:-1:1
        layer = net.layers{i};
        switch layer.type
            case 'conv'
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = vl_nnconv(res(i).x, layer.filters, layer.biases, res(i+1).dzdx, ...
                                                                'pad', layer.pad, 'stride', layer.stride) ;
            case 'pool'
                res(i).dzdx = vl_nnpool(res(i).x, layer.pool, res(i+1).dzdx, ...
                  'pad', layer.pad, 'stride', layer.stride, 'method', layer.method) ;
            case 'relu'
                res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
            case 'tanh'
                res(i).dzdx = vl_nntanh(res(i).x, res(i+1).dzdx);
            case 'dropout'
                if opts.disableDropout
                  res(i).dzdx = res(i+1).dzdx ;
                else
                  res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
                end
            case 'bnorm'
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}, res(i).dzdw{3}] = ...
                  vl_nnbnorm(res(i).x, layer.filters, layer.biases, res(i+1).dzdx) ; % dzdw(3) is the moment of this batch
                % multiply the moments update by the number of images in the batch
                % this is required to make the update additive for subbatches
                % and will eventually be normalized away
%                 res(i).dzdw{3} = res(i).dzdw{3} * size(res(i).x,4) ;
            case 'l2norm'
                FP_unnorm = squeeze(double(gather(res(i).x)));
                dzdx_output = jk_cnn_l2norm(FP_unnorm, res(i+1).dzdx);%得到二维的梯度
                dzdx_output = reshape(dzdx_output, [1, 1, size(dzdx_output,1), size(dzdx_output,2)]);
%                 if gpuMode
%                     dzdy = gpuArray(single(dzdy));
%                 end
                res(i).dzdx = single(dzdx_output); %四维的, in CPU
            case 'softmaxloss'
                res(i).dzdx = vl_nnsoftmaxloss(res(i).x, layer.class, res(i+1).dzdx) ;
            otherwise
                error('Unknown layer type %s', layer.type) ;
        end
        if opts.conserveMemory
          res(i+1).dzdx = [] ;
        end
%         if gpuMode
%     %       gpu =gpuDevice ;
%     %       fprintf('bkg: %d %.1f\n', i, gpu.FreeMemory/1024^2) ;
%     %       wait(gpuDevice) ;
%         end
    end
end
