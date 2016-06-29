function im_aug = augIm(im_ori, type, opts)
% do transformation on the original images, the supported transformations
% include translation and rotation, now we haven't support illumination
% transformation
% input: im, [170, 68, 3, num]
% output:
% im_aug, [160 60, 3, num]
if nargin <= 2
    opts.rotation.ang = [-8 8];
    opts.translation.border = [10 8];
    opts.imsize = [160 60];
    opts.saturation.gamma = [0.8 1];
end

num = size(im_ori, 4);
im_aug = zeros([opts.imsize(1), opts.imsize(2), 3, num]);
for i=1:num
    im = im_ori(:,:,:,i);
    
    % rotation
    switch type
        case 'random'
            ang = rand(1)*(opts.rotation.ang(2) - opts.rotation.ang(1)) + opts.rotation.ang(1);
            sample = 2*rand(1,2)-1;%[-1 1]
            if rand(1) > 0.5
                im = fliplr(im);
            end
        case 'center'
            ang = 0;
            sample = 0;
        otherwise
            error('unknow transformation type %s\n', type);
    end
    im_r = imrotate(im, ang, 'bilinear', 'crop');
    
    % translation
    center = [size(im_r, 1) size(im_r, 2)]/2;
    patch_center = floor(center + sample.*(opts.translation.border./2));
    im_t = im_r(patch_center(1)-floor(opts.imsize(1)/2)+1:patch_center(1)+floor(opts.imsize(1)/2), ...
                patch_center(2)-floor(opts.imsize(2)/2)+1:patch_center(2)+floor(opts.imsize(2)/2), :);
            
    %做饱和度增强
%     switch type
%         case 'random'
%             im_s = rgb2hsv(uint8(im_t));
%             H = im_s(:,:,1);S = im_s(:,:,2); V = im_s(:,:,3);
%             gamma = rand(1)*(opts.saturation.gamma(2) - opts.saturation.gamma(1)) + opts.saturation.gamma(1);
%             if rand(1)>0.5
%                 gamma = 1/gamma;
%             end
% %             S = imadjust(S, [0 1], [0 1], gamma);%对1/x随机采样
%             V = imadjust(V, [0 1], [0 1], gamma);
%             im_s = cat(3, H, S, V);
%             im_s = hsv2rgb(im_s);
%             im_s = im_s*255; % [0 255]
%         case 'center'
%             im_s = im_t; % [0 255]
%         otherwise
%             error('unknow transformation type %s\n', type);
%     end
    
    im_aug(:,:,:,i) = im_t;
end
    
