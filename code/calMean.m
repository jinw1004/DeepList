clear;clc;
matdir = 'E:\wjin\Re-ID-My-Exp-2015-5-15\DeepEmbed_ReId-2015-7-14\mat\';
imdb = importdata(fullfile(matdir, 'imdb_viper_single.mat'));

% 相当于每幅图像随机增加20次
augopts.rotation.ang = [-8 8];
augopts.translation.border = [16 10];
augopts.imsize = [160 80];
im_mean = [];
for i=1:20
    disp(num2str(i));
    im = imdb.data;
    im_aug = augIm(im, 'random', augopts);
    mean_tmp = mean(im_aug, 4);
    im_mean = cat(4, im_mean, mean_tmp);
end
mean = mean(im_mean, 4);
imdb.mean = mean;
save(fullfile(matdir, 'imdb_viper_single.mat'), 'imdb', '-v7.3');
