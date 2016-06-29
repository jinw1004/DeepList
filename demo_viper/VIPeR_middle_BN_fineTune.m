%说明：
%我们在ListMLE_018中加入similarity level，这算是最后一招了，看看管不管用。我们当然是希望它能够
%管用的。看之前SimLev_004的结果似乎还是可以的


clear;clc;
% addpath(genpath('E:\wjin\Re-ID-My-Exp-2015-5-15\DeepEmbed_ReId-2015-7-14'));
% run('E:\wjin\Re-ID-My-Exp-2015-5-15\DeepEmbed_ReId-2015-7-14\code\jkcnn\vl_setupnn.m') ;

matdir = 'E:\wjin\Re-ID-My-Exp-2015-5-15\DeepEmbed_ReId-2015-7-14\mat\';
resDir = 'E:\wjin\Re-ID-My-Exp-2015-5-15\DeepEmbed_ReId-2015-7-14\result_Exp_VIPeR_bn_16_03_02\';
mkdir(resDir);

%=========image sampling layer========
% for example, each batch we sample 20 image pairs(x, x+)
imdb = importdata(fullfile(matdir, 'imdb_viper_Middle.mat'));
partition = importdata(fullfile(matdir, 'partition_random.mat'));
simLev = importdata(fullfile(matdir, 'sim3.mat'));

% train options
opts.numEpochs = 300 ;
batchsize = 40; % 20个id,每个id有4幅图像。对于分view的来说，2幅来自A，两幅来自B，对于不分view的来说，直接随机选了
% batchsize = 64;
g = 0.5;
opts.useGpu = true ;
opts.learningRate = [1e-4*ones(1, 200), 5e-5*ones(1, 150), 2e-5*ones(1, 50)];
opts.continue = false ;
opts.conserveMemory = false ;
opts.prefetch = false ;
opts.weightDecay = 1e-3;
opts.momentum = 0.9;
    
for trial = 1:10
    
    load(fullfile(matdir, 'preTrain_mid_1_1200.mat'));

    % result template
    nLayers = numel(net.layers);
    res_template = struct('x', cell(1, nLayers+1), ...
        'dzdx', cell(1,nLayers+1), ...
        'dzdw', cell(1,nLayers+1), ...
        'aux', cell(1,nLayers+1));
    
    id_train = partition(trial).trnSg;
    id_tst = partition(trial).tstSg;

    % 性能记录用的变量
    info.train.loss = [];
    info.train.rank1 = [];
    info.val.loss = [];
    info.val.rank1 = [];
    augopts.rotation.ang = [-8 8];
    augopts.translation.border = [10 10];
    augopts.imsize = [100 100];
    augopts.saturation.gamma = [1 1];
    lossType = 'auc';

    %==========================================================================
    % 开始训练
    for epoch = 1:400
        if epoch > 200
            batchsize = 53;
        end
        info.train.loss(end+1) = 0;
        info.train.rank1(end+1) = 0;
        info.val.loss(end+1) = 0;
        info.val.rank1(end+1) = 0;

        index_random = randperm(numel(id_train));
        id_train_epoch = id_train(index_random);
    %     id_train_epoch = id_train;

        for t=1:batchsize:numel(id_train_epoch)
            ksi = 0;
            fprintf('trail %d, training: epoch %02d: processing batch %3d of %3d ...\n', trial, epoch, ...
                fix(t/batchsize)+1, ceil(numel(id_train_epoch)/batchsize)) ;
            batch_time = tic;
            batch = id_train_epoch(t:min(t+batchsize-1, numel(id_train)));
            im_batch = getbatch(imdb, batch); % im_batch: 48-by-128-by-3-by-40，每个id最多选出4幅图像
            im_batch = single(im_batch);

            %=========image split layer===========
            % resize images to [68 68], then rotate images with [-5 5], sample a [60
            % 160] region as new samples
            im_aug = augIm(im_batch, 'random', augopts);
    %         im_aug = rgbJitter(im_aug, imdb.vec, imdb.val); % no use
            im_aug = bsxfun(@minus, im_aug, imdb.mean);

            %=========start training, forward and backward=========
            res = res_template; clear im_batch;
            im_batch = single(im_aug); clear im_aug;
            FP_unnorm = [];
            if opts.useGpu
                im_batch = gpuArray(im_batch);
            end
            res(1).x = im_batch;
            res = jk_cnn_singlePart(net, res, [], 'mode', 'train'); % train mode, forward
            FP_unnorm = cat(1, FP_unnorm, squeeze(double(gather(res(end).x))));

            % forward L2 layer
            clear res_L2;
            res_L2 = struct('x', cell(1,2), 'dzdx', cell(1,2), 'aux', cell(1,2));
            res_L2(1).x = FP_unnorm;
            clear FP_unnorm;
            [res_L2(2).x, res_L2(2).aux] = jk_cnn_l2norm(res_L2(1).x);
            FP = res_L2(2).x; mean_mag = mean(res_L2(2).aux);

            %================ listwise loss =======================
            % sample lists。思考：listwise loss和triplet
            % loss到底有什么不同呢？本质上就是是否把它们的梯度加进去的问题
            dist = slmetric_pw(FP, FP, 'sqdist'); 
            sim = 4-dist;
            id_batch = repmat(1:numel(batch), [2 1]); id_batch = id_batch(:)';
            List = genList(id_batch); % [prb pos neg1 ... negk]，每个list只有一个pos

            % import similarity level
            id_batch_abs = repmat(batch, [2,1]); id_batch_abs = id_batch_abs(:)';
            simLev_batch = simLev(id_batch_abs, id_batch_abs);


            dzdfp_list = cell(1,2);
            dzdfp_list{1} = zeros(size(FP));
            dzdfp_list{2} = zeros(size(FP));
            numTriplet = 0;
            for i=1:size(List, 1)
                list = List(i,:);
                prb = list(1); pos = list(2); neg = list(3:end);
                lev1 = find(simLev_batch(prb, :) == 1);
                lev2 = find(simLev_batch(prb, :) == 2);
                lev1 = intersect(lev1, neg);
                lev2 = intersect(lev2, neg);

                sim_pos = sim(prb, pos);
                sim_lev1 = sim(prb, lev1);
                sim_lev2 = sim(prb, lev2);
                [~, idx_lev1] = sort(sim_lev1, 'descend');
                perm_lev1 = lev1(idx_lev1);
                [~, idx_lev2] = sort(sim_lev2, 'descend');
                perm_lev2 = lev2(idx_lev2);

                permutation = [pos, perm_lev1, perm_lev2];
    %             permutation = [pos, perm_neg]; % the selected permutation which has the maximum likelihood in the equivalent set

                 for k=1:1
                    pos_k = permutation(k); neg_k = permutation(k+1:end);
                    sim_pos_k = sim(prb, pos_k); sim_neg_k = sim(prb, neg_k);

                    g_k = linspace(1.5, 0.5, numel(sim_neg_k));
                    [~, index] = sort(sim_neg_k, 'descend');
                    [~, iindex] = sort(index, 'ascend');
                    g_k = g_k(iindex);
    %                 g_k = 0.5;

                    sim_pos_k = sim_pos_k/0.4;
                    sim_neg_k = (sim_neg_k + g_k)/0.4;

                    sim_list_k = [sim_pos_k, sim_neg_k];
                    MAX = max(sim_list_k(:));
                    sim_list_k = sim_list_k - MAX;

                    sim_pos_k = sim_pos_k - MAX;
                    sim_neg_k = sim_neg_k - MAX;
                    Z_k = sum(exp(sim_list_k));
                    for j=1:numel(neg_k)
                        % for positive image
                        dzdp_pos_k = -(2/Z_k)*exp(sim_neg_k(j))*FP(:,prb);
                        dzdfp_list{k}(:,pos_k) = dzdfp_list{k}(:,pos_k) + dzdp_pos_k;

                        % for negative image
                        dzdp_neg_k = -(-2/Z_k)*exp(sim_neg_k(j))*FP(:,prb);
                        dzdfp_list{k}(:,neg_k(j)) = dzdfp_list{k}(:,neg_k(j)) + dzdp_neg_k;

                        % for probe image
                        dzdp = -(2/Z_k)*exp(sim_neg_k(j))*(FP(:,pos_k) - FP(:,neg_k(j)));
                        dzdfp_list{k}(:,prb) = dzdfp_list{k}(:,prb) + dzdp;

                        numTriplet = numTriplet + 1;
                    end
                end
            end                  
    %         dzdfp = dzdfp_list/2000; %k=1,A=3000;k=2,A=6000
            dzdfp = (dzdfp_list{1} + 0.25*dzdfp_list{2})/2000;

            % BP L2 layer
            res_L2(2).dzdx = dzdfp;
            res_L2(1).dzdx = jk_cnn_l2norm(res_L2(1).x, dzdfp);
            clear dzdfp;

            % BP cnn layer
            dzdy = res_L2(1).dzdx;
            dzdy = reshape(dzdy, [1, 1, size(dzdy,1), size(dzdy,2)]);
            if opts.useGpu
                dzdy = gpuArray(single(dzdy));
            end
            res = jk_cnn_singlePart(net, res, dzdy);

            % update parameters
            lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
            for l=1:numel(net.layers)
                ly = net.layers{l} ;
                if strcmp(ly.type, 'conv'), 
                    % 计算更新增量
                    ly.filtersMomentum = opts.momentum * ly.filtersMomentum ...
                                         - opts.weightDecay * ly.filtersWeightDecay * lr * ly.filtersLearningRate * ly.filters ...
                                         - lr * ly.filtersLearningRate * res(l).dzdw{1};

                    ly.biasesMomentum = opts.momentum * ly.biasesMomentum ...
                                        - opts.weightDecay * ly.biasesWeightDecay * lr * ly.biasesLearningRate * ly.biases ...
                                        - lr * ly.biasesLearningRate * res(l).dzdw{2};
                    % 更新参数
                    ly.filters = ly.filters + ly.filtersMomentum ;
                    ly.biases = ly.biases + ly.biasesMomentum ;
                    net.layers{l} = ly ;
                end
                if strcmp(ly.type, 'bnorm'), 
                    % 计算更新增量
                    ly.filtersMomentum = opts.momentum * ly.filtersMomentum ...
                                         - opts.weightDecay * ly.filtersWeightDecay * lr * ly.filtersLearningRate * ly.filters ...
                                         - lr * ly.filtersLearningRate * res(l).dzdw{1};

                    ly.biasesMomentum = opts.momentum * ly.biasesMomentum ...
                                        - opts.weightDecay * ly.biasesWeightDecay * lr * ly.biasesLearningRate * ly.biases ...
                                        - lr * ly.biasesLearningRate * res(l).dzdw{2};
%                     ly.momentsMomentum = opts.momentum * ly.momentsMomentum ...
%                                         - opts.weightDecay * ly.momentsWeightDecay * lr * ly.momentsLearningRate * ly.moments ...
%                                         - lr * ly.momentsLearningRate * res(l).dzdw{3} ; %这个算出来实际上没什么用处
                    % 更新参数
                    ly.filters = ly.filters + ly.filtersMomentum ;
                    ly.biases = ly.biases + ly.biasesMomentum ;
                    ly.moments = (1-ly.momentsLearningRate) * net.layers{l}.moments + ...
                                ly.momentsLearningRate * res(l).dzdw{3} ;
                    net.layers{l} = ly ;
                end
            end
            batch_time = toc(batch_time);
            fprintf(' %.2f s (%.1f persons/s)\n', batch_time, numel(batch)/ batch_time) ;
            fprintf('mean magnitude, batch loss, validTriplet: %.2f, %.4f, %d\n', mean_mag, ksi/size(List, 1), 0);
        end
    %     info.train.loss(end) = ksi;

        % calculate test loss and cmc rank
        fprintf('\n======================= validation set ===================\n');
    %     [loss_tst, cmc_tst] = cnn_test_topHeavy(imdb, id_tst, net, batchsize, opts.useGpu, 'auc');
        clear FP_unnorm FP im_batch im_aug res FP_batch probe gallery FP_probe FP_gallery dist_pg dist_final;
        loss_tst = 0; FP = [];
        for t=1:batchsize:numel(id_tst)
            fprintf('validation: processing batch %3d of %3d ...\n', fix(t/batchsize)+1, ceil(numel(id_tst)/batchsize)) ;
            batch = id_tst(t:min(t+batchsize-1, numel(id_tst)));
            im_batch = getbatch(imdb, batch);
            im_batch = single(im_batch);
            im_aug = augIm(im_batch, 'center', augopts);

            im_aug = bsxfun(@minus, im_aug, imdb.mean); 
            clear im_batch; im_batch = im_aug; clear im_aug;
            FP_unnorm = [];

            res = res_template;
            im_batch = single(im_batch);
            if opts.useGpu
                im_batch = gpuArray(im_batch);
            end
            res(1).x = im_batch;
            res = jk_cnn_singlePart(net, res, [], 'disableDropout', true, 'mode', 'test');

            FP_unnorm = cat(1, FP_unnorm, squeeze(double(gather(res(end).x))));


            % forward L2 layer
            clear res_L2;
            res_L2 = struct('x', cell(1,2), 'dzdx', cell(1,2), 'aux', cell(1,2));
            res_L2(1).x = FP_unnorm;
            clear FP_unnorm;
            [res_L2(2).x, res_L2(2).aux] = jk_cnn_l2norm(res_L2(1).x);
            FP_batch = res_L2(2).x;

            % to obtain the feature maps for all images
            FP = cat(2, FP, FP_batch);
        end

        % calculate dist
        probe = 1:2:size(FP, 2); probe = probe(:)'; % camb as probe
        gallery = 2:2:size(FP, 2); gallery = gallery(:)'; % cama as gallery
        FP_probe = FP(:, probe); FP_gallery = FP(:, gallery);
        dist_pg = slmetric_pw(FP_probe, FP_gallery, 'sqdist');

        % 镜像对称后的四个距离的总和, l为origion，r为symmetric
        dist_final = dist_pg;

        % for each probe image, calculate the slack
        for i=1:size(dist_final,1)
            if mod(i, 10) == 0
                fprintf('.');
            end
            index_pos = i;
            index_neg = setdiff(1:size(dist_final,1), index_pos);

            Sim_pos = -dist_final(i, index_pos); % 只针对SvsS的情况
            Sim_neg = -dist_final(i, index_neg);

            % calculate slack
            y_gt = ones(numel(index_pos), numel(index_neg));
            [slack, ~] = findMostViolatedMvsM_fast(Sim_pos, Sim_neg, y_gt, lossType);

            % calculate loss
            loss_tst = loss_tst + slack;
        end

        cmc_tst = evaluate_cmc(-dist_final)/numel(id_tst);


        fprintf('\n======================= training set ===================\n'); 
    %     [loss_trn, cmc_trn] = cnn_test_topHeavy(imdb, id_train, net, batchsize, opts.useGpu, 'auc');
        clear FP_unnorm FP im_batch im_aug res FP_batch probe gallery FP_probe FP_gallery dist_pg dist_final;
        loss_trn = 0; FP = [];
        for t=1:batchsize:numel(id_train)
            fprintf('trial %d, validation: processing batch %3d of %3d ...\n', trial, fix(t/batchsize)+1, ceil(numel(id_train)/batchsize)) ;
            batch = id_train(t:min(t+batchsize-1, numel(id_train)));
            im_batch = getbatch(imdb, batch);
            im_batch = single(im_batch);
            im_aug = augIm(im_batch, 'center', augopts);

            im_aug = bsxfun(@minus, im_aug, imdb.mean); 
            clear im_batch; im_batch = im_aug; clear im_aug;
            FP_unnorm = [];

            res = res_template;
            im_batch = single(im_batch);
            if opts.useGpu
                im_batch = gpuArray(im_batch);
            end
            res(1).x = im_batch;
            res = jk_cnn_singlePart(net, res, [], 'disableDropout', true, 'mode', 'test');

            FP_unnorm = cat(1, FP_unnorm, squeeze(double(gather(res(end).x))));


            % forward L2 layer
            clear res_L2;
            res_L2 = struct('x', cell(1,2), 'dzdx', cell(1,2), 'aux', cell(1,2));
            res_L2(1).x = FP_unnorm;
            clear FP_unnorm;
            [res_L2(2).x, res_L2(2).aux] = jk_cnn_l2norm(res_L2(1).x);
            FP_batch = res_L2(2).x;

            % to obtain the feature maps for all images
            FP = cat(2, FP, FP_batch);
        end

        % calculate dist
        probe = 1:2:size(FP, 2); probe = probe(:)'; % camb as probe
        gallery = 2:2:size(FP, 2); gallery = gallery(:)'; % cama as gallery
        FP_probe = FP(:, probe); FP_gallery = FP(:, gallery);
        dist_pg = slmetric_pw(FP_probe, FP_gallery, 'sqdist');

        % 镜像对称后的四个距离的总和, l为origion，r为symmetric
        dist_final = dist_pg;

        % for each probe image, calculate the slack
        for i=1:size(dist_final,1)
            if mod(i, 10) == 0
                fprintf('.');
            end
            index_pos = i;
            index_neg = setdiff(1:size(dist_final,1), index_pos);

            Sim_pos = -dist_final(i, index_pos); % 只针对SvsS的情况
            Sim_neg = -dist_final(i, index_neg);

            % calculate slack
            y_gt = ones(numel(index_pos), numel(index_neg));
            [slack, ~] = findMostViolatedMvsM_fast(Sim_pos, Sim_neg, y_gt, lossType);

            % calculate loss
            loss_trn = loss_trn + slack;
        end

        cmc_trn = evaluate_cmc(-dist_final)/numel(id_train);

        %=====================================================================
        info.train.loss(end) = loss_trn;
        info.train.rank1(end) = cmc_trn(1);
        info.val.loss(end) = loss_tst;
        info.val.rank1(end) = cmc_tst(1);

        % plot loss and cmc
        fprintf('\n');
        fprintf('*****trial %d, VIPeR: traing set cmc *****\n', trial);
        fprintf('rank1\trank5\trank10\trank15\trank20\trank30\trank50: \n %2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%% \n', ...
            100*cmc_trn(1), 100*cmc_trn(5), 100*cmc_trn(10), 100*cmc_trn(15), 100*cmc_trn(20), 100*cmc_trn(30), 100*cmc_trn(50));
        fprintf('*****trial %d, VIPeR: testing set cmc *****\n', trial);
        fprintf('rank1\trank5\trank10\trank15\trank20\trank30\trank50: \n %2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%%\t%2.2f%% \n', ...
            100*cmc_tst(1), 100*cmc_tst(5), 100*cmc_tst(10), 100*cmc_tst(15), 100*cmc_tst(20), 100*cmc_tst(30), 100*cmc_tst(50));
        hh=figure(1);clf;
        subplot(1,2,1);
        semilogy(1:epoch, info.train.loss, 'k-'); hold on;
        semilogy(1:epoch, info.val.loss, 'g-');
        xlabel('epoch'); ylabel('loss'); h = legend('train', 'val'); grid on;
        set(h, 'color', 'none');
        title('total loss of train and validation');

        subplot(1,2,2);
        plot(1:epoch, info.train.rank1, 'k-'); hold on;
        plot(1:epoch, info.val.rank1, 'g-');ylim([0 1]);
        xlabel('epoch'); ylabel('error rate'); h = legend('train', 'val'); grid on;
        set(h, 'color', 'none');
        title('error');
        drawnow;
        savefig(hh, fullfile(resDir, ['viper_middle_' num2str(trial) '.fig']));
        if mod(epoch, 20) == 0 && epoch >= 200
            save(fullfile(resDir, ['viper_middle_' num2str(trial) '_' num2str(epoch) '.mat']), 'net', 'cmc_tst', 'info');
        end
        pause(4);
    end
end
        
            
