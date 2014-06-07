% script to specify network arch and train drdae on single machine
% make copies of this script for each architecture you want to try

%% setup paths for code. assumed this script runs in its own directory
codeDir = '.';
minFuncDir = '/home/mkayser/school/classes/2013_14_spring/cs224s/project/other-resources/minFunc_2012';
baseDir = '../scratch/';
%%%%%%%%%%%%%%%%%%%%%
% Log likelihood output file directory
outputDir = '../likelihoods/';
%%%%%%%%%%%%%%%%%%%%%%

%% AMAAS setup
%codeDir = '/afs/cs.stanford.edu/u/amaas/scratch/audio/audio_repo/matlab_wd/drdae';
%% add paths
addpath(codeDir);
addpath(genpath(minFuncDir));

%% setup network architecture
eI = [];
% dimension of each input frame
eI.featDim = 13;
eI.outputDim = 1;
eI.labelSetSize = 234;
eI.dropout = 0;
% context window size of the input.
eI.winSize = 3;
% weight tying in hidden layers
% if you want tied weights, must have odd number of *hidden* layers
eI.tieWeights = 0;
% hidden layers and output layer
eI.layerSizes = [512 eI.labelSetSize];
% highest hidden layer is temporal
eI.temporalLayer = 0;
% dim of network input at each timestep (final size after window & whiten)
eI.inputDim = eI.featDim * eI.winSize;
% length of input sequence chunks.
% eI.seqLen = [1 10 25 50 100];
eI.seqLen = [1 50 100];
% activation function
eI.activationFn = 'tanh';
% temporal initialization type
eI.temporalInit = 'rand';
% weight norm penaly
eI.lambda = 0;
% file containing whitening matrices for outputs
eI.targetWhiten = [codeDir '/aurora_whiten.mat'];
%% setup weight caching
saveDir = '.';
eI.saveDir = [saveDir '/test'];
mkdir(eI.saveDir);
%% initialize weights
[stack_i, W_t_i] = initialize_weights(eI);
[theta] = rnn_stack2params(stack_i, eI, W_t_i);

[stack_new, W_t_new] = rnn_params2stack(theta,eI);
[theta_new] = rnn_stack2params(stack_new, eI, W_t_new);
%% Directory of features
eI.featInBase =baseDir; % '/afs/cs.stanford.edu/u/amaas/scratch/aurora2/features/';

%% load data
eI.useCache = 0;

% number of utterances to use
M=3;
dir='/home/mkayser/school/classes/2013_14_spring/cs224s/project/rnn-speech-denoising/data/output/';
file_num=1;
feat_dim=13;

[data_cell, targets_cell, utt_dat] = load_nn_data(dir, file_num, feat_dim, M, eI);

% drdae prototype
%[cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, data_cell, targets_cell, fprop_only, pred_out)
% if pred_out is true, pred_cell is a cell array of the number of time series containing
% matrix of posteriors for each frame in that time series
[cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, data_cell, targets_cell, true,true)

% Add priors, I dont know where we get our priors...
% loglikelihood = bsxfun(@plus, log(pred_cell), priors);

%%%%%%%%%%%%%%
% Write likelihoods from pred_cell to Kaldi file format
%%%%%%%%%%%%%%
fid = fopen(outputDir, 'w');

numUtts = length(utt_dat.keys);

chunkSize = 100; %Size of utterance chunks to forward prop at a time
numChunks = ceil(numUtts/chunkSize);
numFramesDone = 0; %Number of total frames written
numUttsDone = 0; %Number of utterances written

for i=1:numChunks
    %Get subset of keys and subset of sizes to forward prop and write
    if i==numChunks
        subKeys=utt_dat.keys((i-1)*chunkSize+1:end);
        subSizes=utt_dat.sizes((i-1)*chunkSize+1:end,:);
    else
        subKeys=utt_dat.keys((i-1)*chunkSize+1:i*chunkSize);
        subSizes=utt_dat.sizes((i-1)*chunkSize+1:i*chunkSize);
    end

    %input = feats(numFramesDone+1:numFramesDone+sum(subSizes),:);
    %Forward prop data in cost function
    %[c, g, nC, nE, ceC, wC, output] = spNetCostSlave(theta,eI,input',[],1);

    %take log of forward propped dat and add log inverse priors
    %output = bsxfun(@plus,log(output'),priors);

    %Write each utterance separately so we can write as key value pairs
    numFramesWrit = 0;
    for u=1:length(subKeys)
        uttSize = subSizes(u);
        FLOATSIZE=4;
        %write each key with corresponding nnet value
        fprintf(fid,'%s ',subKeys{u}); % write key
        fprintf(fid,'%cBFM ',char(0)); % write Kaldi header
        fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
        %byte int
        fwrite(fid,uttSize,'int'); % write number rows
        fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
        %byte int
        fwrite(fid,numStates,'int');  % write number cols

        % write full utterance (have to transpose as fwrite is column order
        fwrite(fid,output(numFramesWrit+1:numFramesWrit+uttSize,:)', 'float');
        numFramesWrit = numFramesWrit+uttSize;
    end
    numUttsDone = numUttsDone+length(subKeys);
    numFramesDone = numFramesDone+sum(subSizes);
end
fclose(fid);

