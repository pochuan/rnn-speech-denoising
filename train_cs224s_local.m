% script to specify network arch and train drdae on single machine
% make copies of this script for each architecture you want to try

%% setup paths for code. assumed this script runs in its own directory
codeDir = '.';
minFuncDir = '../minFunc_2012/';
baseDir = '../scratch/';
%% AMAAS setup
%codeDir = '/afs/cs.stanford.edu/u/amaas/scratch/audio/audio_repo/matlab_wd/drdae';
%% add paths
addpath(codeDir);
addpath(genpath(minFuncDir));

%% setup network architecture
eI = [];
% dimension of each input frame
eI.featDim = 14;
eI.dropout = 0;
% context window size of the input.
eI.winSize = 3;
% weight tying in hidden layers
% if you want tied weights, must have odd number of *hidden* layers
eI.tieWeights = 0;
% 2 hidden layers and output layer
eI.layerSizes = [512 eI.featDim];
% highest hidden layer is temporal
eI.temporalLayer = 0;
% dim of network input at each timestep (final size after window & whiten)
eI.inputDim = eI.featDim * eI.winSize;
% length of input sequence chunks.
% eI.seqLen = [1 10 25 50 100];
eI.seqLen = [50];
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

%Get SNR List
snrList = {'clean1', 'N1_SNR5', 'N1_SNR10', 'N1_SNR15', 'N1_SNR20', ...
'clean2', 'N2_SNR5', 'N2_SNR10', 'N2_SNR15', 'N2_SNR20', ...
'clean3', 'N3_SNR5', 'N3_SNR10', 'N3_SNR15', 'N3_SNR20', };
eI.subdirs = snrList;
% [data_cell, targets_cell] = load_aurora( baseDir, 'Mfc08_multiTR', snrList, -1, eI );
data_cell = {};
data_cell{1} = rand(eI.inputDim*50,4);
targets_cell{1} = rand(eI.featDim*50,4);

% dieplay mean as a whitening debug check
%disp(mean(data_cell{1},2));
%% setup minFunc
options.Diagnostics = 'on';
options.Display = 'iter';
options.MaxIter = 2000;
options.MaxFunEvals = 2500;
options.Corr = 50;
options.DerivativeCheck = 'off';
% options.outputFcn = @save_callback;
%% run optimizer
minFunc(@drdae_obj, theta, options, eI, data_cell, targets_cell, false, false);
