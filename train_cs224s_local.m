% script to specify network arch and train drdae on single machine
% make copies of this script for each architecture you want to try

%% setup paths for code. assumed this script runs in its own directory
codeDir = '.';
%minFuncDir = '/home/mkayser/school/classes/2013_14_spring/cs224s/project/other-resources/minFunc_2012';
minFuncDir = '/afs/ir.stanford.edu/users/p/o/pochuan/cs224s/project/minFunc_2012';
baseDir = '../scratch/';

%% add paths
addpath(codeDir);
addpath(genpath(minFuncDir));

%% setup network architecture
eI = [];
% dimension of each input frame
eI.featDim = 13;
eI.labelDim = 1;
eI.labelSetSize = 234;
eI.dropout = 0;
% context window size of the input.
eI.winSize = 3;
% weight tying in hidden layers
% if you want tied weights, must have odd number of *hidden* layers
eI.tieWeights = 0;

%% setup weight caching
saveDir = './models';
%eI.saveDir = [saveDir '/TIMIT_full/recur_3hid/first']; % DO NOT END PATH WITH "/"
%eI.saveDir = [saveDir '/TIMIT_full/sixth'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Different Experiments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 hidden layer, non-recurrent
%eI.layerSizes = [512 eI.labelSetSize]; % hidden layers and output layer
%eI.temporalLayer = 0; % highest hidden layer is temporal
%eI.saveDir = [saveDir '/TIMIT_full/nonrecur_1hid/forth']; % DO NOT END PATH WITH "/"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 hidden layer, recurrent:
%eI.layerSizes = [512 eI.labelSetSize];
%eI.temporalLayer = 1;
%eI.saveDir = [saveDir '/TIMIT_full/recur_1hid/forth']; % DO NOT END PATH WITH "/"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 hidden layers, non-recurrent:
%eI.layerSizes = [512 512 eI.labelSetSize];
%eI.temporalLayer = 0;
%eI.saveDir = [saveDir '/TIMIT_full/nonrecur_2hid/forth']; % DO NOT END PATH WITH "/"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 hidden layers, recurrent:
eI.layerSizes = [512 512 eI.labelSetSize];
eI.temporalLayer = 1;
eI.saveDir = [saveDir '/TIMIT_full/recur_2hid/forth']; % DO NOT END PATH WITH "/"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 hidden layer, non-recurrent:
%eI.layerSizes = [512 512 512 eI.labelSetSize];
%eI.temporalLayer = 0;
%eI.saveDir = [saveDir '/TIMIT_full/nonrecur_3hid/first']; % DO NOT END PATH WITH "/"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 hidden layers, recurrent:
%eI.layerSizes = [512 512 512 eI.labelSetSize];
%eI.temporalLayer = 1;
%eI.saveDir = [saveDir '/TIMIT_full/recur_3hid/first']; % DO NOT END PATH WITH "/"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

mkdir(eI.saveDir);

%%%%%%%%%%%%%%%%%%%%%
%% initialize weights
% Fresh start
[stack_i, W_t_i] = initialize_weights(eI);
[theta] = rnn_stack2params(stack_i, eI, W_t_i);

[stack_new, W_t_new] = rnn_params2stack(theta,eI);
[theta_new] = rnn_stack2params(stack_new, eI, W_t_new);

%%%%%%%%%
% Restart training from previous model
%load('./models/TIMIT_full/forth/model_12.mat', 'theta');

%%%%%%%%%%%%%%%%%%%%

%% load data
eI.useCache = 0;

% number of utterances to use, To run on full dataset, set M = -1
M=-1;
dir='./data/output/';
%dir='/home/mkayser/school/classes/2013_14_spring/cs224s/project/rnn-speech-denoising/data/output/';
file_num=1;
feat_dim=13;

[data_cell, targets_cell] = load_nn_data(dir, file_num, feat_dim, M, eI);
%data_cell{1} = rand(eI.inputDim*50,4);
%targets_cell{1} = rand(eI.featDim*50,4);

%% setup minFunc
options.Diagnostics = 'on';
options.Display = 'iter';
options.MaxIter = 2000;
options.MaxFunEvals = 2500;
options.Corr = 50;
%options.DerivativeCheck = 'on';
options.DerivativeCheck = 'off';
options.outputFcn = @save_callback;
%% run optimizer
minFunc(@drdae_obj, theta, options, eI, data_cell, targets_cell, false, false);
