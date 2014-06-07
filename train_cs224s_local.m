% script to specify network arch and train drdae on single machine
% make copies of this script for each architecture you want to try

paths = load_global_paths();

addpath('.');
addpath(genpath(paths.minFuncDir));
addpath(paths.stanfordNNetUtilDir);

eI = default_model_settings();

% Where to save models to
eI.saveDir = paths.modelDir;
mkdir(eI.saveDir);

%% initialize weights
[stack_i, W_t_i] = initialize_weights(eI);
[theta] = rnn_stack2params(stack_i, eI, W_t_i);

% MRK: These two lines are very weird -- we just load the model again into variables we don't ever appear to use. 
% I'm testing whether we can just comment them out.
%[stack_new, W_t_new] = rnn_params2stack(theta,eI);
%[theta_new] = rnn_stack2params(stack_new, eI, W_t_new);


% number of utterances to use.
% You can set this to -1 and it will use all training utterances.
% Setting to some low positive number especially helps when testing the code.
num_training_utterances=3;
file_num=1;

[data_cell, targets_cell] = load_nn_data(paths.trainingDataDir, file_num, eI.featDim, num_training_utterances, eI, true);

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
