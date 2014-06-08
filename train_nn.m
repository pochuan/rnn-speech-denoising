function [ output ]  = train_nn(eI, num_training_utterances)
  % script to specify network arch and train drdae on single machine
  % make copies of this script for each architecture you want to try

  paths = load_global_paths();

  addpath('.');
  addpath(genpath(paths.minFuncDir));
  addpath(paths.stanfordNNetUtilDir);

  %% initialize weights
  [stack_i, W_t_i] = initialize_weights(eI);
  [theta] = rnn_stack2params(stack_i, eI, W_t_i);

  % number of utterances to use.
  % You can set this to -1 and it will use all training utterances.
  % Setting to some low positive number especially helps when testing the code.
  file_num=1;

  [data_cell, targets_cell] = load_nn_data(paths.trainingDataDir, file_num, eI.featDim, num_training_utterances, eI, true);

  %% setup minFunc
  options.Diagnostics = 'on';
  options.Display = 'iter';
  options.MaxIter = 2000;
  options.MaxFunEvals = 2500;
  options.Corr = 50;
  options.DerivativeCheck = 'off';
  options.outputFcn = @save_callback;
  %% run optimizer
  [~,~,~,output] = minFunc(@drdae_obj, theta, options, eI, data_cell, targets_cell, false, false);
%end;
