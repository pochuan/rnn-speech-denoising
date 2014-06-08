% IN ORDER TO RUN THIS CODE YOU MUST DO THE FOLLOWING:
%
% 1. Create a directory "nnet" as a subdir of rnn-speech-denoising
% 2. Within nnet, create symlinks:
%    a. minFunc_2012: symlink to your minfunc root dir
%    b. stanford-nnet: symlink to your kaldi-stanford-master/stanford-nnet
%    c. training: symlink to your rnn-speech-denoising/data/output
% 3. Make two more directories: nnet/likelihoods and nnet/models
% 4. Now you've set up all the prereqs. Look inside the function
%    definition for run_experiments, and observe that it 
%    defines some experiment setups, indexed by an ID#.
%    To run, you just call run_experiment(i), with i being
%    the ID that you want to execute.
%    Feel free to modify this locally to suit your purposes.

function [] = run_experiment(id_to_run)
	 
  %%% Take a look inside this function and make sure that all of the
  %%% paths in here are things which exist
  paths = load_global_paths();
  
  addpath('.');
  addpath(genpath(paths.minFuncDir));
  addpath(paths.stanfordNNetUtilDir);
  
  mkdir(paths.modelDir);
  
  % Generic model
  eI = default_model_settings();
  eI.saveDir = paths.modelDir;
  
  %%% modify stuff down here if necessary
  num_training_files = -1;
  num_testing_files  = -1;

  switch id_to_run
    case 1
      % 1 layer non-recurrent
      eI.modelName = 'nonrecur_1hid_lambda.0005';
      eI.layerSizes = [512 eI.labelSetSize];
      eI.lambda = 0.0005;
    case 2
      % 2 layer non-recurrent
      eI.modelName = 'nonrecur_2hid_lambda.0005';
      eI.layerSizes = [512 512 eI.labelSetSize];
      eI.lambda = 0.0005;
    case 3
      % 1 layer recurrent
      eI.modelName = 'recur1_1hid_lambda.0005';
      eI.layerSizes = [512 eI.labelSetSize];
      eI.temporalLayer = 1;
      eI.lambda = 0.0005;
    case 4
      % 2 layer recurrent
      eI.modelName = 'recur2_2hid_lambda.0005';
      eI.layerSizes = [512 512 eI.labelSetSize];
      eI.temporalLayer = 2;
      eI.lambda = 0.0005;
  end;

  output = train_nn(eI, num_training_files); 
  compute_likelihoods(eI.modelName, output.iterations, paths.testDataDir, num_testing_files);
%end;
  
