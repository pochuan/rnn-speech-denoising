function [ paths ]  = load_global_paths()
	 
  % Change this if necessary
  % Better yet, just create a nnet subdir of your rnn-speech-denoising
  % directory, and populate with symlinks to the relevant locations below
  % modelDir and likelihoodsDir should probably just be created
  paths = [];
  paths.rootDir='./nnet/';

  % Code locations -- note these can/should be symlinks
  paths.minFuncDir          = sprintf('%s/minFunc_2012', paths.rootDir);
  paths.stanfordNNetUtilDir = sprintf('%s/stanford-nnet/util', paths.rootDir);
  
  %annoyingly the trailing slash is currently necessary
  paths.trainingDataDir     = sprintf('%s/training', paths.rootDir); 
  paths.testDataDir         = sprintf('%s/test', paths.rootDir); 
  paths.likelihoodsDir      = sprintf('%s/likelihoods/', paths.rootDir);
  paths.modelDir            = sprintf('%s/models', paths.rootDir);
end;
