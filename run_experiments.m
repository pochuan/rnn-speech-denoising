
paths = load_global_paths();

addpath('.');
addpath(genpath(paths.minFuncDir));
addpath(paths.stanfordNNetUtilDir);

mkdir(paths.modelDir);

% Generic model
eI = default_model_settings();
eI.saveDir = paths.modelDir;

%%% modify stuff down here 
num_training_files = 10;

% 1 layer non-recursive
eI.modelName = '1layer_normal';
eI.lambda=0;
tic; 
output = train_nn(eI, num_training_files); 
compute_likelihoods(eI.modelName, output.iterations, paths.testDataDir, 10);
toc


