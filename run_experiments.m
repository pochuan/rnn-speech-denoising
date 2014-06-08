
paths = load_global_paths();

addpath('.');
addpath(genpath(paths.minFuncDir));
addpath(paths.stanfordNNetUtilDir);

mkdir(paths.modelDir);

%%% modify stuff down here 

% 1 layer non-recursive
eI = default_model_settings();
eI.saveDir = paths.modelDir;
eI.modelName = '1layer_normal';
tic; output = train_nn(eI, 2); time=toc
%tic; train_nn(eI, 10); time=toc
%tic; train_nn(eI, 15); time=toc

compute_likelihoods(eI.modelName, output.iterations, paths.testDataDir, 1);

