
eI.outputDim = 1;
eI.featDim = 13;
eI.winSize = 3;
eI.inputDim = eI.featDim * eI.winSize;
eI.seqLen = [1 50 100];
M=3;
%dir='/home/mkayser/school/classes/2013_14_spring/cs224s/project/rnn-speech-denoising/data/output/';
file_num=1;
feat_dim=13;

load_nn_data(dir, file_num, feat_dim, M, eI);
