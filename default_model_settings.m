function [ eI ]  = default_model_settings()
  % Settings for non-recurrent, shallow (1 hidden layer) neural network

  eI = [];
  % dimension of each input frame
  eI.featDim = 13;
  eI.labelDim = 1;
  eI.labelSetSize = 117;
  eI.dropout = 0;
  % context window size of the input.
  eI.winSize = 3;
  % hidden layers and output layer
  eI.layerSizes = [512 eI.labelSetSize];
  % highest hidden layer is temporal
  eI.temporalLayer = 0;
  % dim of network input at each timestep (final size after window & whiten)
  eI.inputDim = eI.featDim * eI.winSize;
  % length of input sequence chunks.
  eI.seqLen = [1 50 100];
  % activation function
  eI.activationFn = 'tanh';
  % temporal initialization type
  eI.temporalInit = 'rand';
  % weight norm penaly
  eI.lambda = 0;
end;
