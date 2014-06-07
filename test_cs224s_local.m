% script to specify network arch and train drdae on single machine
% make copies of this script for each architecture you want to try


paths = load_global_paths();

addpath('.');
addpath(genpath(paths.minFuncDir));
addpath(paths.stanfordNNetUtilDir);

eI = default_model_settings();

%% initialize weights
[stack_i, W_t_i] = initialize_weights(eI);
[theta] = rnn_stack2params(stack_i, eI, W_t_i);

%% load data
eI.useCache = 0;

% number of utterances to use
num_test_utterances=-1;
file_num=1;

[data_cell, targets_cell, utt_dat] = load_nn_data(paths.testDataDir, file_num, eI.featDim, num_test_utterances, eI, false);


%%%%%%%%%%%%%%
% Write likelihoods from pred_cell to Kaldi file format
%%%%%%%%%%%%%%
fid = fopen(paths.outputDir, 'w');

numUtts = length(utt_dat.keys);

chunkSize = 100; %Size of utterance chunks to forward prop at a time
numChunks = ceil(numUtts/chunkSize);
numUttsDone = 0; %Number of utterances written

for i=1:numChunks
  %Get subset of keys and subset of sizes to forward prop and write
  if i==numChunks
    data_subarray=data_cell{(i-1)*chunkSize+1:end};
    if(~isempty(targets_cell))
      targets_subarray=targets_cell{(i-1)*chunkSize+1:end};
    end;
    utt_keys_subarray = utt_dat.keys((i-1)*chunkSize+1:end);
  else  
    data_subarray=data_cell{(i-1)*chunkSize+1:i*chunkSize}; 
    if(~isempty(targets_cell))
      targets_subarray=targets_cell{(i-1)*chunkSize+1:i*chunkSize};
    end;
    utt_keys_subarray = utt_dat.keys((i-1)*chunkSize+1:i*chunkSize);
 end
  
  % drdae prototype
  %[cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, data_cell, targets_cell, fprop_only, pred_out)
  % if pred_out is true, pred_cell is a cell array of the number of time series containing
  % matrix of posteriors for each frame in that time series
  [cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, data_subarray, targets_subarray, true,true)
  
  % Add priors, I dont know where we get our priors...
  for i=1:size(pred_cell)
  %% TODO priors ????
  % loglikelihood = bsxfun(@plus, log(pred_cell), priors);    
  %take log of forward propped dat and add log inverse priors
  %output = bsxfun(@plus,log(output'),priors);
  end;

  %Write each utterance separately so we can write as key value pairs
  for u=1:size(pred_cell)
    %uttSize = subSizes(u);
    uttSize = size(pred_cell{u},1); %% TODO: make sure the dimensions are right here
    outputs = pred_cell{u}
    FLOATSIZE=4;
    %write each key with corresponding nnet value
    fprintf(fid,'%s ', utt_keys_subarray[u]); % write key
    fprintf(fid,'%cBFM ',char(0)); % write Kaldi header
    fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
    %byte int
    fwrite(fid,uttSize,'int'); % write number rows
    fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
    %byte int
    fwrite(fid,numStates,'int');  % write number cols
    
    % write full utterance (have to transpose as fwrite is column order
    fwrite(fid, outputs, 'float');
  end
  numUttsDone = numUttsDone+length(subKeys);
end
fclose(fid);

