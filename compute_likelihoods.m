function [ ]  = compute_likelihoods(model_name, model_iter_num, test_input_dir, num_test_utterances)

  [~, test_set_name, ~] = fileparts(test_input_dir);
  assert(length(test_set_name)>0);

  paths = load_global_paths();
  
  addpath('.');
  addpath(genpath(paths.minFuncDir));
  addpath(paths.stanfordNNetUtilDir);

  
  model_file       = sprintf('%s/%s_%d.mat'   , paths.modelDir, model_name, model_iter_num);
  likelihoods_file = sprintf('%s/%s.%s_%d.ark', paths.likelihoodsDir, test_set_name, model_name, model_iter_num);
  
  load(model_file, 'theta','eI');
  eI.labelSetSize = 117;

  % A major assumption of this code is that there is only ever one file in the given directory.
  file_num=1;
  
  [data_cell, targets_cell, utt_dat] = load_nn_data(test_input_dir, file_num, eI.featDim, num_test_utterances, eI, false);
  priors = load_priors(paths.trainingDataDir, file_num, eI.featDim, eI.labelSetSize);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Write likelihoods from pred_cell to Kaldi file format
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  fid = fopen(likelihoods_file, 'w');
  
  numUtts = size(data_cell, 2);
  
  chunkSize = 100; %Size of utterance chunks to forward prop at a time
  numChunks = ceil(numUtts/chunkSize);
  numUttsDone = 0; %Number of utterances written
  
  for i=1:numChunks
    %Get subset of keys and subset of sizes to forward prop and write
      
    if i==numChunks
      data_subarray={data_cell{(i-1)*chunkSize+1:numUtts}};
      if(~isempty(targets_cell))
	targets_subarray={targets_cell{(i-1)*chunkSize+1:numUtts}};
      end;
      utt_keys_subarray = {utt_dat.keys{(i-1)*chunkSize+1:numUtts}};
    else  
      data_subarray={data_cell{(i-1)*chunkSize+1:i*chunkSize}};
      if(~isempty(targets_cell))
	targets_subarray={targets_cell{(i-1)*chunkSize+1:i*chunkSize}};
      end;
      utt_keys_subarray = {utt_dat.keys{(i-1)*chunkSize+1:i*chunkSize}};
    end;
    
    % drdae prototype
    %[cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, data_cell, targets_cell, fprop_only, pred_out)
    % if pred_out is true, pred_cell is a cell array whose cells correspond 1-to-1 to utterances from data_cell.
    % Each cell contains a matrix with one column per frame. The values in a column are the label posteriors for
    % that frame.
    [cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, data_subarray, targets_subarray, true,true);
    
    % Add priors
    for i=1:numel(pred_cell)
      pred_cell{i} = bsxfun(@plus, log(pred_cell{i}), -log(priors));
    end;
    
    %Write each utterance separately so we can write as key value pairs
    for u=1:numel(pred_cell)
	
      % Notice we take the transpose here so that instead of one column per frame it now is one row per frame
      outputs = pred_cell{u}';
      utt_num_frames = size(pred_cell{u},1);
      
      FLOATSIZE=4;
      %write each key with corresponding nnet value
      fprintf(fid,'%s ', utt_keys_subarray{u}); % write key
      fprintf(fid,'%cBFM ',char(0)); % write Kaldi header
      fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
      %byte int
      fwrite(fid,utt_num_frames,'int'); % write number rows
      fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
      %byte int
      fwrite(fid, eI.labelSetSize ,'int');  % write number cols
      
      % MRK: don't understand the comment below about transposing. Ignoring...
      % write full utterance (have to transpose as fwrite is column order
      fwrite(fid, outputs, 'float');
    end
    numUttsDone = numUttsDone+length(utt_keys_subarray);
  end
  fclose(fid);
%end;
