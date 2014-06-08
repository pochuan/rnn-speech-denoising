function [ data_cell, target_cell, utt_dat]  = load_nn_data(dir, file_num, feat_dim, num_utterances, eI, train_mode)
  % num_utterances: Number of sample files to use. value < 0 loads all
  % eI.winSize: Size of window
  % eI.seqLen: unique lengths (in ascending order) 
  %             files are chopped by these lengths. (ex: [1, 10, 100])
  % data_cell: noisy data. cell array of different training lengths. 
  % target_cell: clean data. cell array of different training lengths.
  % train_mode: true for training data, false for test.
  %             If true, ignore seqLen and perform no chopping and
  %             no aggregating of inputs.
	 
  %addpath('/home/mkayser/school/classes/2013_14_spring/cs224s/project/other-resources/kaldi-stanford-master/stanford-nnet/util');
%addpath('../kaldi-stanford-master/stanford-nnet/util');


	 
  
  [fdata, utt_dat, adata] = load_kaldi_data([dir,'/'],file_num,feat_dim);

  [nframes , nfeats] = size(fdata);
  [nframes2, labeldim] = size(adata);

  adata = uint32(adata);

  assert(~train_mode || nframes == nframes2);
  assert(~train_mode || labeldim == eI.labelDim);
  assert(nfeats == eI.featDim);


  % features loaded here are #frames x 13
  % alignments are #frames x 1
  
  %% Loop through every utterance
  if num_utterances < 0
    num_utterances = size(utt_dat.sizes,1);
  end

  % winSize must be odd for padding to work
  if mod(eI.winSize,2) ~= 1
    fprintf(1,'error! winSize must be odd!');
    return
  end;  


  if train_mode 
    %% Training. Allocate data and targets matrices
    %% Count number of series of various lengths for pre-allocation
    seqLenSizes = zeros(1,length(eI.seqLen));
    start_index=1;
    for i=1:num_utterances
      T = utt_dat.sizes(i);
      %T
      remainder = T;  
      for i=length(eI.seqLen):-1:1
	num = floor(single(remainder)/eI.seqLen(i));  % Nasty octave bug here if no typecast
	remainder = mod(remainder,eI.seqLen(i));
	seqLenSizes(i) = seqLenSizes(i)+num;
      end
    end
    
    % Allocate data and target cells and their matrices
    data_cell = cell(1,length(eI.seqLen));
    target_cell = cell(1,length(eI.seqLen));
    for i=length(eI.seqLen):-1:1
      data_cell{i} = zeros(eI.inputDim*eI.seqLen(i),seqLenSizes(i));
      target_cell{i} = zeros(eI.labelDim*eI.seqLen(i),seqLenSizes(i));
    end
  else
      % Test mode.
      data_cell = cell(1, 0);
      target_cell = cell(1, 0);
  end;

  % I guess this is a way to remember the current number of samples of
  % each length of time-series
  seqLenPositions = ones(1,length(eI.seqLen));

  % Package utterance into data_cell
  start_index=1;
  for i=1:num_utterances,
    T = utt_dat.sizes(i);
    % Note we take the transpose here
    utt_fdata = fdata(start_index:start_index+T-1,:)';
    if ~isempty(adata)
      utt_adata = adata(start_index:start_index+T-1,:)';
    end;
    start_index = start_index+T;

    % now utt_fdata is 13 x #frames
    %     utt_adata is 1 x #frames

    %% pad with repeated frames
    if eI.winSize > 1
      padlen = (eI.winSize-1)/2;
      utt_fdata = [repmat(utt_fdata(:,1),1,padlen), utt_fdata, repmat(utt_fdata(:,end),1,padlen)];
    end;

    %% im2col puts winSize frames in each column
    utt_fwindows = im2col(utt_fdata,[nfeats, eI.winSize],'sliding');

    assert(size(utt_fwindows,2) == T); %, '%d windows which does not equal %d frames', utt_fwindows, nframes);

    if train_mode
      %% put it in the correct cell area.
      while T > 0
	% assumes length in ascending order.
	% Finds longest length shorter than utterance
	c = find(eI.seqLen <= T, 1,'last');
	
	binLen = eI.seqLen(c);
	%binLen
	assert(~isempty(c),'could not find length bin for %d',T);
	% copy data for this chunk
	data_cell{c}(:,seqLenPositions(c))=reshape(utt_fwindows(:,1:binLen),[],1);
	target_cell{c}(:,seqLenPositions(c))=reshape(utt_adata(:,1:binLen),[],1);
	seqLenPositions(c) = seqLenPositions(c)+1;
	% trim for next iteration
	T = T-binLen;
	if T > 0
	  utt_fwindows = utt_fwindows(:,(binLen+1):end);
	  utt_adata = utt_adata(:,(binLen+1):end);
	end;
      end;
    else
	% test mode. no chopping, just put entire utterance into new cell of data_cell.
	% Also, there may be no alignments at all
	c = numel(data_cell)+1;
	data_cell{c} = reshape(utt_fwindows, [], 1);
	if ~isempty(utt_adata) 
	  target_cell{c} = reshape(utt_adata, [], 1);
	end;
    end;
  end;

  if(train_mode) 
    assert(isequal(seqLenPositions,seqLenSizes+1));
  end;


%end;




