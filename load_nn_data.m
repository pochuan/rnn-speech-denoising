function [ data_cell, target_cell ]  = load_nn_data(dir, file_num, feat_dim, M, eI)
  % M: Number of sample files to use. value < 0 loads all
  % eI.winSize: Size of window
  % eI.seqLen: unique lengths (in ascending order) 
  %             files are chopped by these lengths. (ex: [1, 10, 100])
  % eI.useCache: If true, read + save to aurora_data.mat
  %              !!!WARNING!!!: If you change this function, delete
  %              aurora_data.mat.
  % data_cell: noisy data. cell array of different training lengths. 
  % target_cell: clean data. cell array of different training lengths.
	 
  addpath('/home/mkayser/school/classes/2013_14_spring/cs224s/project/other-resources/kaldi-stanford-master/stanford-nnet/util');

	 
	 
  [f, utt_dat, a] = load_kaldi_data(dir,file_num,feat_dim);

  [nframes , nfeats] = size(f);
  [nframes2, nlabels] = size(a);

  %f = repmat((1:nframes)', 1, nfeats);
  %[nframes , nfeats] = size(f);
  %disp(f(1:10,:));

  assert(nframes == nframes2);
  assert(nlabels == eI.outputDim);
  assert(nfeats == eI.featDim);

  %disp(size(f))
  %disp(size(utt_dat.keys))
  %disp(size(utt_dat.sizes))
  %disp(size(a))

  % features loaded here are #frames x 13
  % alignments are #frames x 1
  
  %% Loop through every utterance
  if M < 0
    M = size(utt_dat.sizes,1);
  end
  
  %% Count number of series of various lengths for pre-allocation
  seqLenSizes = zeros(1,length(eI.seqLen));
  start_index=1;
  for i=1:M
    T = utt_dat.sizes(i);
    remainder = T;  
    for i=length(eI.seqLen):-1:1
      num = floor(remainder/eI.seqLen(i));
      remainder = mod(remainder,eI.seqLen(i));
      seqLenSizes(i) = seqLenSizes(i)+num;
    end
  end

  % Allocate data and target cells and their matrices
  data_cell = cell(1,length(eI.seqLen));
  target_cell = cell(1,length(eI.seqLen));
  for i=length(eI.seqLen):-1:1
    data_cell{i} = zeros(eI.inputDim*eI.seqLen(i),seqLenSizes(i));
    target_cell{i} = zeros(eI.outputDim*eI.seqLen(i),seqLenSizes(i));
  end


  % winSize must be odd for padding to work
  if mod(eI.winSize,2) ~= 1
    fprintf(1,'error! winSize must be odd!');
    return
  end;  

  % I guess this is a way to remember the current number of samples of
  % each length of time-series
  seqLenPositions = ones(1,length(eI.seqLen));

  % Perform chopping on each utterance
  start_index=1;
  for i=1:M,
    T = utt_dat.sizes(i);
    % Note we take the transpose here
    fdata = f(start_index:start_index+T-1,:)';
    adata = a(start_index:start_index+T-1,:)';
    start_index = start_index+T;

    % now fdata is 13 x #frames
    %     adata is 1 x #frames

    %% pad with repeated frames
    if eI.winSize > 1
      padlen = (eI.winSize-1)/2;
      fdata = [repmat(fdata(:,1),1,padlen), fdata, repmat(fdata(:,end),1,padlen)];
    end;

    %disp(fdata(:,1:10));

    %% im2col puts winSize frames in each column
    fwindows = im2col(fdata,[nfeats, eI.winSize],'sliding');

    %disp([nfeats, eI.winSize]);
    %disp(size(fwindows));
    %disp(fwindows(:,1:10));

    assert(size(fwindows,2) == T); %, '%d windows which does not equal %d frames', fwindows, nframes);

    %% put it in the correct cell area.
    while T > 0
      % assumes length in ascending order.
      % Finds longest length shorter than utterance
      c = find(eI.seqLen <= T, 1,'last');
      
      binLen = eI.seqLen(c);
      assert(~isempty(c),'could not find length bin for %d',T);
      % copy data for this chunk
      data_cell{c}(:,seqLenPositions(c))=reshape(fwindows(:,1:binLen),[],1);
      target_cell{c}(:,seqLenPositions(c))=reshape(adata(:,1:binLen),[],1);
      seqLenPositions(c) = seqLenPositions(c)+1;
      % trim for next iteration
      T = T-binLen;
      if T > 0
	fwindows = fwindows(:,(binLen+1):end);
	adata = adata(:,(binLen+1):end);
      end;
    end;
  end;

  for j=1:length(data_cell)
      disp(size(data_cell{j}));
  end
  %disp(data_cell{3}(:,1));
  %a=data_cell{1};
  %b=data_cell{2};
  %c=data_cell{3};
  %save(sprintf('data/data.%d',eI.seqLen(1)), 'a', '-ascii');
  %save(sprintf('data/data.%d',eI.seqLen(2)), 'b', '-ascii');
  %save(sprintf('data/data.%d',eI.seqLen(3)), 'c', '-ascii');
end;




