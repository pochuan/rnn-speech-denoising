function [ priors ]  = load_priors(dir, file_num, feat_dim, label_set_size)
  [~, utt_dat, adata] = load_kaldi_data([dir,'/'],file_num,feat_dim);

  [nframes, labeldim] = size(adata);
  adata = uint32(adata);

  uniq_a = sort(unique(adata,'rows'));
  count_a = histc(adata, uniq_a);
  
  if(uniq_a' == 1:label_set_size)
  else
      uniq_a
      error ('Expected label set 1:%d but got something else.', label_set_size);
  end;
  
  priors = count_a / nframes;
%end;




