function [ stop ] = save_callback( p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, varargin)
% The newest version of minfunc updated the callback calling pattern
% Looking at minFunc.m it is called by in line 365
%    stop = outputFcn(x,'init',0,funEvals,f,[],[],g,[],max(abs(g)),varargin{:});
% x is theta
% funEvals is apparently just always 1
% f is "cost" as returned by drdae_obj
% g is "grad" as returned by drdae_obj
% varargin are the parameters passed into minFunc after 'options'
%   which in our case are eI, data_cell, target_cell, 0, 0

% save model while minfunc is running
theta = p1;
state = p2; % appears to be internal state of minFunc, can be 'init', 'iter', or 'done'
iter = p3 % this appears to be the iteration 
eI = varargin{1};

if (strcmp(state, 'iter')) 
    if isfield(eI, 'iterStart')
      iter = iter +eI.iterStart;
    end

    % write theta
    %thetaPath = sprintf('%s/theta_%d.csv',eI.saveDir,iter);
    %dlmwrite(thetaPath,theta);

    % Save as .mat
    saveName = sprintf('%s/%s_%d.mat', eI.saveDir, eI.modelName, iter);
    save(saveName, 'theta', 'eI');
end;

stop = 0;

end

