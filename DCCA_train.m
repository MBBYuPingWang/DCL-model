function [F1opt,F2opt,CORR_train,CORR_tune,CORR_test]=DCCA_train( ...
  X1,X2,Y,XV1,XV2,YV,XTe1,XTe2,K,hiddentype,NN1,NN2,rcov1,rcov2,l2penalty, ...
  batchsize,eta0,decay,momentum,maxepoch,randseed)
% [F1opt,F2opt,CORR_train,CORR_tune,CORR_test]=DCCA_train( ...
%   X1,X2,XV1,XV2,XTe1,XTe2,K,hiddentype,NN1,NN2,rcov1,rcov2,l2penalty, ...
%   batchsize,eta0,decay,momentum,maxepoch,randseed) trains the DCCA model
%   using stochastic gradient descent.
%
% Inputs
%   X1: training data for view 1, each row contains a data sample.
%   X2: training data for view 2, each row contains a data sample.
%   XV1: tuning data for view 1.
%   XV2: tuning data for view 2.
%   XTe1: optional testing data for view 1, can be empty.
%   XTe2: optional testing data for view 2, can be empty.
%   K: dimension of DCCA projection.
%   hiddentype: type of hidden units for the networks. Can be 'linear',
%     'sigmoid','tanh','relu','cubic'.
%   NN1: vector of hidden layer sizes for view 1 network.
%   NN2: vector of hidden layer sizes for view 2 network.
%   rcov1: regularization parameter for view 1.
%   rcov2: regularization parameter for view 2.
%   l2penalty: weight decay parameters for all weight parameters.
%   batchsize: number of samples in a minibatch for estimating gradient.
%   eta0: initial learning rate.
%   decay: the geometric rate in which the learning rate decays.
%   momentum: momentum parameter for SGD, with value in [0 1).
%   maxepoch: number of passes over the training set.
%   randseed: random seed for initializing weights and shuffling training
%     set at each epoch.
%
% Outputs
%   F1opt: trained network for view 1, it is a cell array containing each
%     all layers of the network. Each layer has a field 'type' indicating
%     the type of hidden activation, a field 'units' indicating the output
%     dimension of the layer, a filed 'l' indicating the weight decay
%     parameter, and a field 'W' containing the weight matrix.
%   F2opt: trained network for view 2, same structure as F1opt.
%   CORR_train: training correlation at each epoch.
%   CORR_tune: tuning correlation at each epoch.
%   CORR_test: testing correlation at each epoch if test data is given.


%% change hyper parameters for SGD

% l2penalty
% eta0
% decay
% batchsize = 
% 
% maxepoch; iteration step maximum
% momentum

if isempty(XTe1) || isempty(XTe2)
  CORR_test=[];
end
if ~exist('randseed','var') || isempty(randseed)
  randseed=0;
end
rng(randseed);

%% Filename to save intermediate result.
filename=['result_K=' num2str(K) ...
  '_rcov1=' num2str(rcov1) '_rcov2=' num2str(rcov2) ...
  '_l2penalty=' num2str(l2penalty) ...
  '_batchsize=' num2str(batchsize) ...
  '_eta0=' num2str(eta0) '_decay=' num2str(decay) ...
  '_momentum=' num2str(momentum) ...
  '_maxepoch=' num2str(maxepoch) ...
  '.mat'];

if exist(filename,'file')
  
  load(filename,'randseed','F1opt','F2opt','F1','F2','TIME',...
    'eta','delta','optvalid','CORR_train','CORR_tune','CORR_test');
  its=length(CORR_train)-1;
  if its>=maxepoch;
    fprintf('Neural networks have already been trained!\nExiting ...\n');
    return;
  else
    fprintf('Neural networks trained halfway!\nLoading ...\n');
    [N,D1]=size(X1);  [~,D2]=size(X2);
  end
else
  
  % fprintf('Result will be saved in %s\n',filename);
  [N,D1]=size(X1); [~,D2]=size(X2);
  batchsize = N;
  %% Set view1 architecture.
  Layersizes1=[D1 NN1];  Layertypes1={};
  for nn1=1:length(NN1)-1;
    Layertypes1=[Layertypes1, {hiddentype}];
  end
  % I choose to set the last layer to be linear.
  Layertypes1{end+1}='linear';
  %% Set view2 architecture.
  Layersizes2=[D2 NN2];  Layertypes2={};
  for nn2=1:length(NN2)-1;
    Layertypes2=[Layertypes2, {hiddentype}];
  end
  Layertypes2{end+1}='linear';
  %% Random initialization of weights.
  F1=deepnetinit(Layersizes1,Layertypes1);      % WX: [D1 1024 1024 ... 1024 K]
  F2=deepnetinit(Layersizes2,Layertypes2);
  
  % % %   %% Pretrained network.
  % % %   load ./F1G1_RBMPRETRAIN.mat F1;
  
  %% L2 penalty on weights is used for DCCA training.
  for j=1:length(F1)  F1{j}.l=l2penalty;  end   % WX: layerwise L2-norm penalty (not networkwise)
  for j=1:length(F2)  F2{j}.l=l2penalty;  end
  
  %% Compute canonical correlations at the outputs. WX 04/10/2018 Fix duplicated CCA procedure
%   FX1=deepnetfwd(X1,F1); FX2=deepnetfwd(X2,F2);
%   [A,B,m1,m2,D]=linCCA(FX1,FX2,K,rcov1,rcov2);  clear FX1 FX2;              %% question: duplicate CCAed; 
% %   SIGN=sign(A(1,:)+eps); A=bsxfun(@times,A,SIGN); B=bsxfun(@times,B,SIGN);   %%  ????????????   %% MUST be a mistake
%   f1.type='linear'; f1.units=K; f1.W=[A;-m1*A]; F1tmp=F1; F1tmp{end+1}=f1;
%   f2.type='linear'; f2.units=K; f2.W=[B;-m2*B]; F2tmp=F2; F2tmp{end+1}=f2;
%   clear A B m1 m2 D;
  
  %% WX 04/10/2018 Fix duplicated CCA procedure
  F1tmp=F1; F2tmp=F2;
  
  X_train=deepnetfwd(X1,F1tmp); % don't need all layers, waste time         %% question: duplicate CCAed; because X_train is already the 
  CORR_train=DCCA_corr(X_train,deepnetfwd(X2,F2tmp),Y,K); clear X_train;      %% output of [deep network + CCA]; DCCA_corr(X_train) will
  X_tune=deepnetfwd(XV1,F1tmp);                                             %% double CCA the output
  CORR_tune=DCCA_corr(X_tune,deepnetfwd(XV2,F2tmp),YV,K); clear X_tune;
  if ~isempty(XTe1) && ~isempty(XTe2)
    X_test=deepnetfwd(XTe1,F1tmp);
    CORR_test=DCCA_corr(X_test,deepnetfwd(XTe2,F2tmp),K); clear X_test;
  end
                                                                            %% bookmark
  its=0; TIME=0; delta=0; eta=eta0;
  optvalid=CORR_tune; F1opt=F1tmp; F2opt=F2tmp;
  fprintf('Initialization: tuning correlation %f!\n',optvalid);
%   save(filename,'randseed','F1opt','F2opt','F1','F2','TIME',...
%     'eta','delta','optvalid','CORR_train','CORR_tune','CORR_test');
end

%% Concatenate the weights in a long vector. %% WX: VV is matrix rather than a vector
VV=[];
Nlayers=length(F1); net1=cell(1,Nlayers);
for k=1:Nlayers
  VV=[VV; F1{k}.W(:)];  net1{k}=rmfield(F1{k},'W');
end
Nlayers=length(F2); net2=cell(1,Nlayers);
for k=1:Nlayers
  VV=[VV; F2{k}.W(:)];  net2{k}=rmfield(F2{k},'W');
end
fprintf('Number of weight parameters: %d\n',length(VV));

%% Use GPU if equipped. GPU significantly speeds up optimization.
% if gpuDeviceCount>0
%   fprintf('GPU detected. Trying to use it ...\n');
%   try
%     VV=gpuArray(VV);
%     X1=gpuArray(X1);
%     X2=gpuArray(X2);
%     fprintf('Using GPU ...\n');
%   catch
%   end
% end
% batchsize = N; %% WX
%% Start stochastic gradient descent.                      
numbatches=ceil(N/batchsize);
while its<maxepoch
  eta=eta0*decay^its; % Reduce learning rate.
  t0=tic;
  rp=randperm(N);   % Shuffle the data set.
  for i=1:numbatches
    idx1=(i-1)*batchsize+1;
    idx2=min(i*batchsize,N);
    idx=[rp(idx1:idx2),rp(1:max(0,i*batchsize-N))]; % rp(1:max(0,i*batchsize-N)) makes sure length(idx) = batchsize;
    X1batch=X1(idx,:);  X2batch=X2(idx,:); Ybatch = Y(idx);
    
    % Evaluate stochastic gradient.
    [E,grad]=DCCA_grad(VV,X1batch,X2batch,Ybatch,net1,net2,K,rcov1,rcov2); %  % ***** need to change    %% bookmark
    
    %% WX
%     eta =10000;

    
    delta=momentum*delta - eta*grad;  % Momentum.   % WX: delta(batch) ????
    VV=VV + delta;
  end
  
  %% Record the time spent for each epoch.
  its=its+1; TIME=[TIME, toc(t0)];
  %% Assemble the networks.
  idx=0;
  D=size(X1,2);
  for j=1:length(F1)
      % ==================================== ignore
    if strcmp(F1{j}.type,'conv')
      convdin=F1{j}.filternumrows*F1{j}.filternumcols*F1{j}.numinputmaps;
      convdout=F1{j}.numoutputmaps;
      W_seg=VV(idx+1:idx+(convdin+1)*convdout);
      F1{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F1{j}.units;
      % ==================================== ignore
    else
      units=F1{j}.units;
      W_seg=VV(idx+1:idx+(D+1)*units);
      F1{j}.W=reshape(W_seg,D+1,units);
      idx=idx+(D+1)*units; D=units;
    end
  end
  
  D=size(X2,2);
  for j=1:length(F2)
      % ==================================== ignore
    if strcmp(F2{j}.type,'conv')
      convdin=F2{j}.filternumrows*F2{j}.filternumcols*F2{j}.numinputmaps;
      convdout=F2{j}.numoutputmaps;
      W_seg=VV(idx+1:idx+(convdin+1)*convdout);
      F2{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F2{j}.units;
      % ==================================== ignore
    else
      units=F2{j}.units;
      W_seg=VV(idx+1:idx+(D+1)*units);
      F2{j}.W=reshape(W_seg,D+1,units);
      idx=idx+(D+1)*units; D=units;
    end
  end
  
  %% Compute correlations and errors.   WX 04/10/2018 Fix duplicated CCA procedure
%   FX1=deepnetfwd(X1,F1);  FX2=deepnetfwd(X2,F2);
%   [A,B,m1,m2,D]=linCCA(FX1,FX2,K,rcov1,rcov2);  clear FX1 FX2;
% %   SIGN=sign(A(1,:)+eps); A=bsxfun(@times,A,SIGN); B=bsxfun(@times,B,SIGN);      %% WX:  ???????????? 
%   f1.type='linear'; f1.units=K; f1.W=[A;-m1*A]; F1tmp=F1; F1tmp{end+1}=f1;
%   f2.type='linear'; f2.units=K; f2.W=[B;-m2*B]; F2tmp=F2; F2tmp{end+1}=f2;
%   clear A B m1 m2 D;
  
  %% WX 04/10/2018 Fix duplicated CCA procedure
  F1tmp=F1; F2tmp=F2;
  
  X_train=deepnetfwd(X1,F1tmp);
  CORR_train=[CORR_train, DCCA_corr(X_train,deepnetfwd(X2,F2tmp),Y,K)]; clear X_train;
  X_tune=deepnetfwd(XV1,F1tmp);
  CORR_tune=[CORR_tune, DCCA_corr(X_tune,deepnetfwd(XV2,F2tmp),YV,K)]; clear X_tune;
  if ~isempty(XTe1) && ~isempty(XTe2)
    X_test=deepnetfwd(XTe1,F1tmp);
    CORR_test=[CORR_test, DCCA_corr(X_test,deepnetfwd(XTe2,F2tmp),K)]; clear X_test;
  end
  %% the function of tuning data
  if CORR_tune(end)>optvalid
    optvalid=CORR_tune(end);
    fprintf('Epoch %d: getting better tuning correlation %f!\n',its,optvalid);
%     CORR_train
    F1opt=F1tmp;  F2opt=F2tmp;
  end
  F1opt=F1tmp;  F2opt=F2tmp;    %% WX !!!!
%   save(filename,'randseed','F1opt','F2opt','F1','F2','TIME', ...
%     'eta','delta','optvalid','CORR_train','CORR_tune','CORR_test');
end
