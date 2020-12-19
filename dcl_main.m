function [tYV svmtYV] = dcldebug(datatype)

% [tt svmtt] = dcldebug(1);
%% tYV: the ground-truth label
%% svmtYV: the predicted label by DCL+SVM
%% datatype: 1, 2, 3 <-  [rest&nback, rest&emoid, nback&emoid]


%% 1 -> multi
% [X1, X2, XV1, XV2, Y, YV] = pncdataload(datatype);
addpath('C:\Disk D\one year\deepCCA\code\updated_data_1006');
dfold = 10;
tYV_all = cell(dfold,1); svmtYV_all = cell(dfold,1);

for ifold = 1:dfold
    ifold
        
    if datatype == 1
        filename = ['rest_nback_age07_f' num2str(ifold) '.mat'];
    	load(filename);
    end
    if datatype == 2
        filename = ['rest_emoid_age_f' num2str(ifold) '.mat'];
    	load(filename);
    end
    if datatype == 3
        filename = ['nback_emoid_age_f' num2str(ifold) '.mat'];
    	load(filename);
    end
    
X1 = snormalize(X1); XV1 = snormalize(XV1);
X2 = snormalize(X2); XV2 = snormalize(XV2); % vital command; improve performace dramatically
Y = Y_wrat; YV  = YV_wrat;

%% Fix the projection dimensionality to 10.
K=100;
%% Use the seed to reproduce the errors listed below.
randseed=8409;
%% Hyperparameters for DCCA network architecture.
% Regularizations for each view.
rcov1=1e-4; rcov2=1e-4;
% Hidden activation type.
hiddentype='relu';
% % Architecture (hidden layer sizes) for view 1 neural network.
% % Architecture (hidden layer sizes)  for view 2 neural network.
% NN2 = zeros(1,10)+1548; NN2(end+1) = K;
NN1 = zeros(1,10)+1548; NN1(end+1) = K;
NN2 = zeros(1,10)+1548; NN2(end+1) = K;
% Weight decay parameter.
l2penalty=1e-4;

%% Run DCCA with SGD. No pretraining is used.
% Minibatchsize.                                                           %%% Need to modify
% batchsize=800;
batchsize=800;
% Learning rate.
eta0=0.01;
%% WX: eta0
eta0 = 0.01;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay=1;
%% WX: decay
% Momentum.
momentum=0.9;
% How many passes of the data you run SGD with.
maxepoch=5; % WX: iteration steps
% addpath ./deepnet/
addpath('C:\Disk D\one year\deepCCA\code\deepnet');
[F1opt,F2opt]=DCCA_train(X1,X2,Y,XV1,XV2,YV,[],[],K,hiddentype,NN1,NN2, ...
  rcov1,rcov2,l2penalty,batchsize,eta0,decay,momentum,maxepoch);

% extract Top/Bottom 20% phenotype subjects
[temp iY] = sort(Y);
iy = iY([1:round(0.2*length(Y)) round(0.8*length(Y)):end]);
tX1 = X1(iy,:); tX2 = X2(iy,:);
tY_btm = Y( iY(1:round(0.2*length(Y)) )); tY_btm = zeros(size(tY_btm));
tY_top = Y( iY(round(0.8*length(Y)):end) ); tY_top = ones(size(tY_top));
tY = [tY_btm;tY_top];

[temp iYV] = sort(YV);
iyV = iYV([1:round(0.2*length(YV)) round(0.8*length(YV)):end]);
tXV1 = XV1(iyV,:); tXV2 = XV2(iyV,:);
tYV_btm = YV( iYV(1:round(0.2*length(YV)) )); tYV_btm = zeros(size(tYV_btm));
tYV_top = YV( iYV(round(0.8*length(YV)):end) ); tYV_top = ones(size(tYV_top));
tYV = [tYV_btm;tYV_top];
% Testing the learned networks.
X1proj = deepnetfwd(tX1,F1opt);
XV1proj = deepnetfwd(tXV1,F1opt);
%  XTe1proj = deepnetfwd(XTe1,F1opt);
X2proj = deepnetfwd(tX2,F2opt);
XV2proj = deepnetfwd(tXV2,F2opt);

Xproj = [X1proj X2proj]; XVproj = [XV1proj XV2proj];
Xproj = snormalize(Xproj); XVproj = snormalize(XVproj);

Xproj(isnan(Xproj)) = 0;
XVproj(isnan(XVproj)) = 0;
 
svmwork = svmtrain(Xproj,tY,'kernel_function','rbf');
% svmwork = svmtrain(Xproj,tY);
svmtYV = svmclassify(svmwork,XVproj);
% save('tsne_age_dcl_age_new.mat','XVproj','tYV')
% mappedX = tsne(XVproj,tYV);

tYV_all{ifold} = tYV; svmtYV_all{ifold} = svmtYV;

end

    if datatype == 1
    	save('rest_nback_wrat_dcl_result.mat','tYV_all','svmtYV_all');
    end
    if datatype == 2
    	save('rest_emoid_age_dcl_result.mat','tYV_all','svmtYV_all');
    end
    if datatype == 3
        save('nback_emoid_age_dcl_result.mat','tYV_all','svmtYV_all');
    end

end

