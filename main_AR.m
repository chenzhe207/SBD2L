clear all
clc
close all
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP



load AR_120_50_40.mat
DATA = sample_data;
DATA = DATA./ repmat(sqrt(sum(DATA .* DATA)),[size(DATA, 1) 1]);
Label = sample_label;
c = length(unique(Label));
K = 120;
train_num = 8;
EachClassNum = 26;
cand_set = [9 : 13, 22 : 26]; %% 遮挡样本候选集

%% parameters

alpha = 0.0005;
beta =  0.0001;
lambda1 = 0.0005;
lambda2 = 0.0001;
    
%% select training and test samples
cand_set = [9 : 13, 22 : 26];
for ii = 1 : 10
%% select training and test samples
temp = zeros(1, 26);
temp([1 : 7, cand_set(ii)]) = 1;
train_ind=logical(repmat(temp, 1, 120));%训练样本索引

cand_set_res = cand_set;
cand_set_res(ii) = [];
temp1 = zeros(1, 26);
temp1([14 : 20, cand_set_res]) = 1;
test_ind = logical(repmat(temp1, 1, 120));%测试样本索引
    
train_data = DATA(:,train_ind);
test_data = DATA(:,test_ind);
    
train_label = Label(train_ind);
test_label = Label(test_ind);
   
for i = 1 : size(train_data, 2)
    a = train_label(i);
    Htr(a, i) = 1;
end
for i = 1 : (size(test_data, 2))
    a = test_label(i);
    Htt(a, i) = 1;
end
%% dictionary initialize

[Dinit] = initializationDictionary(train_data, Htr, K, 10, 30);

%% starting robust sparse and low-rank recovery
[A, B, D, R, value] = DL(train_data, Dinit, train_num, c, alpha, beta, lambda1, lambda2);


[At, Bt] = Test_coeff(test_data, D, alpha, beta);
  
%%  starting learning linear classifier
% gama = 0.001;
% [W, M] = classifier_learning(A, c, Htr, gama);
% accuracy(ii) = classification(W, Htt, At) * 100


W = inv(A * A' + 0.1 * eye(size(A * A'))) * A * Htr';
W = W';
accuracy(ii)  = classification(W, Htt, At) * 100

end

mean(accuracy) 
std(accuracy)

