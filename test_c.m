clc;
clear;

A = load('fake.mat');
B = load('test.mat');
A_attr = A.attr;
A_label = A.label;
B_attr = B.attr;
B_label = B.label;


count = 0;

for i = 1:200
    min = 1e10;
    attr_a = A_attr(i, :);
    label_a = A_label(i);
    for j=1:200
        
        attr_b = B_attr(j, :);
        pd = (attr_a - attr_b) * (attr_a - attr_b)';
        if pd <= min
            ind = j;
            min = pd;
        end
    end
    if B_label(ind) == label_a
        count = count + 1;
    end    
end

acc = count /200;

%%
load('/home/zhanghf/work/deeplearning/dataset/SUNReady/sunready.mat');
X=[];Y=[];
for i=1:10
    X=[X;cnn_feat(labels==testClass(i),:)];
    Y=[Y;ones(sum(labels==testClass(i)),1)*testClass(i)];
end
load('/home/zhanghf/work/deeplearning/GAN/GAN_v2/fakefeat.mat');

idx=knnsearch(feat,X);
acc=sum(ind(idx)==Y)/200;

%%
idx=knnsearch(feat,feat,'k',2);
acc=sum(ind(idx(:,2))==ind)/200;