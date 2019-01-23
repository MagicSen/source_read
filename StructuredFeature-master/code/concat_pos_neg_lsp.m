function pos_train = concat_pos_neg_lsp(pos_train,neg_train)
%% 为方便连接，为负样本配置变量
for i = 1:length(neg_train)
     neg_train(i).idx = [];
     neg_train(i).scale_x = [];
     neg_train(i).scale_y = [];
end

% pos_train = neg_train;
pos_train = [pos_train; neg_train];