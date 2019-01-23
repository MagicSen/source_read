function Shuffled = data_shuffle(name,pos_train,cachedir)
%% 随机训练集
Shuffledir = [cachedir 'Shuffled_'];

try
    load([Shuffledir name])
catch
    order = randperm(length(pos_train));
    Shuffled = pos_train(order);
    save([Shuffledir name],'Shuffled');
end