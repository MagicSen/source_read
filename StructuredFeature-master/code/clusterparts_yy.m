%% 聚类，K=13, pa=26
function idx = clusterparts_yy(deffeat,K,pa)

%% kmeans 重复聚类次数, K表示聚类的类别数
R = 50;
K = ones(1,length(pa))*K;
idx = cell(1,length(deffeat));
%% 遍历每个keypoints
for p = 1:length(deffeat)
    % create clustering feature
    %% 聚类前计算，如果为头部节点，计算头部节点到其他节点的偏移
    if pa(p) == 0
        i = 1;
        %% 得到头部节点后一个节点，这里 p == 1 , pa(p) = 0 , pa(i) == p ==> i = 2
        while pa(i) ~= p
            i = i+1;
        end
        %% X = deffeat{2} - deffeat{1}
        %% 计算相邻节点距离
        X = deffeat{i} - deffeat{p};
    else
        %% 非头部节点，计算相邻节点的距离
        %% X = deffeat{5} - deffeat{4}
        %% X = deffeat{6} - deffeat{3}
        X = deffeat{p} - deffeat{pa(p)};
    end
    % try multiple times kmeans
    gInd = cell(1,R);
    cen  = cell(1,R);
    sumdist = zeros(1,R);
    %% 关节点之间距离聚类
    fprintf('Clustering Class: %d \n',p);
    %% 循环聚类
    for trial = 1:R
        if mod(trial,10)==0
            fprintf('     trial: %d \n',trial);
        end
        %% 聚类 K(p) = K_old ==> 13
        [gInd{trial} cen{trial} sumdist(trial)] = k_means(X,K(p));
    end
    % take the smallest distance one
    %% 选择最小聚类误差的结果
    [dummy ind] = min(sumdist);
    %% 得到节点为p的聚类结果，每两个节点间类别为13
    %% 得到聚类误差最小的一次，得到类别下标
    idx{p} = gInd{ind(1)};
end






