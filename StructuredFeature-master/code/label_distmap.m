function dist_map = label_distmap(cache,num)

try
    load([cache 'dist_map' num2str(num) '.mat']);
catch
    %% 29 X 29 矩阵
    dist_map = zeros(num,num);
    %% 计算中心点坐标
    cent = (num-1)/2+1;
    for i = 1:num
        for j = 1:num
            dist = sqrt((i-cent)^2+(j-cent)^2);
            %% 距离map
            dist_map(i,j) = dist;
        end
    end
    save([cache 'dist_map' num2str(num) '.mat'],'dist_map');
end