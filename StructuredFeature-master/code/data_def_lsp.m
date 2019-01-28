function deffeat = data_def_lsp(pos)
% get absolute positions of parts with respect to HOG cell

% width  = zeros(1,length(pos));
%% 获得样本数目，一行为一个样本
height = zeros(1,length(pos));
%% 创建 3维矩阵，第三维表示样本数目，每一行表示一个keypoints详情，把keypoint信息整合到一起
points = zeros(size(pos(1).joints,1),size(pos(1).joints,2),length(pos));
for n = 1:length(pos)
    points(:,:,n) = pos(n).joints(:,1:2,:);
end

%% 根据 points 的 joints 数目分配 cell数组
deffeat = cell(1,size(points,1));
for p = 1:size(points,1)
  %% 得到 keypoints index为 p 的所有坐标，并且从 1 X 2 X n 转换为 2 X n
  def = squeeze(points(p,1:2,:));
  %% 坐标值scale缩小为原来1/8，并且转置
  deffeat{p} = (def/8)';
end