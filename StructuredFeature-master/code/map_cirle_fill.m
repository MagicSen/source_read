function mask = map_cirle_fill(mask,joints,lgt_scale,dist_map)

x = joints(2);
y = joints(1);
%% dist map中心
dist_cen = (size(dist_map,1)-1)/2+1;
%   crop_dist = dist_map( dist_cen - scale_limt: dist_cen + scale_limt, ...
%   dist_cen - scale_limt: dist_cen + scale_limt);
lgt_crop_scale = ceil(lgt_scale);
% 剪切距离图
crop_dist = dist_map( dist_cen - lgt_crop_scale: dist_cen + lgt_crop_scale, ...
    dist_cen - lgt_crop_scale: dist_cen + lgt_crop_scale);

%% 小于阈值的置为-1，大于阈值置为0
crop = crop_dist;
crop(crop_dist<=lgt_scale) = -1;    % value within thresh
crop(crop_dist>lgt_scale) = 0;       % value outside the thresh

%---------- stick it to label map ------------------
%% 以这个关节点为中心，设置crop的结果
mask(x-lgt_crop_scale: x+lgt_crop_scale, y-lgt_crop_scale:y+lgt_crop_scale) = crop;


