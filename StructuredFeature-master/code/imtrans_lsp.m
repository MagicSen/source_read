function [crop,iminfo] = imtrans_lsp(iminfo,patchsize,inputsize)

im = imreadx(iminfo);
% im = imresize(im,1.6);
%-------  resize the lenger size to 336   ------------
%% 需要图像统一resize到336*336
[y,x,~] = size(im);
%% inputsize(1) == 336
ratio = inputsize(1)/max([x,y]);

if ~isempty(iminfo.joints)
    %---------- positive -----------------
    im = imresize(im,ratio);
    %% keypoint也需要resize
    iminfo.joints = iminfo.joints.*ratio;
    %------- padding around then crop ---------
    [y,x,~] = size(im);
    padx = patchsize/2;  % 112
    pady = patchsize/2;  % 112 
    cent = [x/2+padx, y/2+pady];
    %% inputsize(1) == 336
    box = floor([cent-inputsize(1)/2+1 cent+inputsize(1)/2]);
    
    %% 填充矩阵
    padded = padarray(im, [padx, pady, 0], 0);
    
    crop = subarray(padded, box(2), box(4), box(1), box(3), 0);
    %% 得到新keypoints坐标
    joints = [iminfo.joints(:,1)-box(1) iminfo.joints(:,2)-box(2)]+1+padx;
    iminfo.joints = joints;
    iminfo.cropbox = box;
    iminfo.imrescale = ratio;
    %% 绘制结果
    if 0  % visualize
        conf = global_conf();
        pa = conf.pa;
        showskeletons_joints(crop,iminfo.joints,pa);
    end
else
    %----------- negative ------------------
    %     if size(im,1)<inputsize(1) || size(im,2)<inputsize(2)
    crop = imresize(im,[inputsize(1),inputsize(2)]);
    iminfo.cropbox = [];
    iminfo.imrescale = [];
end