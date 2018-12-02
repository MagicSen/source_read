addpath('dataset/COCO/coco/MatlabAPI/');
addpath('../testing/util');

mkdir('dataset/COCO/mask2014')
vis = 0;

for mode = 0:1
    
    %% 根据类型读取评估集 or 训练集
    if mode == 1 
        load('dataset/COCO/mat/coco_kpt.mat');
    else
        load('dataset/COCO/mat/coco_val.mat');
        coco_kpt = coco_val;
    end
    %% 得到图像总数，准备遍历所有图像
    L = length(coco_kpt);
    %%
    
    for i = 1:L
        %% 打开原图以及存储mask的地址
        if mode == 1
            img_paths = sprintf('images/train2014/COCO_train2014_%012d.jpg', coco_kpt(i).image_id);
            img_name1 = sprintf('dataset/COCO/mask2014/train2014_mask_all_%012d.png', coco_kpt(i).image_id);
            img_name2 = sprintf('dataset/COCO/mask2014/train2014_mask_miss_%012d.png', coco_kpt(i).image_id);
        else
            img_paths = sprintf('images/val2014/COCO_val2014_%012d.jpg', coco_kpt(i).image_id);
            img_name1 = sprintf('dataset/COCO/mask2014/val2014_mask_all_%012d.png', coco_kpt(i).image_id);
            img_name2 = sprintf('dataset/COCO/mask2014/val2014_mask_miss_%012d.png', coco_kpt(i).image_id);
        end
        
        try
            %% 尝试读取mask，如果不成功则放弃
            display([num2str(i) '/ ' num2str(L)]);
            imread(img_name1);
            imread(img_name2);
            continue;
        catch
            display([num2str(i) '/ ' num2str(L)]);
            %joint_all(count).img_paths = RELEASE(i).image_id;
            %% 读取原图
            [h,w,~] = size(imread(['dataset/COCO/', img_paths]));
            %% 创建逻辑矩阵，假设全为false
            mask_all = false(h,w);
            mask_miss = false(h,w);
            %% flag 表示已有标记的mask label, 否则需要调用api得到
            flag = 0;
            for p = 1:length(coco_kpt(i).annorect)
                %if this person is annotated
                try
                    %% 如果这个人已经被标记过，得到图像i中第p个人的分割结果
                    seg = coco_kpt(i).annorect(p).segmentation{1};
                catch
                    %display([num2str(i) ' ' num2str(p)]);
                    %% 待确认
                    mask_crowd = logical(MaskApi.decode( coco_kpt(i).annorect(p).segmentation ));
                    %% 逻辑操作与，取多个人的联合mask_miss
                    temp = and(mask_all, mask_crowd);
                    mask_crowd = mask_crowd - temp;
                    flag = flag + 1;
                    coco_kpt(i).mask_crowd = mask_crowd;
                    continue;
                end
                %% 生成原图大小的mashgrid，对应X、Y为下标，
                %% inpolygon 给定的多边形，判断X,Y组成的点是否在多边形内部，返回一个逻辑矩阵
                [X,Y] = meshgrid( 1:w, 1:h );
                mask = inpolygon( X, Y, seg(1:2:end), seg(2:2:end));
                %% mask_all的就是这幅图内所有用户segment
                mask_all = or(mask, mask_all);
                %% 如果某个人的关键点没有，则需要将他的mask放入mask_miss
                if coco_kpt(i).annorect(p).num_keypoints <= 0
                    mask_miss = or(mask, mask_miss);
                end
            end
            if flag == 1
                %% mask_miss 就是背景
                mask_miss = not(or(mask_miss,mask_crowd));
                %% 求得所有人的mask
                mask_all = or(mask_all, mask_crowd);
            else
                mask_miss = not(mask_miss);
            end
            
            coco_kpt(i).mask_all = mask_all;
            coco_kpt(i).mask_miss = mask_miss;
            
            if mode == 1
                img_name = sprintf('dataset/COCO/mask2014/train2014_mask_all_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_all,img_name);
                img_name = sprintf('dataset/COCO/mask2014/train2014_mask_miss_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_miss,img_name);
            else
                img_name = sprintf('dataset/COCO/mask2014/val2014_mask_all_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_all,img_name);
                img_name = sprintf('dataset/COCO/mask2014/val2014_mask_miss_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_miss,img_name);
            end
            
            if flag == 1 && vis == 1
                im = imread(['dataset/COCO/', img_paths]);
                mapIm = mat2im(mask_all, jet(100), [0 1]);
                mapIm = mapIm*0.5 + (single(im)/255)*0.5;
                figure(1),imshow(mapIm);
                mapIm = mat2im(mask_miss, jet(100), [0 1]);
                mapIm = mapIm*0.5 + (single(im)/255)*0.5;
                figure(2),imshow(mapIm);
                mapIm = mat2im(mask_crowd, jet(100), [0 1]);
                mapIm = mapIm*0.5 + (single(im)/255)*0.5;
                figure(3),imshow(mapIm);
                pause;
                close all;
            elseif flag > 1
                display([num2str(i) ' ' num2str(p)]);
            end
        end
    end
    
    if mode == 1 
        save('coco_kpt_mask.mat', 'coco_kpt', '-v7.3');
    else
        coco_val = coco_kpt;
        save('coco_val_mask.mat', 'coco_val', '-v7.3');
    end
    
end
