dataType = '';
addpath('dataset/COCO/coco/MatlabAPI');

mkdir('dataset/COCO/mat')

annTypes = { 'instances', 'captions', 'person_keypoints' };
annType=annTypes{3}; % specify dataType/annType

for mode = 0:1
    
    if mode == 0
        dataType= 'val2014';
        annFile=sprintf('dataset/COCO/annotations/%s_%s.json',annType,dataType);
    else
        dataType = 'train2014';
        annFile=sprintf('dataset/COCO/annotations/%s_%s.json',annType,dataType);
    end
    %% 调用Coco Matlab API
    coco=CocoApi(annFile);
    %%
    my_anno = coco.data.annotations;
    %%
    prev_id = -1;
    p_cnt = 1;
    cnt = 0;
    coco_kpt = [];
    
    for i = 1:1:size(my_anno,2)
        %% 遍历载入label的json文件，curr_id表示当前label所在的图像id，prev_id表示前一个label所在的图像id
        curr_id = my_anno(i).image_id;
        %% p_cnt表示当前图像中标记的总人数，cnt表示总图像数
        if(curr_id == prev_id)
            p_cnt = p_cnt + 1;
        else
            p_cnt = 1;
            cnt = cnt + 1;
        end
        %% coco_kpt这样的结构
        %% {"图像id1": {"人物1": {"bbox": xx, "segmentation": xx,... } "人物2": {"bbox": xx, "segmentation": xx,... }, ...}, "图像id2", {"人物1": {"bbox": xx, "segmentation": xx,... } "人物2": {"bbox": xx, "segmentation": xx,... }, ...}, ...}
        coco_kpt(cnt).image_id = curr_id;
        coco_kpt(cnt).annorect(p_cnt).bbox = my_anno(i).bbox;
        coco_kpt(cnt).annorect(p_cnt).segmentation = my_anno(i).segmentation;
        coco_kpt(cnt).annorect(p_cnt).area = my_anno(i).area;
        coco_kpt(cnt).annorect(p_cnt).id = my_anno(i).id;
        coco_kpt(cnt).annorect(p_cnt).iscrowd = my_anno(i).iscrowd;
        coco_kpt(cnt).annorect(p_cnt).keypoints = my_anno(i).keypoints;
        coco_kpt(cnt).annorect(p_cnt).num_keypoints = my_anno(i).num_keypoints;
        coco_kpt(cnt).annorect(p_cnt).img_width = coco.loadImgs(curr_id).width;
        coco_kpt(cnt).annorect(p_cnt).img_height = coco.loadImgs(curr_id).height;
        
        prev_id = curr_id;
        
        fprintf('%d/%d \n', i, size(my_anno, 2));
    end
    %%
    %% 两个集合，一个评估集，一个训练集
    if mode == 0
        coco_val = coco_kpt;
        save('dataset/COCO/mat/coco_val.mat', 'coco_val');
    else
        save('dataset/COCO/mat/coco_kpt.mat', 'coco_kpt');
    end
    
end