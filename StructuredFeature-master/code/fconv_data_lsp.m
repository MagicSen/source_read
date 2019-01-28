function data = fconv_data_lsp(name,pos_train,pos_val,neg_train,lmdb_dir,cachedir,taskid)

%----- set parameters whenever use ------------------
patchsize = [224,224];
inputsize = [336 336];
labelsize = inputsize/8;

%% 临时文件保存路径
cachedir = [cachedir 'lsp_fconv/' name '_datainfo/'];  % dataset statistics

if ~isdir(lmdb_dir),mkdir(lmdb_dir);end
if ~isdir(cachedir),mkdir(cachedir);end

%% 得到关键点的mask以及map
switch taskid
    case 1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %-------------- pos_train data ------------------------
        %% 组合正负样本
        train_data = concat_pos_neg_lsp(pos_train,neg_train);
        train_data = data_shuffle('train',train_data,cachedir);
        %% 剪裁图像，变换尺寸，保证图像为336X336
        data = lmdb_fconv_data_lsp('train',train_data,patchsize,inputsize,cachedir,lmdb_dir);
        LabelMapGen_strict('train',data,labelsize,cachedir,lmdb_dir);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %-------------- pos_val data ------------------------
    case 2
        data = lmdb_fconv_data_lsp('val',pos_val,patchsize,inputsize,cachedir,lmdb_dir);
        LabelMapGen_strict('val',data,labelsize,cachedir,lmdb_dir);
end