function globals
%% 设置路径
if ~isdeployed
    addpath('./external/Dt_yy');
    addpath('./external/qpsolver');
    addpath('./dataio');
    addpath('./evaluation');
    addpath('./visualization');
    addpath('./tools');
    addpath('./external');
    addpath('./code');
    %% 配置全局变量
    conf = global_conf();
    %% 得到caffe的根目录
    caffe_root = conf.caffe_root;
    
    if exist(fullfile(caffe_root, '/matlab'), 'dir')
        addpath(genpath(fullfile(caffe_root, '/matlab')));
    else
        warning('Please install Caffe in %s', caffe_root);
    end
    %
    %   if exist(fullfile(caffe_root, '/matlab/caffe'), 'dir')
    %     addpath(fullfile(caffe_root, '/matlab/caffe'));
    %   else
    %     warning('Please install Caffe in %s', caffe_root);
    %   end
end
