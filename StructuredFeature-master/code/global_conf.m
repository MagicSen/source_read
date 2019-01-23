function conf = global_conf()
%% 不支持多进程
assert_not_in_parallel_worker();
% dataset
%% 设置数据db生产的参数
conf.interval = 10; % 10 levels from 1 to 1/2
conf.memsize = 0.5; % 0.5 gb
conf.NEG_N = 80;
%% GPU id
conf.device_id = 0;
% % conf.caffe_root = './external/caffe';
% conf.caffe_root = './external/caffe2.0';
%% 设置caffe路径
conf.caffe_root = './caffe-multi';
% default configurations
conf.mining_neg = true;
conf.mining_pos = false;
%% 设置关键点数量
conf.K = 13;
conf.test_with_detection = false;

conf.useGpu = 1;
conf.batch_size = 1024;

conf.at_least_one = true;

% override some configurations
global GLOBAL_OVERRIDER;
if ~isempty(GLOBAL_OVERRIDER)
    % modify some fields of conf
    conf = GLOBAL_OVERRIDER(conf);
    % ======= fields constructed from existing configurations =======
    conf.note = ['CNN_Deep_', num2str(conf.K)];
    conf.cachedir = ['./cache/flic/'];
    conf.lmdbdir = 'external/data/';
    conf.fconvdir = 'cache/flic_fconv/';
    if ~exist(conf.cachedir, 'dir')
        mkdir(conf.cachedir);
    end
    %% 设置平均图像
    conf.cnn.image_mean_file = [conf.cachedir, conf.dataset, '_mean.mat'];
end

function assert_not_in_parallel_worker()
% Matlab does not support accessing global variables from
% parallel workers. The result of reading a global is undefined
% and in practice has odd and inconsistent behavoir.
% The configuraton override mechanism relies on a global
% variable. To avoid hard-to-find bugs, we make sure that
% global_conf cannot be called from a parallel worker.
t = [];
if usejava('jvm')
    try
        t = getCurrentTask();
    catch
    end
end
if ~isempty(t)
    msg = ['global_conf() cannot be called from a parallel worker '];
    error(msg);
end
