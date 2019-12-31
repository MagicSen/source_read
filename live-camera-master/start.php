<?php
/**
 * run with command 
 * php start.php start
 */
// 入口
ini_set('display_errors', 'on');
// 声明命名空间
use Workerman\Worker;

// pcntl扩展是PHP在Linux环境下进程控制的重要扩展
// 检查扩展
if(!extension_loaded('pcntl'))
{
    exit("Please install pcntl extension. See http://doc3.workerman.net/install/install.html\n");
}

// posix扩展使得PHP在Linux环境可以调用系统通过POSIX标准提供的接口。WorkerMan主要使用了其相关的接口实现了守护进程化、用户组控制等功能。此扩展win平台不支持
if(!extension_loaded('posix'))
{
    exit("Please install posix extension. See http://doc3.workerman.net/install/install.html\n");
}

// 标记是全局启动
define('GLOBAL_START', 1);
# 包含文件
require_once __DIR__ . '/vendor/autoload.php';

// 加载所有Applications/*/start.php，以便启动所有服务
foreach(glob(__DIR__.'/start_*.php') as $start_file)
{
    // 这部分运行在子进程
    require_once $start_file;
}

# 以debug（调试）方式启动
# php start.php start
# 以daemon（守护进程）方式启动
# php start.php start -d
# 一般来说在Worker::runAll();调用前运行的代码都是在主进程运行的，onXXX回调运行的代码都属于子进程。注意写在Worker::runAll();后面的代码永远不会被执行
// 运行所有服务
Worker::runAll();
