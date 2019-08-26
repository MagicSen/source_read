# 设置cmake最小支持的版本
cmake_minimum_required(VERSION 2.8.7)

# 设置cmake的策略支持项 
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# 设置项目名称及代码语言，对应设置cmake的内部参数PROJECT_NAME，如果是第一级目录同时设置CMAKE_PROJECT_NAME，默认项目语言是C CXX
# ---[ Caffe project
project(Caffe C CXX)

# 设置caffe的版本信息，set设置变量名，cmake中变量名大小写敏感，对应unset用于取消变量绑定，
# cmake中变量范围分为三种，function scope, director scope, persistent cache，优先访问function scope(函数范围内) ==> director scope(本级目录及子目录) ==> persistent cache(全局范围) ，可以访问系统环境变量也可在运行过程中改变环境变量的值(不是真的改变系统变量的值)
# cmake中未set的变量默认值为空
# cmake中调用系统变量 $ENV{<variable>}，调用Cache变量$CACHE{<variable>}
# CACHE STRING后面跟的是对这个变量的注释，STRING可以替换为BOOL等，激活cmake-gui里面的定义
# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")

# 为该目录及子目录下增加编译器flag参数
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# list将把value的值append到CMAKE_MODULE_PATH，用分号间隔
# PROJECT_SOURCE_DIR变量是CMakeList.txt同级目录，项目根目录
# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# 载入及运行cmake脚本或者模块
# 如果是模块名会在CMAKE_MODULE_PATH中寻找相应名称的 *.cmake 脚本文件
# ExternalProject为cmake支持的下载第三方库工具，配置及解释详见gflags.cmake glog.cmake
# 载入GNU标准安装软件的目录定义方式
include(ExternalProject)
include(GNUInstallDirs)

# 依次调用cmake脚本
# Utils.cmake: 加载定义好的工具函数
# Targets.cmake: 加载源文件及定义link参数的配置函数
# Misc.cmake: 配置项目及工程，设置debug/release环境
# Summary.cmake: caffe配置项状态打印
# ConfigGen.cmake: 生成工程文件
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)


# 配置caffe的编译flag
# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
# 设置python版本
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)

# This code is taken from https://github.com/sh1r0/caffe-android-lib
caffe_option(USE_HDF5 "Build with hdf5" ON)

# 设置项目相关的include,lib依赖
# ---[ Dependencies
include(cmake/Dependencies.cmake)

# 设置g++ 编译参数，例如警告级别
# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

# 设置caffe的link选项
caffe_set_caffe_link()

# 使用标准c++库
if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  # 打印配置文件
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# 设置编译警告级别
# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# PROJECT_BINARY_DIR表示build的根目录完整目录
# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# 设置Caffe的源文件及include路径
# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

# 添加CUDA相关依赖
# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  # 删除list中带有关键字的item
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  # 删除list中带有关键字的item
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  # 添加编译的flag
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# 添加子目录
# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

#添加python编译依赖
# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

