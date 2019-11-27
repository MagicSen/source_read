################################################################################################
# Defines global Caffe_LINK flag, This flag is required to prevent linker from excluding
# some objects which are not addressed directly but are registered via static constructors
macro(caffe_set_caffe_link)
  if(BUILD_SHARED_LIBS)
    set(Caffe_LINK caffe)
  else()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(Caffe_LINK -Wl,-force_load caffe)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(Caffe_LINK -Wl,--whole-archive caffe -Wl,--no-whole-archive)
    endif()
  endif()
endmacro()
# cmake_parse_arguments 用来生成以指定规则定义的变量名，例如这里生成 CAFFE_SOURCE_GROUP_GLOB="xxx;xxx;xxx"
################################################################################################
# Convenient command to setup source group for IDEs that support this feature (VS, XCode)
# Usage:
#   caffe_source_group(<group> GLOB[_RECURSE] <globbing_expression>)
function(caffe_source_group group)
  cmake_parse_arguments(CAFFE_SOURCE_GROUP "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(CAFFE_SOURCE_GROUP_GLOB)
    # 用来从路径里获取匹配GLOB模式的文件，GLOB类似正则匹配，GLOB_RECURSE表示递归匹配
    file(GLOB srcs1 ${CAFFE_SOURCE_GROUP_GLOB})
    # srcs1中文件的相对路径放到group里
    source_group(${group} FILES ${srcs1})
  endif()

  if(CAFFE_SOURCE_GROUP_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${CAFFE_SOURCE_GROUP_GLOB_RECURSE})
    source_group(${group} FILES ${srcs2})
  endif()
endfunction()

################################################################################################
# Collecting sources from globbing and appending to output list variable
# Usage:
#   caffe_collect_sources(<output_variable> GLOB[_RECURSE] <globbing_expression>)
function(caffe_collect_sources variable)
  cmake_parse_arguments(CAFFE_COLLECT_SOURCES "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(CAFFE_COLLECT_SOURCES_GLOB)
    file(GLOB srcs1 ${CAFFE_COLLECT_SOURCES_GLOB})
    set(${variable} ${variable} ${srcs1})
  endif()

  if(CAFFE_COLLECT_SOURCES_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${CAFFE_COLLECT_SOURCES_GLOB_RECURSE})
    set(${variable} ${variable} ${srcs2})
  endif()
endfunction()

################################################################################################
# Short command getting caffe sources (assuming standard Caffe code tree)
# Usage:
#   caffe_pickup_caffe_sources(<root>)
function(caffe_pickup_caffe_sources root)
  # put all files in source groups (visible as subfolder in many IDEs)
  caffe_source_group("Include"        GLOB "${root}/include/caffe/*.h*")
  caffe_source_group("Include\\Util"  GLOB "${root}/include/caffe/util/*.h*")
  caffe_source_group("Include"        GLOB "${PROJECT_BINARY_DIR}/caffe_config.h*")
  caffe_source_group("Source"         GLOB "${root}/src/caffe/*.cpp")
  caffe_source_group("Source\\Util"   GLOB "${root}/src/caffe/util/*.cpp")
  caffe_source_group("Source\\Layers" GLOB "${root}/src/caffe/layers/*.cpp")
  caffe_source_group("Source\\Cuda"   GLOB "${root}/src/caffe/layers/*.cu")
  caffe_source_group("Source\\Cuda"   GLOB "${root}/src/caffe/util/*.cu")
  caffe_source_group("Source\\Proto"  GLOB "${root}/src/caffe/proto/*.proto")

  # source groups for test target
  caffe_source_group("Include"      GLOB "${root}/include/caffe/test/test_*.h*")
  caffe_source_group("Source"       GLOB "${root}/src/caffe/test/test_*.cpp")
  caffe_source_group("Source\\Cuda" GLOB "${root}/src/caffe/test/test_*.cu")

  # collect files
  file(GLOB test_hdrs    ${root}/include/caffe/test/test_*.h*)
  file(GLOB test_srcs    ${root}/src/caffe/test/test_*.cpp)
  file(GLOB_RECURSE hdrs ${root}/include/caffe/*.h*)
  file(GLOB_RECURSE srcs ${root}/src/caffe/*.cpp)
  # 去掉重复的源文件
  list(REMOVE_ITEM  hdrs ${test_hdrs})
  list(REMOVE_ITEM  srcs ${test_srcs})

  # adding headers to make the visible in some IDEs (Qt, VS, Xcode)
  list(APPEND srcs ${hdrs} ${PROJECT_BINARY_DIR}/caffe_config.h)
  list(APPEND test_srcs ${test_hdrs})

  # collect cuda files
  file(GLOB    test_cuda ${root}/src/caffe/test/test_*.cu)
  file(GLOB_RECURSE cuda ${root}/src/caffe/*.cu)
  list(REMOVE_ITEM  cuda ${test_cuda})

  # add proto to make them editable in IDEs too
  file(GLOB_RECURSE proto_files ${root}/src/caffe/*.proto)
  list(APPEND srcs ${proto_files})

  # 转变为绝对路径
  # convert to absolute paths
  caffe_convert_absolute_paths(srcs)
  caffe_convert_absolute_paths(cuda)
  caffe_convert_absolute_paths(test_srcs)
  caffe_convert_absolute_paths(test_cuda)

  # 通过PARENT_SCOPE关键字修改父级目录的参数
  # propagate to parent scope
  set(srcs ${srcs} PARENT_SCOPE)
  set(cuda ${cuda} PARENT_SCOPE)
  set(test_srcs ${test_srcs} PARENT_SCOPE)
  set(test_cuda ${test_cuda} PARENT_SCOPE)
endfunction()

# 设置目标属性-值对儿
################################################################################################
# Short command for setting default target properties
# Usage:
#   caffe_default_properties(<target>)
function(caffe_default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX ${Caffe_DEBUG_POSTFIX}
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  # make sure we build all external dependencies first
  if (DEFINED external_project_dependencies)
    # 设置目标依赖
    add_dependencies(${target} ${external_project_dependencies})
  endif()
endfunction()

# 设置运行目录
################################################################################################
# Short command for setting runtime directory for build target
# Usage:
#   caffe_set_runtime_directory(<target> <dir>)
function(caffe_set_runtime_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()

# 设置解决方案目录
################################################################################################
# Short command for setting solution folder property for target
# Usage:
#   caffe_set_solution_folder(<target> <folder>)
function(caffe_set_solution_folder target folder)
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "${folder}")
  endif()
endfunction()

################################################################################################
# Reads lines from input file, prepends source directory to each line and writes to output file
# Usage:
#   caffe_configure_testdatafile(<testdatafile>)
function(caffe_configure_testdatafile file)
  file(STRINGS ${file} __lines)
  set(result "")
  foreach(line ${__lines})
    set(result "${result}${PROJECT_SOURCE_DIR}/${line}\n")
  endforeach()
  file(WRITE ${file}.gen.cmake ${result})
endfunction()

################################################################################################
# Filter out all files that are not included in selected list
# Usage:
#   caffe_leave_only_selected_tests(<filelist_variable> <selected_list>)
function(caffe_leave_only_selected_tests file_list)
  if(NOT ARGN)
    return() # blank list means leave all
  endif()
  string(REPLACE "," ";" __selected ${ARGN})
  list(APPEND __selected caffe_main)

  set(result "")
  foreach(f ${${file_list}})
    get_filename_component(name ${f} NAME_WE)
    string(REGEX REPLACE "^test_" "" name ${name})
    list(FIND __selected ${name} __index)
    if(NOT __index EQUAL -1)
      list(APPEND result ${f})
    endif()
  endforeach()
  set(${file_list} ${result} PARENT_SCOPE)
endfunction()

