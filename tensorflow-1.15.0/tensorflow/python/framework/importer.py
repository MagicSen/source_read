# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A utility function for importing TensorFlow graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.core.framework import graph_pb2
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python import tf2
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


def _IsControlInput(input_name):
  # Expected format: '^operation_name' (control input).
  return input_name.startswith('^')


# 解析Tensor名称，返回操作数名称以及输出下标
def _ParseTensorName(tensor_name):
  """Parses a tensor name into an operation name and output index.

  This function will canonicalize tensor names as follows:

  * "foo:0"       -> ("foo", 0)
  * "foo:7"       -> ("foo", 7)
  * "foo"         -> ("foo", 0)
  * "foo:bar:baz" -> ValueError

  Args:
    tensor_name: The name of a tensor.

  Returns:
    A tuple containing the operation name, and the output index.

  Raises:
    ValueError: If `tensor_name' cannot be interpreted as the name of a tensor.
  """
  components = tensor_name.split(':')
  if len(components) == 2:
    # Expected format: 'operation_name:output_index'.
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError('Cannot convert %r to a tensor name.' % (tensor_name,))
    return components[0], output_index
  elif len(components) == 1:
    # Expected format: 'operation_name' (implicit 0th output).
    return components[0], 0
  else:
    raise ValueError('Cannot convert %r to a tensor name.' % (tensor_name,))


@contextlib.contextmanager
def _MaybeDevice(device):
  """Applies the given device only if device is not None or empty."""
  if device:
    with ops.device(device):
      yield
  else:
    yield


def _ProcessGraphDefParam(graph_def, op_dict):
  """Type-checks and possibly canonicalizes `graph_def`."""
  if not isinstance(graph_def, graph_pb2.GraphDef):
    # `graph_def` could be a dynamically-created message, so try a duck-typed
    # approach
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('graph_def must be a GraphDef proto.')
  else:
    # If we're using the graph_def provided by the caller, modify graph_def
    # in-place to add attr defaults to the NodeDefs (this is visible to the
    # caller).
    # NOTE(skyewm): this is undocumented behavior that at least meta_graph.py
    # depends on. It might make sense to move this to meta_graph.py and have
    # import_graph_def not modify the graph_def argument (we'd have to make sure
    # this doesn't break anything else.)
    # 如果节点操作不在op_dict，则跳过
    for node in graph_def.node:
      if node.op not in op_dict:
        # Assume unrecognized ops are functions for now. TF_ImportGraphDef will
        # report an error if the op is actually missing.
        continue
      # 根据op_dict得到操作的定义
      op_def = op_dict[node.op]
      # 通过op_def补全node中没有的value
      _SetDefaultAttrValues(node, op_def)

  return graph_def


# 确保输入input_map合法
def _ProcessInputMapParam(input_map):
  """Type-checks and possibly canonicalizes `input_map`."""
  if input_map is None:
    input_map = {}
  else:
    # 确认input_map数据格式为: string: Tensor
    if not (isinstance(input_map, dict) and all(
        isinstance(k, compat.bytes_or_text_types) for k in input_map.keys())):
      raise TypeError('input_map must be a dictionary mapping strings to '
                      'Tensor objects.')
  return input_map


# 检查返回的参数
def _ProcessReturnElementsParam(return_elements):
  """Type-checks and possibly canonicalizes `return_elements`."""
  if return_elements is None:
    return None
  if not all(
      isinstance(x, compat.bytes_or_text_types) for x in return_elements):
    raise TypeError('return_elements must be a list of strings.')
  return tuple(compat.as_str(x) for x in return_elements)


def _FindAttrInOpDef(attr_name, op_def):
  for attr_def in op_def.attr:
    if attr_name == attr_def.name:
      return attr_def
  return None


def _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def):
  """Removes unknown default attrs according to `producer_op_list`.

  Removes any unknown attrs in `graph_def` (i.e. attrs that do not appear in
  the OpDefs in `op_dict`) that have a default value in `producer_op_list`.

  Args:
    op_dict: dict mapping operation name to OpDef.
    producer_op_list: OpList proto.
    graph_def: GraphDef proto
  """
  producer_op_dict = {op.name: op for op in producer_op_list.op}
  for node in graph_def.node:
    # Remove any default attr values that aren't in op_def.
    if (node.op in producer_op_dict
        # Some custom op registrations won't show up here. That's OK, attribute
        # stripping just won't be available.
        and node.op in op_dict):
      op_def = op_dict[node.op]
      producer_op_def = producer_op_dict[node.op]
      # We make a copy of node.attr to iterate through since we may modify
      # node.attr inside the loop.
      for key in list(node.attr):
        if _FindAttrInOpDef(key, op_def) is None:
          # No attr_def in consumer, look in producer.
          attr_def = _FindAttrInOpDef(key, producer_op_def)
          if (attr_def and attr_def.HasField('default_value') and
              node.attr[key] == attr_def.default_value):
            # Unknown attr had default value in producer, delete it so it can be
            # understood by consumer.
            del node.attr[key]


def _ConvertInputMapValues(name, input_map):
  """Ensures all input map values are tensors.

  This should be called from inside the import name scope.

  Args:
    name: the `name` argument passed to import_graph_def
    input_map: the `input_map` argument passed to import_graph_def.

  Returns:
    An possibly-updated version of `input_map`.

  Raises:
    ValueError: if input map values cannot be converted due to empty name scope.
  """
  # 确保input_map的value都是Tensor
  if not all(isinstance(v, ops.Tensor) for v in input_map.values()):
    if name == '':  # pylint: disable=g-explicit-bool-comparison
      raise ValueError(
          'tf.import_graph_def() requires a non-empty `name` if `input_map` '
          'contains non-Tensor values. Try calling tf.convert_to_tensor() on '
          '`input_map` values before calling tf.import_graph_def().')
    # 在新命名空间_inputs中，创建输入的tensor
    with ops.name_scope('_inputs'):
      input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}
  return input_map


def _PopulateTFImportGraphDefOptions(options, prefix, input_map,
                                     return_elements,
                                     validate_colocation_constraints):
  # 设置前缀
  """Populates the TF_ImportGraphDefOptions `options`."""
  c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix)
  c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options, True)

  for input_src, input_dst in input_map.items():
    input_src = compat.as_str(input_src)
    # 解析操作符OPS
    if input_src.startswith('^'):
      src_name = compat.as_str(input_src[1:])
      dst_op = input_dst._as_tf_output().oper  # pylint: disable=protected-access
      c_api.TF_ImportGraphDefOptionsRemapControlDependency(
          options, src_name, dst_op)
    else:
    # 解析Tensor
      src_name, src_idx = _ParseTensorName(input_src)
      src_name = compat.as_str(src_name)
      dst_output = input_dst._as_tf_output()  # pylint: disable=protected-access
      c_api.TF_ImportGraphDefOptionsAddInputMapping(options, src_name, src_idx,
                                                    dst_output)
  for name in return_elements or []:
    # 解析tensor
    if ':' in name:
      op_name, index = _ParseTensorName(name)
      op_name = compat.as_str(op_name)
      c_api.TF_ImportGraphDefOptionsAddReturnOutput(options, op_name, index)
    # 解析操作数
    else:
      c_api.TF_ImportGraphDefOptionsAddReturnOperation(options,
                                                       compat.as_str(name))

  c_api.TF_ImportGraphDefOptionsSetValidateColocationConstraints(
      options, validate_colocation_constraints)


def _ProcessNewOps(graph):
  """Processes the newly-added TF_Operations in `graph`."""
  # Maps from a node to the names of the ops it's colocated with, if colocation
  # is specified in the attributes.
  colocation_pairs = {}

  for new_op in graph._add_new_tf_operations(compute_devices=False):  # pylint: disable=protected-access
    original_device = new_op.device
    new_op._set_device('')  # pylint: disable=protected-access
    colocation_names = _GetColocationNames(new_op)
    if colocation_names:
      colocation_pairs[new_op] = colocation_names
      # Don't set a device for this op, since colocation constraints override
      # device functions and the original device. Note that this op's device may
      # still be set by the loop below.
      # TODO(skyewm): why does it override the original device?
    else:
      with _MaybeDevice(original_device):
        graph._apply_device_functions(new_op)  # pylint: disable=protected-access

  # The following loop populates the device field of ops that are colocated
  # with another op.  This is implied by the colocation attribute, but we
  # propagate the device field for completeness.
  for op, coloc_op_list in colocation_pairs.items():
    coloc_device = None
    # Find any device in the list of colocated ops that have a device, if it
    # exists.  We assume that if multiple ops have devices, they refer to the
    # same device.  Otherwise, a runtime error will occur since the colocation
    # property cannot be guaranteed.  Note in TF2 colocations have been removed
    # from the public API and will be considered a hint, so there is no runtime
    # error.
    #
    # One possible improvement is to try to check for compatibility of all
    # devices in this list at import time here, which would require
    # implementing a compatibility function for device specs in python.
    for coloc_op_name in coloc_op_list:
      try:
        coloc_op = graph._get_operation_by_name_unsafe(coloc_op_name)  # pylint: disable=protected-access
      except KeyError:
        # Do not error in TF2 if the colocation cannot be guaranteed
        if tf2.enabled() or control_flow_util.EnableControlFlowV2(graph):
          continue

        raise ValueError('Specified colocation to an op that '
                         'does not exist during import: %s in %s' %
                         (coloc_op_name, op.name))
      if coloc_op.device:
        coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
        break
    if coloc_device:
      op._set_device(coloc_device)  # pylint: disable=protected-access


def _GetColocationNames(op):
  """Returns names of the ops that `op` should be colocated with."""
  colocation_names = []
  try:
    class_values = op.get_attr('_class')
  except ValueError:
    # No _class attr
    return
  for val in class_values:
    val = compat.as_str(val)
    if val.startswith('loc:@'):
      colocation_node_name = val[len('loc:@'):]
      if colocation_node_name != op.name:
        colocation_names.append(colocation_node_name)
  return colocation_names


def _GatherReturnElements(requested_return_elements, graph, results):
  """Returns the requested return elements from results.

  Args:
    requested_return_elements: list of strings of operation and tensor names
    graph: Graph
    results: wrapped TF_ImportGraphDefResults

  Returns:
    list of `Operation` and/or `Tensor` objects
  """
  return_outputs = c_api.TF_ImportGraphDefResultsReturnOutputs(results)
  return_opers = c_api.TF_ImportGraphDefResultsReturnOperations(results)

  combined_return_elements = []
  outputs_idx = 0
  opers_idx = 0
  for name in requested_return_elements:
    # 如果是tensor
    if ':' in name:
      combined_return_elements.append(
          graph._get_tensor_by_tf_output(return_outputs[outputs_idx]))  # pylint: disable=protected-access
      outputs_idx += 1
    # 如果是操作数
    else:
      combined_return_elements.append(
          graph._get_operation_by_tf_operation(return_opers[opers_idx]))  # pylint: disable=protected-access
      opers_idx += 1
  return combined_return_elements


def _SetDefaultAttrValues(node_def, op_def):
  """Set any default attr values in `node_def` that aren't present."""
  assert node_def.op == op_def.name
  # 获取操作的属性
  for attr_def in op_def.attr:
    key = attr_def.name
    # 有这个属性域且有默认数值
    if attr_def.HasField('default_value'):
      value = node_def.attr[key]
      # 如果value为空或者没有，则从全集中拷贝过来
      if value is None or value.WhichOneof('value') is None:
        node_def.attr[key].CopyFrom(attr_def.default_value)


@tf_export('graph_util.import_graph_def', 'import_graph_def')
@deprecated_args(None, 'Please file an issue at '
                 'https://github.com/tensorflow/tensorflow/issues if you depend'
                 ' on this feature.', 'op_dict')
def import_graph_def(graph_def,
                     input_map=None,
                     return_elements=None,
                     name=None,
                     op_dict=None,
                     producer_op_list=None):
  # import graph为默认的graph
  """Imports the graph from `graph_def` into the current default `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.

  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into
      the default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def`
      to `Tensor` objects. The values of the named input tensors in the
      imported graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in
      `graph_def` that will be returned as `Operation` objects; and/or
      tensor names in `graph_def` that will be returned as `Tensor` objects.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    op_dict: (Optional.) Deprecated, do not use.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  return _import_graph_def_internal(
      graph_def,
      input_map=input_map,
      return_elements=return_elements,
      name=name,
      op_dict=op_dict,
      producer_op_list=producer_op_list)


def import_graph_def_for_function(  # pylint: disable=invalid-name
    graph_def, name=None):
  """Like import_graph_def but does not validate colocation constraints."""
  return _import_graph_def_internal(
      graph_def, validate_colocation_constraints=False, name=name)


def _import_graph_def_internal(  # pylint: disable=invalid-name
    graph_def,
    input_map=None,
    return_elements=None,
    validate_colocation_constraints=True,
    name=None,
    op_dict=None,
    producer_op_list=None):
  """Imports the graph from `graph_def` into the current default `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.

  Args:
    # 一个按照proto格式定义的graph，可直接载入到默认的graph中
    graph_def: A `GraphDef` proto containing operations to be imported into the
      default graph.
    # 一个输入映射表，用来得到graph_def中的Tensor与名称的映射关系
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    # 一系列在graph_def中的操作符/Tensor的名称(字符串)
    return_elements: A list of strings containing operation names in `graph_def`
      that will be returned as `Operation` objects; and/or tensor names in
      `graph_def` that will be returned as `Tensor` objects.
    # 是否需要验证
    validate_colocation_constraints: Whether to validate colocation constraints.
    # 添加前缀，对于函数名不起作用，函数名前默认加import
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    op_dict: (Optional.) Deprecated, do not use.
    # 使某些默认参数失效
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.

  Returns:
    # 返回returen_elements里需要得到的Operation或Tensor
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  # 获取所有注册的op操作
  op_dict = op_def_registry.get_registered_ops()

  # 初始化内部某些属性，并且确认参数有效
  graph_def = _ProcessGraphDefParam(graph_def, op_dict)
  input_map = _ProcessInputMapParam(input_map)
  return_elements = _ProcessReturnElementsParam(return_elements)

  # 处理Tensor
  # 移除producer_op_list相关的操作
  if producer_op_list is not None:
    # TODO(skyewm): make a copy of graph_def so we're not mutating the argument?
    _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def)

  # 获取默认graph
  graph = ops.get_default_graph()
  # 在指定名称作用域里获取Tensor
  with ops.name_scope(name, 'import', input_map.values()) as scope:
    # 保存前缀名称，剔除'/'这个符号
    # Save unique prefix generated by name_scope
    if scope:
      # 如果结尾不为'/'报警
      assert scope.endswith('/')
      prefix = scope[:-1]
    else:
      prefix = ''

    # 在该名称空间中构造输入的tensor映射
    # Generate any input map tensors inside name scope
    input_map = _ConvertInputMapValues(name, input_map)

  # 类型 TF_NewImportGraphDefOptions，增加一个析构函数
  # 这里几乎都是调用底层C函数完成模型抽取
  scoped_options = c_api_util.ScopedTFImportGraphDefOptions()
  options = scoped_options.options
  _PopulateTFImportGraphDefOptions(options, prefix, input_map, return_elements,
                                   validate_colocation_constraints)

  # _ProcessNewOps mutates the new operations. _mutation_lock ensures a
  # Session.run call cannot occur between creating the TF_Operations in the
  # TF_GraphImportGraphDefWithResults call and mutating the them in
  # _ProcessNewOps.
  # 加锁
  with graph._mutation_lock():  # pylint: disable=protected-access
    with c_api_util.tf_buffer(graph_def.SerializeToString()) as serialized:
      try:
        results = c_api.TF_GraphImportGraphDefWithResults(
            graph._c_graph, serialized, options)  # pylint: disable=protected-access
        results = c_api_util.ScopedTFImportGraphDefResults(results)
      except errors.InvalidArgumentError as e:
        # Convert to ValueError for backwards compatibility.
        raise ValueError(str(e))

    # Create _DefinedFunctions for any imported functions.
    #
    # We do this by creating _DefinedFunctions directly from `graph_def`, and
    # adding them to `graph`. Adding an existing function to a TF_Graph is a
    # no-op, so this only has the effect of updating the Python state (usually
    # _DefinedFunction.add_to_graph also adds the function to the TF_Graph).
    #
    # TODO(skyewm): fetch the TF_Functions directly from the TF_Graph
    # TODO(skyewm): avoid sending serialized FunctionDefs back to the TF_Graph
    # 添加新的Ops，这块儿没太看懂
    _ProcessNewOps(graph)

  if graph_def.library and graph_def.library.function:
    functions = function.from_library(graph_def.library)
    for f in functions:
      f.add_to_graph(graph)
  # 检查input map 是否都在graph中，如果不在则认为是一个错误
  # Treat input mappings that don't appear in the graph as an error, because
  # they are likely to be due to a typo.
  missing_unused_input_keys = (
      c_api.TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
          results.results))
  if missing_unused_input_keys:
    missing_unused_input_keys = [
        compat.as_str(s) for s in missing_unused_input_keys
    ]
    raise ValueError(
        'Attempted to map inputs that were not found in graph_def: [%s]' %
        ', '.join(missing_unused_input_keys))

  if return_elements is None:
    return None
  else:
    return _GatherReturnElements(return_elements, graph, results.results)
