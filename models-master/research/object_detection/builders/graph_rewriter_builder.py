# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for quantized training and evaluation."""

import tensorflow as tf

# 根据graph rewriter参数，构建graph rewriter，返回一个函数能够修改默认网络
def build(graph_rewriter_config, is_training):
  """Returns a function that modifies default graph based on options.

  Args:
    graph_rewriter_config: graph_rewriter_pb2.GraphRewriter proto.
    is_training: whether in training of eval mode.
  """
  def graph_rewrite_fn():
    # 只支持8位量化，否则报错
    """Function to quantize weights and activation of the default graph."""
    if (graph_rewriter_config.quantization.weight_bits != 8 or
        graph_rewriter_config.quantization.activation_bits != 8):
      raise ValueError('Only 8bit quantization is supported')

    # 根据是否训练来量化网络，包括权重与激活函数
    # Quantize the graph by inserting quantize ops for weights and activations
    if is_training:
      tf.contrib.quantize.experimental_create_training_graph(
          input_graph=tf.get_default_graph(),
          # 延迟量化
          quant_delay=graph_rewriter_config.quantization.delay
      )
    else:
      # 量化默认网络图
      tf.contrib.quantize.experimental_create_eval_graph(
          input_graph=tf.get_default_graph()
      )
    # 方便tensorboard查看量化参数
    tf.contrib.layers.summarize_collection('quant_vars')
  return graph_rewrite_fn
