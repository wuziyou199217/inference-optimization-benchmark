# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with PAI-Blade.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# os and numpy
import os
import numpy as np

# Tensorflow imports
import tensorflow.compat.v1 as tf

# Blade
import blade

import logging
import logging.handlers
import datetime
import time

# Ask tensorflow to limit its GPU memory to what's actually needed
# instead of gobbling everything that's available.
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# This way this tutorial is a little more friendly to sphinx-gallery.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))

######################################################################
# Required files
# -----------------------

model_path = "../models/BERT-Base_Uncased_tf.pb" 

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph_and_dic(model, batch_size, precise):
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Prepare input dictionary
    input_ids_1 = np.random.random((batch_size, 128)).astype(precise)
    input_mask_1 = np.random.random((batch_size, 128)).astype(precise)
    segment_ids_1 = np.random.random((batch_size, 128)).astype(precise)

    input_dic = {"input_ids_1:0": input_ids_1,
                    "input_mask_1:0": input_mask_1,
                    "segment_ids_1:0": segment_ids_1}
    
    return graph_def, input_dic 

graph_def, input_dic = create_graph_and_dic(model_path, 1, np.float32)

######################################################################
# Tensorflow Benchmark
# ------------------
# Generate Benchmark for tensorflow pb graph inference.

def tf_benchmark(graph_def, input_dic, run_count, times_per_run):
    """
    Benchmark the model
    """    
    tf.reset_default_graph()

    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
        softmax_tensor = sess.graph.get_tensor_by_name("loss/Softmax:0")
        # Benchmark! 
        duration_list = np.array([])
        for i in range(run_count):
            time_start = time.time()
            sess.run(softmax_tensor, input_dic)
            time_end = time.time()
            duration = (time_end - time_start) * 1000
            duration_list = np.append(duration_list, duration)

        # statistial analysis
        min = np.min(duration_list)
        max = np.max(duration_list)
        mean = np.mean(duration_list)

        ordered_list = np.sort(duration_list, axis=-1, kind='quicksort', order=None)
        pos99 = (np.rint(run_count * 0.99) - 1) if np.rint(run_count * 0.99) > 0 else 0
        p99 = ordered_list[pos99.astype(np.int)]

        print("tf_benchmark:")
        print("min max mean p99    unit/ms") 
        print([min, max, mean, p99]) 
        print("debug:")
        print("ordered_list:")
        print(ordered_list)
        print("pos99:")
        print(pos99)
    return

# tf_benchmark(graph_def, input_dic, 1000, 1)

######################################################################
# Tensorflow+Blade Benchmark
# ------------------
# Generate optimized model by Blade.

def blade_optimize(graph_def, input_dic, aggressive_opt):
    """
    Optimize the model by Blade
    """    
    opt_level = 'o1'
    if (aggressive_opt):
        opt_level = 'o2'

    optimized_model, opt_spec, report = blade.optimize(
        graph_def,             # 模型graph。
        opt_level,             # O1无损优化。或O2有损优化
        device_type='gpu',     # 面向GPU设备优化。
        test_data=[input_dic]  # 测试数据
    )
    
    # 打印优化报告
    print(report)
    # 存储优化模型
    # with tf.gfile.FastGFile('bert_base_opt.pb', mode='wb') as f:
    #     f.write(optimized_model.SerializeToString())
    return optimized_model, opt_spec

optimized_model, opt_spec = blade_optimize(graph_def, input_dic, False)
# optimized_model, opt_spec = blade_optimize(graph_def, input_dic, True)

with opt_spec:
    tf_benchmark(optimized_model, input_dic, 1000, 1)
