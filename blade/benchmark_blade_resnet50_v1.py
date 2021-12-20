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
# from blade.model.tf_model import TfModel

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
# Download required files
# -----------------------

model_path = "../models/frozen_resnet50_v1.pb" 
img_path = "../images/elephant-299.jpg"
# img_path = download_testdata(image_url, img_name, module="data")
# map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
# label_path = download_testdata(label_map_url, label_map, module="data")

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph_and_image(model, image):
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Prepare image data
    if not tf.gfile.Exists(image):
        tf.logging.fatal("File does not exist %s", image)
    raw_data = tf.gfile.GFile(image, "rb").read()
    tf_data = tf.image.decode_jpeg(raw_data)
    tf_data = tf.image.resize_images(tf_data, (224, 224))
    image_data = tf.keras.preprocessing.image.img_to_array(tf_data)
    image_data = np.expand_dims(image_data, axis = 0)
    return graph_def, image_data

graph_def, image_data = create_graph_and_image(model_path, img_path)

######################################################################
# Tensorflow Benchmark
# ------------------
# Generate Benchmark for tensorflow pb graph inference.

def tf_benchmark(graph_def, image_data, run_count, times_per_run):
    """
    Benchmark the model
    """    
    tf.reset_default_graph()

    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
        softmax_tensor = sess.graph.get_tensor_by_name("resnet_v1_50/predictions/Reshape_1:0")
        # Benchmark! 
        duration_list = np.array([])
        for i in range(run_count):
            time_start = time.time()
            sess.run(softmax_tensor, {"input:0": image_data})
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

# tf_benchmark(graph_def, image_data, 1000, 1)

######################################################################
# Tensorflow+Blade Benchmark
# ------------------
# Generate optimized model by Blade.

def blade_optimize(graph_def, image_data, aggressive_opt):
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
        test_data=[{"input:0": image_data}] # 测试数据
    )
    
    # 打印优化报告
    print(report)
    # 存储优化模型
    # with tf.gfile.FastGFile('frozen_resnet50_v1_opt.pb', mode='wb') as f:
        # f.write(optimized_model.SerializeToString())
    return optimized_model, opt_spec

optimized_model, opt_spec = blade_optimize(graph_def, image_data, False)

with opt_spec:
    tf_benchmark(optimized_model, image_data, 1000, 1)
