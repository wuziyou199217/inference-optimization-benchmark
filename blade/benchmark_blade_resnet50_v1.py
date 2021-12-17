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
import numpy as np
import os.path

# Tensorflow imports
import tensorflow.compat.v1 as tf

# Blade
import blade
from blade.model.tf_model import TfModel

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

# Base location for model related files.
# repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"

######################################################################
# Tutorials
# ---------
# Please refer docs/frontend/tensorflow.md for more details for various models
# from tensorflow.

# model_name = "frozen_resnet50_v1.pb"
# model_url = os.path.join(repo_base, model_name)

# Image label map
# map_proto = "imagenet_2012_challenge_label_map_proto.pbtxt"
# map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
# label_map = "imagenet_synset_to_human_label_map.txt"
# label_map_url = os.path.join(repo_base, label_map)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
# from tvm.contrib.download import download_testdata

model_path = "../models/frozen_resnet50_v1.pb" 
model_opt_path = "frozen_resnet50_v1_opt.pb" 
img_path = "../images/elephant-299.jpg"
# img_path = download_testdata(image_url, img_name, module="data")
# map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
# label_path = download_testdata(label_map_url, label_map, module="data")

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph(model):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")

######################################################################
# Benchmark
# ------------------
# Generate Benchmark for built tvm module inference.

# def benchmark(compiled_model, target, input_data, run_count, times_per_run):
#     """
#     Benchmark the model
#     """
#     dev = tvm.device(target, 0)
#     module = graph_executor.GraphModule(compiled_model["default"](dev))
# 
#     module.set_input("input", input_data)
# 
#     # Evaluate
#     return module.benchmark(dev, number=run_count, repeat=times_per_run)

# print(benchmark(lib, "cuda", tvm.nd.array(x.astype(dtype)), 1, 1000))

######################################################################
# Tensorflow Benchmark
# ------------------
# Generate Benchmark for tensorflow pb graph inference.

def tf_benchmark(model, image, run_count, times_per_run):
    """
    Benchmark the model
    """    
    # Creates graph from saved GraphDef.
    create_graph(model)

    # Prepare image data
    if not tf.gfile.Exists(image):
        tf.logging.fatal("File does not exist %s", image)
    raw_data = tf.gfile.GFile(image, "rb").read()
    tf_data = tf.image.decode_jpeg(raw_data)
    tf_data = tf.image.resize_images(tf_data, (224, 224))
    image_data = tf.keras.preprocessing.image.img_to_array(tf_data)
    image_data = np.expand_dims(image_data, axis = 0)

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name("resnet_v1_50/predictions/Reshape_1:0")
        
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

# tf_benchmark(model_path, img_path, 1000, 1)
tf_benchmark(model_opt_path, img_path, 1000, 1)

######################################################################
# Tensorflow+Blade Benchmark
# ------------------
# Generate optimized model by Blade.

def blade_optimize():
    """
    Optimize the model by Blade
    """    
    saved_model_dir = '../models/'

    # 零输入
    optimized_model, _, report = blade.optimize(
        saved_model_dir,       # 模型路径。
        'o1',                  # O1无损优化。或O2有损优化
        device_type='gpu'      # 面向GPU设备优化。
    )
    
    # +测试/校正数据集
    # optimized_model, _, report = blade.optimize(
    #     saved_model_dir,       # 模型路径。
    #     'o0',                  # O0无损优化。
    #     device_type='gpu',     # 面向GPU设备优化。
    #     test_data=[test_data]  # 测试数据。
    # )
    
    # 打印优化报告
    print(report)
    # 存储优化模型
    with tf.gfile.FastGFile('frozen_resnet50_v1_opt.pb', mode='wb') as f:
        f.write(optimized_model.SerializeToString())
    return

# blade_optimize()
