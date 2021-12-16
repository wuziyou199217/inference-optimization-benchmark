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
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm, relay
import tvm
from tvm import te
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf

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


try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Base location for model related files.
repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"

# Test image
img_name = "elephant-299.jpg"
image_url = os.path.join(repo_base, img_name)

######################################################################
# Tutorials
# ---------
# Please refer docs/frontend/tensorflow.md for more details for various models
# from tensorflow.

model_name = "frozen_resnet50_v1.pb"
model_url = os.path.join(repo_base, model_name)

# Image label map
map_proto = "imagenet_2012_challenge_label_map_proto.pbtxt"
map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
label_map = "imagenet_synset_to_human_label_map.txt"
label_map_url = os.path.join(repo_base, label_map)

# Target settings
# Use these commented settings to build for cuda.
target = tvm.target.Target("cuda", host="llvm")
# layout = "NCHW"
dev = tvm.cuda(0)
#target = tvm.target.Target("llvm", host="llvm")
layout = None
#dev = tvm.cpu(0)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
from tvm.contrib.download import download_testdata

img_path = download_testdata(image_url, img_name, module="data")
model_path = "./frozen_resnet50_v1.pb" 
map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
label_path = download_testdata(label_map_url, label_map, module="data")

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "resnet_v1_50/predictions/Reshape_1")

######################################################################
# Decode image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#

from PIL import Image

image = Image.open(img_path).resize((224, 224))

x = np.array(image)
x = np.expand_dims(x, axis = 0)
print("x.shape = ", x.shape)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {"input": x.shape}
dtype_dict = {"input": "float32"}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# set inputs
m.set_input("input", tvm.nd.array(x.astype(dtype)))
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1000)), "float32"))

######################################################################
# Process the output
# ------------------
# Process the model output to human readable text for InceptionV1.
predictions = tvm_output.numpy()
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path, uid_lookup_path=label_path)

# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print("%s (score = %.5f)" % (human_string, score))

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    if not tf_compat_v1.gfile.Exists(image):
        tf.logging.fatal("File does not exist %s", image)
    raw_data = tf_compat_v1.gfile.GFile(image, "rb").read()
    tf_data = tf.image.decode_jpeg(raw_data)
    tf_data = tf_compat_v1.image.resize_images(tf_data, (224, 224))
    image_data = tf.keras.preprocessing.image.img_to_array(tf_data)
    image_data = np.expand_dims(image_data, axis = 0)

    # Creates graph from saved GraphDef.
    create_graph()

    with tf_compat_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name("resnet_v1_50/predictions/Reshape_1:0")
        predictions = sess.run(softmax_tensor, {"input:0": image_data})

        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = tf_testing.NodeLookup(
            label_lookup_path=map_proto_path, uid_lookup_path=label_path
        )

        # Print top 5 predictions from tensorflow.
        top_k = predictions.argsort()[-5:][::-1]
        print("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print("%s (score = %.5f)" % (human_string, score))

run_inference_on_image(img_path)

######################################################################
# Benchmark
# ------------------
# Generate Benchmark for built tvm module inference.

def benchmark(compiled_model, target, input_data, run_count, times_per_run):
    """
    Benchmark the model
    """
    dev = tvm.device(target, 0)
    module = graph_executor.GraphModule(compiled_model["default"](dev))

    module.set_input("input", input_data)

    # Evaluate
    return module.benchmark(dev, number=run_count, repeat=times_per_run)

print(benchmark(lib, "cuda", tvm.nd.array(x.astype(dtype)), 1, 1000))

######################################################################
# Tensorflow Benchmark
# ------------------
# Generate Benchmark for tensorflow pb graph inference.

def tf_benchmark(image, run_count, times_per_run):
    """
    Benchmark the model
    """    
    if not tf_compat_v1.gfile.Exists(image):
        tf.logging.fatal("File does not exist %s", image)
    raw_data = tf_compat_v1.gfile.GFile(image, "rb").read()
    tf_data = tf.image.decode_jpeg(raw_data)
    tf_data = tf_compat_v1.image.resize_images(tf_data, (224, 224))
    image_data = tf.keras.preprocessing.image.img_to_array(tf_data)
    image_data = np.expand_dims(image_data, axis = 0)

    # Creates graph from saved GraphDef.
    create_graph()

    with tf_compat_v1.Session() as sess:
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
        pos99 = np.rint(run_count * 0.99)
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

tf_benchmark(img_path, 1000, 1)


