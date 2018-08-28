# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import imghdr
import tensorflow as tf
import os.path as path


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def find_img_ex(img_path):
  """
  Always return a string
  """
  return imghdr.what(img_path) or 'jpeg'


def read_tensor_from_image_file(input_height=480,
                                input_width=480,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  file_name = tf.placeholder("string", name="fname")
  file_reader = tf.read_file(file_name, input_name)
  file_type = tf.py_func(find_img_ex, [file_name], "string", False)

  image_reader = tf.case({
      tf.equal(file_type, 'jpeg'): lambda: tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader"),
      tf.equal(file_type, 'png'): lambda: tf.image.decode_png(file_reader, channels=3, name="png_reader"),
      tf.equal(file_type, 'gif'): lambda: tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader")),
      tf.equal(file_type, 'bmp'): lambda: tf.image.decode_bmp(file_reader, channels=3, name="bmp_reader"),
  }, default=lambda: tf.image.decode_image(file_reader, channels=3, name="image_reader"))

  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  return normalized


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 480
  input_width = 480
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"
  result_file = "tf_files/result.csv"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", help="image folder to be processed")
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--result", help="result file")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph

  # Only image_folder or image is used
  if args.image_dir:
    image_files = tf.gfile.Glob(args.image_dir + '*.jpg')
  elif args.image:
    image_files = [args.image]

  if args.result:
    result_file = args.result
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  labels = load_labels(label_file)

  with tf.Session(graph=graph) as sess, open(result_file, 'w') as g:
    g.write('id,predicted\n')
    read_tensor_from_image_file_op = read_tensor_from_image_file(
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std,
    )
    count = 0
    for file_name in image_files:
      print(file_name)
      t = sess.run(read_tensor_from_image_file_op, feed_dict={"fname:0": file_name})
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
      results = np.squeeze(results)
      top_k = results.argsort()[-5:][::-1]
      id = path.splitext(path.basename(file_name))[0]
      predicted = ' '.join(labels[i] for i in top_k[:3])
      g.write("{},{}\n".format(id, predicted))

      count += 1
      if count % 100 == 0:
        print("Predicted {} images".format(count))
