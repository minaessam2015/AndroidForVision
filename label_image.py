

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def get_ground_truth_label(file_name):
    """
    Maps the predicted index with the original file index
    original file name index         prediction index
    ------------------------------------------------
    #2ch     1          |             2ch      0
    #3ch     2          |             3ch      1
    #4ch     3          |             4ch      2
    #basal   4          |             apical   3
    #mid     5          |             basal    4
    #apical  6          |             mid      5
    ----------------------------------------------------
    Input:
      image basename

    Returns :
      the correct index according to file name

    """
    index = int(file_name.split('_')[2])-1
    data = [0,1,2,4,5,3]
    return data[index]

if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
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
  if args.image:
    file_name = args.image
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
  directory = False
  files = []
  labels = load_labels(label_file)
  failures = open('testing_failures.txt', 'w')
  failures.write(
      "Image Name                          prediction                  GroundTruth                    confidence\n")
  failures.write(
      "---------------------------------------------------------------------------------------------------------\n")
  
  if not file_name.endswith('.jpg'):
	  directory = True
  
  if directory:
  	files = [os.path.join(file_name,f) for f in os.listdir(file_name)]
  else:
        files = [file_name]
  labels = load_labels(label_file)
  print(files)
  counter = 0
  line = ""

  for subdir in files:
      print(subdir)

      for image in os.listdir(subdir):
            
            file_name = os.path.join(subdir,image)
            print(file_name)
            counter+=1
            
            t = read_tensor_from_image_file(
                file_name,
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std)
            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
            results = np.squeeze(results)

            top_index = np.argmax(results)
            line += file_name+ " "+labels[top_index]+" "+labels[get_ground_truth_label(os.path.basename(file_name))]+" "+str(results[top_index])+"\n"
  failures.write(line)
  failures.close()
  print("total images "+str(counter))

  #2ch 1                       2ch      
  #3ch 2                       3ch
  #4ch 3                       4ch
  #basal 4                     apical
  #mid 5                       basal
  #apical 6                    mid

  
  
  
     