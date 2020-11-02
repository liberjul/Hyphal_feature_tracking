import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse, os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="input file to predict objects")
parser.add_argument("-b", "--labels", type=str, default= "./annotations/label_map.pbtxt", help="path to label map")
parser.add_argument("-c", "--ckpt", type=str, default="./training/frozen_inference_graph_v4.pb", help="path to checkpoint inference graph")
parser.add_argument("-t", "--threshold", type=float, default=0.30, help="confidence threshold for annotations, betweem 0 and 1")
args = parser.parse_args()

PATH_TO_LABELS = args.labels
PATH_TO_CKPT = args.ckpt
CONF_THR = args.threshold
NUM_CLASSES = 1


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def save_image(data, filename):
  sizes = np.shape(data)
  fig = plt.figure(figsize=(1,1))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(data, cmap = plt.get_cmap("bone"))
  plt.savefig(filename,dpi = 1200)
  plt.close()

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_np = plt.imread(args.file).copy()
        if len(image_np.shape) == 2 or image_np.shape[2] == 1:
            print("Converting grayscale image ...")
            image_np = np.stack((image_np,)*3, axis=-1)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Extract image tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Extract detection boxes
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detectionsd
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                        max_boxes_to_draw=100,
                        min_score_thresh=CONF_THR)
        save_image(image_np, args.file.split(".")[0] + "_annot.jpg")
