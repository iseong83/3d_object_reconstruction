import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ExifTags

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

model_dict = {'mobile_coco':'mobilenetv2_coco_voctrainval',
        'xception_coco': 'xception_coco_voctrainaug',
        'mobile_ade': 'mobilenetv2_ade20k_train',
        'exception_ade': 'xception65_ade20k_train'}
# @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    'mobilenetv2_ade20k_train':
        'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz',
    'xception65_ade20k_train':
        'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
}

def load_image(fpath):
    try:
        image = Image.open(fpath)
        try:
            for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : break 
            exif=dict(image._getexif().items())

            if   exif[orientation] == 3 : 
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6 : 
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8 : 
                image=image.rotate(90, expand=True)
        except AttributeError:
            pass

        return image
    except IOError:
        print ("Cannot open the image file:", fpath)
        return

def run_visualization_file(fpath, model):
    """Inferences DeepLab model and visualizes result."""
    print('running deeplab on image %s...' % fpath)
    image = load_image(fpath)
    resized_im, seg_map = model.run(image)
    
    vis_segmentation(resized_im, seg_map)

def run_visualization(url, model):
  """Inferences DeepLab model and visualizes result."""
  """  
    SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
    IMAGE_URL = 'https://www.schoolsin.com/Merchant5/graphics/00000001/chair-9010-blu51-chrm_540x540.jpg'  #@param {type:"string"}
    _SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
                   'deeplab/g3doc/img/%s.jpg?raw=true')
    image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
    run_visualization(image_url)
  """

  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = model.run(original_im)

  vis_segmentation(resized_im, seg_map)


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  image = np.array(image)
  mask = np.repeat(seg_map[:,:,np.newaxis],3,axis=2)
  #object = np.where(mask>0, image, 255)
  object = np.where(mask==9, image, 255)

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #plt.imshow(seg_image)
  plt.imshow(object)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, model_name):
    """Creates and loads pretrained deeplab model."""
    self._get_model(model_name)
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(self.download_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.
    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

  def _get_model(self, model_name):
      #model_dir = tempfile.mkdtemp()
      model_dir = './models_deeplab'
      tf.gfile.MakeDirs(model_dir)
      _TARBALL_NAME = _MODEL_URLS[model_dict[model_name]]

      self.download_path = os.path.join(model_dir, _TARBALL_NAME)
      if not os.path.isfile(self.download_path):
          print('downloading model, this might take a while...')
          urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[model_dict[model_name]], self.download_path)
          print('download completed! loading DeepLab model...')

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


