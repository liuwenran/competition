"""
This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all 
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Following my blogpost at:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of classes.
The structure of this script is strongly inspired by the fast.ai Deep Learning
class by Jeremy Howard and Rachel Thomas, especially their vgg16 finetuning
script:  
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same folder: 
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/  

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np
from alexnet import fc,dropout

class TwofcNet(object):
  
  def __init__(self, x, keep_prob, num_classes, skip_layer, 
               weights_path = None):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    self.WEIGHTS_PATH = weights_path
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    
    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(self.X, [-1, 4000])
    fc1 = fc(flattened, 4000, 2048, name='fc1')
    dropout1 = dropout(fc1, self.KEEP_PROB)
    
    # 7th Layer: FC (w ReLu) -> Dropout
    self.fc2 = fc(dropout1, 2048, self.NUM_CLASSES, relu = False, name = 'fc2')
    # dropout7 = dropout(fc7, self.KEEP_PROB)
    
    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    # self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')

    
    
  def load_initial_weights(self, session):
    """
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come 
    as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of 
    dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
    need a special load function
    """
    
    # Load the weights into memory
    if  self.WEIGHTS_PATH == None:
      return

    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
    
    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:
        
      # Check if the layer is one of the layers that should be reinitialized
      if op_name not in self.SKIP_LAYER:
        
        with tf.variable_scope(op_name, reuse = True):
            
          # Loop over list of weights/biases and assign them to their corresponding tf variable
          for data in weights_dict[op_name]:
            
            # Biases
            if len(data.shape) == 1:
              
              var = tf.get_variable('biases', trainable = False)
              session.run(var.assign(data))
              
            # Weights
            else:
              
              var = tf.get_variable('weights', trainable = False)
              session.run(var.assign(data))
            

  
    