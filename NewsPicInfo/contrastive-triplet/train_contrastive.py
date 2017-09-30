"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. 
Specify the configuration settings at the beginning according to your 
problem.
This script was written for TensorFlow 1.0 and come with a blog post 
you can find here:
  
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert 
contact: f.kratzert(at)gmail.com
"""
from __future__ import division, absolute_import, print_function
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from twofcnet import TwofcNet

# from datagenerator import ImageDataGenerator
from generateData import DataGenerator, VadDataGenerator

"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_im_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/pic_resize_train'
val_im_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/pic_resize_val'
train_vec_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/docFeature_train'
val_vec_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/docFeature_val'
alexnet_weight_path = 'bvlc_alexnet.npy'

# Learning params
learning_rate = 0.001
num_epochs = 10
batch_size = 8
batch_divide = 5

if(batch_size < batch_divide):
  raise Exception("batch_size should be larger than batch_divide.")

# Network params
dropout_rate = 0.5
num_classes = 100
train_layers = ['fc8', 'fc7', 'fc6']
doc_train_layers = ['fc1', 'fc2']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./output/file/"
checkpoint_path = "./output/snapshot/"

# Create parent path if it doesn't exist
try:
  os.makedirs(checkpoint_path)
except OSError:
  pass

# TF placeholder for graph input and output
# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
# y = tf.placeholder(tf.float32, [None, num_classes])
# keep_prob = tf.placeholder(tf.float32)

images = tf.placeholder(tf.float32, [None, 227, 227, 3])
docvecs = tf.placeholder(tf.float32, [None, 200, 20])
issame = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)
margin = 0.1

# Initialize model
# model = AlexNet(x, keep_prob, num_classes, train_layers)

alexmodel = AlexNet(images, keep_prob, num_classes, train_layers,weights_path = alexnet_weight_path)
twofcmodel = TwofcNet(docvecs, keep_prob, num_classes, doc_train_layers)


# Link variable to model output
# score = model.fc8
alexout = alexmodel.fc8
twofcout = twofcmodel.fc2


# List of trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
alex_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
twofc_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in doc_train_layers]
all_var_list = alex_var_list + twofc_var_list

for i in range(len(all_var_list)):
  print(all_var_list[i].name)
# Op for calculating the loss
# with tf.name_scope("cross_ent"):
#   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

with tf.name_scope("contrastive"):
  d = tf.reduce_sum(tf.square(alexout - twofcout), 1)
  d_sqrt = tf.sqrt(d)
  loss = issame * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - issame) * d
  loss = 0.5 * tf.reduce_mean(loss)


# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, all_var_list)
  gradients = list(zip(gradients, all_var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in all_var_list:
  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('contrastive_loss', loss)
  

# Evaluation op: Accuracy of the model
# with tf.name_scope("accuracy"):
#   correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
#   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# Add the accuracy to the summary
# tf.summary.scalar('accuracy', accuracy)

#Merge all summaries together
merged_summary = tf.summary.merge_all()


# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = DataGenerator(train_im_path, train_vec_path, 
                                     horizontal_flip = True, shuffle = True)
val_generator = VadDataGenerator(val_im_path, val_vec_path, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(train_generator.datasize / int(batch_size / batch_divide))
# train_batches_per_epoch = -1
val_batches_per_epoch = val_generator.datasize
# val_batches_per_epoch = 10

# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  # model.load_initial_weights(sess)
  alexmodel.load_initial_weights(sess)
  twofcmodel.load_initial_weights(sess)
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:
            
            # Get a batch of images and labels
            # batch_xs, batch_ys = train_generator.next_batch(batch_size)
            batch_images, batch_vecs, batch_issame = train_generator.next_batch(batch_size, batch_divide)
            
            # And run the training op
            sess.run(train_op, feed_dict={images: batch_images, 
                                          docvecs: batch_vecs, 
                                          issame: batch_issame,
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                res = sess.run(merged_summary, feed_dict={images: batch_images, 
                                                        docvecs: batch_vecs, 
                                                        issame: batch_issame,
                                                        keep_prob: 1.})
                writer.add_summary(res, epoch*train_batches_per_epoch + step)
                
            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        image_feature = np.zeros((val_generator.datasize, num_classes))
        text_feature = np.zeros((val_generator.datasize, num_classes))
        for testind in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch()
            feature_out = sess.run([alexout, twofcout], feed_dict={images: batch_tx, 
                                                docvecs: batch_ty, 
                                                keep_prob: 1.})
            image_feature[testind, :] = feature_out[0]
            text_feature[testind, :] = feature_out[1]

        # test_acc /= test_count
        # print("Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
        hittop10_num = 0
        for i in range(val_batches_per_epoch):
            cur_imfeature = image_feature[i,:]
            cur_disvec = []
            for j in range(val_batches_per_epoch):
              cur_textfeature = text_feature[j,:]
              cur_dis = np.square(cur_imfeature - cur_textfeature)
              cur_dis = np.sum(cur_dis)
              cur_disvec.append(cur_dis)

            sortind = np.argsort(cur_disvec)
            if i in sortind[:10]:
              hittop10_num += 1
        print("hit top 10 num: {} in all {} val samples".format(hittop10_num, val_batches_per_epoch))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        
