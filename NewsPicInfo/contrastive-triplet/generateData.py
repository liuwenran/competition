from __future__ import division, absolute_import, print_function
import numpy as np
import cv2
import os

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class DataGenerator:
    def __init__(self, im_rootpath, docvec_rootpath, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 veclength = 200,vecperdoc = 20, vec_mean = 0.5, batch_divide = 5, 
                 nb_classes = 2):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        self.veclength = veclength
        self.vecperdoc = vecperdoc
        self.vec_mean = vec_mean
        
        self.get_im_docvec_list(im_rootpath, docvec_rootpath)
        
        if self.shuffle:
            self.shuffle_data()

    def get_im_docvec_list(self, im_rootpath, docvec_rootpath):
        # unvalidlist = np.load(unvalidim)
        imfilelist = os.listdir(im_rootpath)
        self.absImPath = []
        self.absDocvecPath = []
        for i in range(len(imfilelist)):
            dotind = imfilelist[i].rfind('.')
            curfile = imfilelist[i][:dotind]
            self.absImPath.append(os.path.join(im_rootpath,imfilelist[i]))
            self.absDocvecPath.append(os.path.join(docvec_rootpath,curfile + '.npy'))
        self.datasize = len(imfilelist)


    def shuffle_data(self):
        images = self.absImPath
        docvec = self.absDocvecPath
        self.absImPath = []
        self.absDocvecPath = []

        idx = np.random.permutation(len(docvec))
        for i in idx:
            self.absImPath.append(images[i])
            self.absDocvecPath.append(docvec[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        

    def next_batch(self, batch_size, batch_divide):
        if batch_size%batch_divide != 0:
            print('warning by lwr: batch_size should be times of batch_divide, you will get a smaller batch_size.')
        sametuple_size = int(batch_size / batch_divide)
        true_batch_size = sametuple_size * batch_divide

        paths = self.absImPath[self.pointer:self.pointer + sametuple_size]
        vecs = self.absDocvecPath[self.pointer:self.pointer + sametuple_size]

        self.pointer += sametuple_size

        images = np.ndarray([true_batch_size, self.scale_size[0], self.scale_size[1], 3])
        docvec = np.ndarray([true_batch_size, self.veclength, self.vecperdoc])
        issame = np.ndarray([true_batch_size, 1])
        for i in range(sametuple_size):
            img = cv2.imread(paths[i])

            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img,1)
            img = img.astype(np.float32)

            img -= self.mean
            images[i] = img

            vec = np.load(vecs[i])
            vec -= self.vec_mean
            docvec[i] = vec

            issame[i] = 0

            for j in range(1, batch_divide):
                images[i + j * sametuple_size] = img

                randtmp = np.random.randint(len(self.absDocvecPath))
                vec = np.load(self.absDocvecPath[(self.pointer + i + randtmp - 1) % len(self.absDocvecPath)])
                vec -= self.vec_mean
                docvec[i + j * sametuple_size] = vec

                issame[i + j * sametuple_size] = 1

        return images, docvec, issame

                
class VadDataGenerator:
    def __init__(self, im_rootpath, docvec_rootpath, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 veclength = 200,vecperdoc = 20, vec_mean = 0.5, 
                 nb_classes = 2):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        self.veclength = veclength
        self.vecperdoc = vecperdoc
        self.vec_mean = vec_mean
        
        self.get_im_docvec_list(im_rootpath, docvec_rootpath)
        
        if self.shuffle:
            self.shuffle_data()

    def get_im_docvec_list(self, im_rootpath, docvec_rootpath):
        # unvalidlist = np.load(unvalidim)
        imfilelist = os.listdir(im_rootpath)
        self.absImPath = []
        self.absDocvecPath = []
        for i in range(len(imfilelist)):
            dotind = imfilelist[i].rfind('.')
            curfile = imfilelist[i][:dotind]
            self.absImPath.append(os.path.join(im_rootpath,imfilelist[i]))
            self.absDocvecPath.append(os.path.join(docvec_rootpath,curfile + '.npy'))
        self.datasize = len(imfilelist)

    def shuffle_data(self):
        images = self.absImPath
        docvec = self.absDocvecPath
        self.absImPath = []
        self.absDocvecPath = []

        idx = np.random.permutation(len(docvec))
        for i in idx:
            self.absImPath.append(images[i])
            self.absDocvecPath.append(docvec[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        

    def next_batch(self):

        paths = self.absImPath[self.pointer]
        vecs = self.absDocvecPath[self.pointer]

        self.pointer += 1

        images = np.ndarray([1, self.scale_size[0], self.scale_size[1], 3])
        docvec = np.ndarray([1, self.veclength, self.vecperdoc])

        img = cv2.imread(paths)

        if self.horizontal_flip and np.random.random() < 0.5:
            img = cv2.flip(img,1)
        img = img.astype(np.float32)

        img -= self.mean
        images[0] = img

        vec = np.load(vecs)
        vec -= self.vec_mean
        docvec[0] = vec

        return images, docvec



