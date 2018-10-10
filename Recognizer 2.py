from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import cv2
import align.detect_face
import random
import math
import pickle
from sklearn.svm import SVC
from time import sleep

class Recognizer:
    def __init__(self):
        self.load()

    def load(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)
        
        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        self.nrof_successfully_aligned = 0
        self.output_filename = './PhotoCaptured.png'
        self.my_graph = tf.Graph()
        with self.my_graph.as_default():
            # Load the model once
            print('Loading feature extraction model ...')
            
            # Use your path where you have saved pretrained facenet model
            facenet.load_model('../../models/20180402-114759/20180402-114759.pb')
            
            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]

            # your custom classifier trained the last layer with your own image database. Please refer to Facenet repo for training custom classifier
            classifier_filename_exp = os.path.expanduser('../../models/test_classifier.pkl')

            # Classify images
            print('Testing classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % classifier_filename_exp)
    
    def facedetect(self, img, callback):
        if img.ndim<2:
            print('Unable to align "%s"' % img)
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]
        bounding_boxes, box_cord = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Number of faces ******* %s', nrof_faces)
        if(nrof_faces > 0):
            self.predict(img, nrof_faces, bounding_boxes)
        callback()
    
    def predict(self, img, nrof_faces, bounding_boxes):
        face_list = []
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-32/2, 0)
                bb[1] = np.maximum(det[1]-32/2, 0)
                bb[2] = np.minimum(det[2]+32/2, img_size[1])
                bb[3] = np.minimum(det[3]+32/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                self.nrof_successfully_aligned += 1
                filename_base, file_extension = os.path.splitext(self.output_filename)
                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                misc.imsave(output_filename_n, scaled)
                face_list.append(scaled)
        else:
            print('No Image or - Unable to align')
            return None
        
        nrof_images = nrof_faces
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 1000))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        
        with self.my_graph.as_default():
            with tf.Session() as sess:
                for i in range(nrof_batches_per_epoch):
                    start_index = i * 1000
                    end_index = min((i + 1) * 1000, nrof_images)
                    images = self.Face_load_data(face_list, False, False, 160)
                    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                ans = {}
                for i in range(len(best_class_indices)):
                    ans[self.class_names[best_class_indices[i]]] = best_class_probabilities[i]
                    print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))
                return str(ans)
        return  None
    
    def Face_load_data(self, face_list, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        nrof_samples = len(face_list)
        images = np.zeros((nrof_samples, image_size, image_size, 3))
        for i in range(nrof_samples):
            img = face_list[i]
            if img.ndim == 2:
                img = self.to_rgb(img)
            if do_prewhiten:
                img = self.prewhiten(img)
            img = self.crop(img, do_random_crop, image_size)
            img = self.flip(img, do_random_flip)
            images[i,:,:,:] = img
        return images

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y


    def crop(self, image, random_crop, image_size):
        if image.shape[1] > image_size:
            sz1 = int(image.shape[1] // 2)
            sz2 = int(image_size // 2)
            if random_crop:
                diff = sz1 - sz2
                (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
            else:
                (h, v) = (0, 0)
            image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
        return image


    def flip(self, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image


    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret     
