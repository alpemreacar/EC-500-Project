# Reference https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch

import tensorflow as tf
import tensorflow.contrib.slim as slim
import copy
import glob
import numpy as np
import os
import scipy
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Discriminator Network

def discriminator(net, scope, dim=64, train=True):
	with tf.variable_scope(scope + '_discriminator', reuse=tf.AUTO_REUSE):
		net = slim.conv2d(net, dim * 1, kernel_size=4, stride=2, activation_fn=None)
		net = tf.nn.leaky_relu(net, alpha=.2)

		net = slim.conv2d(net, dim * 2, kernel_size=4, stride=2, activation_fn=None, biases_initializer=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.leaky_relu(net, alpha=.2)

		net = slim.conv2d(net, dim * 4, kernel_size=4, stride=2, activation_fn=None, biases_initializer=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.leaky_relu(net, alpha=.2)

		net = slim.conv2d(net, dim * 8, kernel_size=4, stride=1, activation_fn=None, biases_initializer=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.leaky_relu(net, alpha=.2)

		net = slim.conv2d(net, 1      , kernel_size=4, stride=1, activation_fn=None)

	return net

# Generator Network

def generator(net, scope, dim=64, train=True):
	def _residual_block(net_i, dim):
		net_o = slim.conv2d(net_i, dim * 1, kernel_size=3, stride=1, activation_fn=None, biases_initializer=None)
		net_o = slim.batch_norm(net_o, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net_o = tf.nn.relu(net_o)
		
		net_o = slim.conv2d(net_o, dim * 1, kernel_size=3, stride=1, activation_fn=None, biases_initializer=None)
		net_o = slim.batch_norm(net_o, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)

		return net_o + net_i
	
	with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
		net = slim.conv2d(net, dim * 1, kernel_size=7, stride=1, activation_fn=None, biases_initializer=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.relu(net)

		net = slim.conv2d(net, dim * 2, kernel_size=3, stride=2, activation_fn=None, biases_initializer=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.relu(net)

		net = slim.conv2d(net, dim * 4, kernel_size=3, stride=2, activation_fn=None, biases_initializer=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.relu(net)

		for i in range(9):
			net = _residual_block(net, dim * 4)

		net = slim.conv2d_transpose(net, dim * 2, kernel_size=3, stride=2, activation_fn=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.relu(net)

		net = slim.conv2d_transpose(net, dim * 1, kernel_size=3, stride=2, activation_fn=None)
		net = slim.batch_norm(net, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None, is_training=train)
		net = tf.nn.relu(net)

		net = slim.conv2d(net, 3, kernel_size=7, stride=1, activation_fn=None)
		net = tf.nn.tanh(net)

	return net

# Image History implementation for discriminator input

class ImageHistory(object):

	def __init__(self, limit=50):
		self.limit = limit
		self.curr_lim = 0
		self.stored_images = []

	def __call__(self, curr_images):
		# Return curr_items if limit is 0
		if self.limit == 0:
			return curr_images
		return_images = []
		for image in curr_images:
			if self.curr_lim < self.limit: # Populate images
				self.stored_images.append(image)
				self.curr_lim += 1
				return_images.append(image)
			else: # Randomly get curr_image or an already stored one
				if np.random.rand() > .5:
					index = np.random.randint(0, self.limit)
					image_copy = copy.copy(self.stored_images[index])
					self.stored_images[index] = image
					return_images.append(image_copy)
				else:
					return_images.append(image)
		return return_images

# Image class

class ImageData:

	def __init__(self, session, image_paths, batch_size, load_size=286, crop_size=256, channels=3):

		self._sess = session
		self._img_batch = ImageData._image_batch(image_paths, batch_size, load_size, crop_size, channels)
		self._img_num = len(image_paths)

	def __len__(self):
		return self._img_num

	def batch(self):
		return self._sess.run(self._img_batch)


    # TODO: Delete this method
	@staticmethod
	def _image_batch(image_paths, batch_size, load_size=286, crop_size=256, channels=3):

		def _parse_func(path):
		    img = tf.read_file(path)
		    img = tf.image.decode_jpeg(img, channels=channels)
		    img = tf.image.random_flip_left_right(img)
		    img = tf.image.resize_images(img, [load_size, load_size])
		    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
		    img = tf.random_crop(img, [crop_size, crop_size, channels])
		    img = img * 2 - 1
		    return img

		dataset = tf.data.Dataset.from_tensor_slices(image_paths)
		dataset = dataset.map(_parse_func, num_parallel_calls=16)
		dataset = dataset.shuffle(buffer_size=4096)
		dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
		dataset = dataset.repeat().prefetch(buffer_size=2)
		
		return dataset.make_one_shot_iterator().get_next()


# Merge images

def img_merge(image_arr, row, col):

	height  = image_arr.shape[1]
	width   = image_arr.shape[2]
	channel = image_arr.shape[3]

	merged_image = np.zeros((height * row, width * col, channel))

	for j in range(col):
		for i in range(row):
			merged_image[i * height: (i+1) * height, j * width: (j+1) * width, :] = \
			image_arr[j + i * col,: ,: ,:]

	return merged_image

# Write images in range (-1., 1.)

def img_write(image, path):

	return scipy.misc.imsave(path, ((image + 1.) / 2. * 255))


