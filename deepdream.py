from __future__ import print_function

from keras.preprocessing import image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras import backend as K

import numpy as np
import scipy
import argparse
import os

model_dict = {
				1: 'Xception',
				2: 'VGG16',
				3: 'VGG19',
				4: 'ResNet50',
				5: 'InceptionV3',
				6: 'InceptionResNetV2',
				7: 'MobileNet',
				8: 'MobileNetV2',
				9: 'DenseNet121',
				10: 'DenseNet169',
				11: 'DenseNet201',
				12: 'NASNetMobile',
				13: 'NASNetLarge'
			 }

default_settings = {
	'Xception':{
		'add_13': 0.33,
		'add_14': 0.67,
		'add_15': 1.0,
		'add_16': 1.33,
		'add_17': 1.67,
		'add_18': 2.0,
		'add_19': 1.67,
		'add_20': 1.33,
		'add_21': 1.0,
		'add_22': 0.67,
		'add_23': 0.33,
		'add_24': 0.2,
		'avg_pool': 0.1
	},

	'VGG16':{
		'block1_pool': 0.67,
		'block2_pool': 1.33,
		'block3_pool': 2.0,
		'block4_pool': 1.5,
		'block5_pool': 1.0
	},

	'VGG19':{
		'block1_pool': 0.67,
		'block2_pool': 1.33,
		'block3_pool': 2.0,
		'block4_pool': 1.5,
		'block5_pool': 1.0
	},

	'ResNet50': {
		'add_1': 0.25,
		'add_2': 0.5,
		'add_3': 0.75,
		'add_4': 1.0,
		'add_5': 1.25,
		'add_6': 1.75,
		'add_7': 2.0,
		'add_8': 1.85,
		'add_9': 1.75,
		'add_10': 1.5,
		'add_11': 1.25,
		'add_12': 1.0,
		'add_13': 0.75,
		'add_14': 0.5,
		'add_15': 0.25,
		'add_16': 0.1
	},

	'InceptionV3': {
		'mixed0': 0.33,
		'mixed1': 0.67,
		'mixed2': 1.0,
		'mixed3': 1.33,
		'mixed4': 1.67,
		'mixed5': 2.0,
		'mixed6': 1.67,
		'mixed7': 1.33,
		'mixed8': 1.0,
		'mixed9_0': 0.67,
		'mixed9': 0.33,
		'mixed9_1': 0.2,
		'mixed10': 0.1
	},

	'InceptionResNetV2': {
		'mixed_5b': 0.2,
		'mixed_6a': 0.5,
		'mixed_7a': 2.0,
		'avg_pool': 1.5
	},
	 
	'MobileNet': {
		'conv_pw_2_relu': 0.33,
		'conv_pw_3_relu': 0.67,
		'conv_pw_4_relu': 1.0,
		'conv_pw_5_relu': 1.33,
		'conv_pw_6_relu': 1.67,
		'conv_pw_7_relu': 2.0,
		'conv_pw_8_relu': 1.67,
		'conv_pw_9_relu': 1.5,
		'conv_pw_10_relu': 1.33,
		'conv_pw_11_relu': 1.0,
		'conv_pw_12_relu': 0.67,
		'conv_pw_13_relu': 0.33
	},

	'MobileNetV2': {
		'block_2_add': 0.4,
		'block_4_add': 0.8,
		'block_5_add': 1.2,
		'block_7_add': 1.6,
		'block_8_add': 1.8,
		'block_9_add': 2.0,
		'block_11_add': 1.8,
		'block_12_add': 1.6,
		'block_14_add': 1.4,
		'block_15_add': 0.2
	},

	'DenseNet121': {
		'pool2_pool': 0.2,
		'pool3_pool': 0.5,
		'pool4_pool': 2.0,
		'relu': 1.5
	},

	'DenseNet169': {
		'pool2_pool': 0.2,
		'pool3_pool': 0.5,
		'pool4_pool': 2.0,
		'relu': 1.5
	},

	'DenseNet201': {
		'pool1': 0.2,
		'pool2_pool': 0.5,
		'pool3_pool': 1.5,
		'pool4_pool': 2.0,
		'relu': 1.5
	},

	'NASNetMobile': {
		'add_1': 0.2,
		'add_2': 0.5,
		'add_3': 1.5,
		'add_4': 2.0,
		'global_average_pooling2d_1': 0.3
	},

	'NASNetLarge': {
		'concatenate_1': 0.2,
		'concatenate_2': 0.5,
		'concatenate_3': 2.0,
		'concatenate_4': 1.5,
		'normal_concat_18': 0.2
	}
}

def take_layer_no_input(note):
	try:
		layer_no = input(note)
		numbers = []
		for i in layer_no.split(','):
			numbers.append(int(i.strip()))
		return numbers
	except:
		print('Give numbers in comma seperated format')
		take_layer_no_input(note)

def take_layer_proportion_input(note):
	try:
		proportion = input(note)
		numbers = []
		for i in proportion.split(','):
			numbers.append(float(i.strip()))
		return numbers
	except:
		print('Give numbers in comma seperated format: ')
		take_layer_proportion_input(note)

def take_y_n(note):
	try:
		ans = input(note).lower()
		if ans == 'y':
			return True
		elif ans == 'n':
			return False
		else:
			take_y_n(note)
	except:
		take_y_n(note)

def select_layer_tweak_setting(model, model_name):
	global model_dict
	global default_settings
	settings = {}
	settings['features'] = {}

	print('', '#'*120, '\n', "Select the contribution of the layers for which we try to maximize activation, as well as their weight in the final loss.", '\n', "You can tweak these setting to obtain new visual effects.", '\n', '#'*120)

	want_default = take_y_n("Do you want to use DEFAULT settings? (Y/N): ")

	if want_default and model_name in model_dict.values():
		settings['features'] = default_settings[model_name]
	else:
		for i, layer in enumerate(model.layers):
			print(i+1, layer.name)

		layer_no = take_layer_no_input('#'*120 + '\n' + "Choose the layer number for which you want to maximize the activation. (Comma Sepearted--> ex: 10, 25, 54, 60)" + '\n' + '#'*120)
		numbers = take_layer_proportion_input('#'*120 + '\n' + "Choose the proportion for the layers which you have choosed to maximize the activation. (Comma Sepearted--> ex: 0.2, 0.5, 2.0, 1.5)" + '\n' + '#'*120)

		for i, j in zip(layer_no, numbers):
			settings['features'][model.layers[i - 1].name] = j

	return settings

def preprocess_image(image_path, target_size, preprocess_fun):
	img = image.load_img(image_path)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_fun(x)
	return x	

def deprocess_image(x):
	# Util function to convert a tensor into a valid image.
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, x.shape[2], x.shape[3]))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((x.shape[1], x.shape[2], 3))
	x /= 2.
	x += 0.5
	x *= 255.
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def define_loss_fun(model, settings):
	dream = model.input

	# Get the symbolic outputs of each "key" layer (we gave them unique names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	# Define the loss.
	loss = K.variable(0.)
	for layer_name in settings['features']:
		# Add the L2 norm of the features of a layer to the loss.
		if layer_name not in layer_dict:
			raise ValueError('Layer ' + layer_name + ' not found in model.')
		coeff = settings['features'][layer_name]
		x = layer_dict[layer_name].output
		# We avoid border artifacts by only involving non-border pixels in the loss.
		scaling = K.prod(K.cast(K.shape(x), 'float32'))
		if K.image_data_format() == 'channels_first':
			loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
		else:
			loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

	# Compute the gradients of the dream wrt the loss.
	grads = K.gradients(loss, dream)[0]
	# Normalize gradients.
	grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

	# Set up function to retrieve the value
	# of the loss and gradients given an input image.
	outputs = [loss, grads]
	fetch_loss_and_grads = K.function([dream], outputs)
	return fetch_loss_and_grads

def eval_loss_and_grads(x, fetch_loss_and_grads):
	outs = fetch_loss_and_grads([x])
	loss_value = outs[0]
	grad_values = outs[1]
	return loss_value, grad_values

def resize_img(img, size):
	img = np.copy(img)
	if K.image_data_format() == 'channels_first':
		factors = (1, 1, float(size[0]) / img.shape[2], float(size[1]) / img.shape[3])
	else:
		factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
	return scipy.ndimage.zoom(img, factors, order=1)

def gradient_ascent(x, fetch_loss_and_grads, iterations, step, max_loss=None):
	for i in range(iterations):
		loss_value, grad_values = eval_loss_and_grads(x, fetch_loss_and_grads)
		if max_loss is not None and loss_value > max_loss:
			break
		print('..Loss value at', i, ':', loss_value)
		x += step * grad_values
	return x

def hyperparameters():
	hyperparameters_value = take_y_n("Do you want to use default hyperparameters? (Y/N): ")
	if hyperparameters_value:
		step = 0.01
		num_octave = 3
		octave_scale = 1.4
		iterations = 20
		max_loss = 10.
		return step, num_octave, octave_scale, iterations, max_loss
	else:
		step = input("Gradient ascent learning rate (DEFAULT= 0.01): ")
		num_octave = input("Number of scales at which to run gradient ascent (DEFAULT= 3): ")
		octave_scale = input("Size ratio between scales (DEFAULT= 1.4): ")
		iterations = input("Number of ascent steps per scale (DEFAULT= 20): ")
		max_loss = input("Maximum loss allowed (DEFAULT= 10.0): ")
		return float(step), int(num_octave), float(octave_scale), int(iterations), float(max_loss)

def run_gradient_ascent(image_path, target_size, preprocess_fun, fetch_loss_and_grads, result_prefix):
	step, num_octave, octave_scale, iterations, max_loss = hyperparameters()

	img = preprocess_image(image_path, target_size, preprocess_fun)
	if K.image_data_format() == 'channels_first':
		original_shape = img.shape[2:]
	else:
		original_shape = img.shape[1:3]
	successive_shapes = [original_shape]
	for i in range(1, num_octave):
		shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
		successive_shapes.append(shape)
	successive_shapes = successive_shapes[::-1]
	original_img = np.copy(img)
	shrunk_original_img = resize_img(img, successive_shapes[0])

	for shape in successive_shapes:
		print('Processing image shape', shape)
		img = resize_img(img, shape)
		img = gradient_ascent(img, fetch_loss_and_grads, iterations=iterations, step=step, max_loss=max_loss)
		upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
		same_size_original = resize_img(original_img, shape)
		lost_detail = same_size_original - upscaled_shrunk_original_img

		img += lost_detail
		shrunk_original_img = resize_img(original_img, shape)

	if not os.path.exists(result_prefix):
		os.makedirs(result_prefix)
	save_img(result_prefix + 'dd_image.png', deprocess_image(np.copy(img)))

def deep_dream(model, model_name, image_path, target_size, preprocess_fun, result_prefix):
	settings = select_layer_tweak_setting(model, model_name)
	K.set_learning_phase(0)
	fetch_loss_and_grads = define_loss_fun(model, settings)
	run_gradient_ascent(image_path, target_size, preprocess_fun, fetch_loss_and_grads, result_prefix)

if __name__ == "__main__":
	from keras.applications import inception_v3
	from keras.applications.inception_v3 import preprocess_input, decode_predictions
	model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
	model_name = 'InceptionV3'
	preprocess_fun = preprocess_input
	decode_fun = decode_predictions
	target_size = (299, 299)
	result_prefix = './vis/deepdream/' + model_name + '/output/'
	deep_dream(model, model_name, 'images/cat.jpg', target_size, preprocess_fun, result_prefix)
