import cnn_vis
import deepdream
import keras.backend as K
from keras.applications import *
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

class ConvNet_Vis():

	#constructor
	def __init__(self, image_path, model=None, name='custom_convnet'):
		if model == None:
			self.select_model()
			self.model_dict = {
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
			self.set_model()
			self.name = self.model_dict[self.model_no]
		else:
			if type(model) == str:
				if os.path.exists(model):
					self.model = load_model(model)
				else:
					raise ValueError("Model path doesn't exist.")
			else:
				self.model = model
			self.name = name

		# choose visulization method
		self.selcet_vis()
		self.image_path = image_path
		self.load_image()
		self.generate_vis()
		
	def select_model(self):
		try:
			print('Choose the number for ConvNet architecture from following list: ')
			print('1. Xception')
			print('2. VGG16')
			print('3. VGG19')
			print('4. ResNet50')
			print('5. InceptionV3')
			print('6. InceptionResNetV2')
			print('7. MobileNet')
			print('8. MobileNetV2')
			print('9. DenseNet121')
			print('10. DenseNet169')
			print('11. DenseNet201')
			print('12. NASNetMobile')
			print('13. NASNetLarge')
			self.model_no = int(input())
		except:
			print('Choose right number :)')
			self.select_model()

	def set_model(self):
		if self.model_dict[self.model_no] == 'Xception':
			K.clear_session()
			self.model = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.xception import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_dict[self.model_no] == 'VGG16':
			K.clear_session()
			self.model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.vgg16 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'VGG19':
			K.clear_session()
			self.model = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.vgg19 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'ResNet50':
			K.clear_session()
			self.model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.resnet50 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'InceptionV3':
			K.clear_session()
			self.model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.inception_v3 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_dict[self.model_no] == 'InceptionResNetV2':
			K.clear_session()
			self.model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_dict[self.model_no] == 'MobileNet':
			K.clear_session()
			self.model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.mobilenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'MobileNetV2':
			K.clear_session()
			self.model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000) 
			from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'DenseNet121':
			K.clear_session()
			self.model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'DenseNet169':
			K.clear_session()
			self.model = densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'DenseNet201':
			K.clear_session()
			self.model = densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'NASNetMobile':
			K.clear_session()
			self.model = nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.nasnet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_dict[self.model_no] == 'NASNetLarge':
			K.clear_session()
			self.model = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.nasnet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (331, 331)

	def selcet_vis(self):
		try:
			print('Choose the number for Visulization method from following list: ')
			print('1. Activation Visualization')
			print('2. Deconvolution Visualization')
			print('3. Deep Dream Visualization')
			self.vis_method_no = int(input())
		except:
			print('Choose right number :)')
			self.selcet_vis()

	def load_image(self):
		img = image.load_img(self.image_path, target_size=self.target_size)
		x = image.img_to_array(img)
		x = self.preprocess_fun(x)
		x = np.expand_dims(x, axis=0)
		self.loaded_image = x

	def generate_vis(self):
		sess = K.get_session()
		layers = ['r', 'p', 'c']

		with sess.as_default():
			if self.vis_method_no == 1:
				is_success = cnn_vis.activation_visualization(sess_graph_path = None,
									value_feed_dict = {self.model.get_layer('input_1').input : self.loaded_image}, 
									layers=layers, path_logdir='./vis/activation/' + self.model_dict[self.model_no] + '/log/', 
									path_outdir='./vis/activation/' + self.model_dict[self.model_no] + '/output/')

			if self.vis_method_no == 2:
				is_success = cnn_vis.deconv_visualization(sess_graph_path = None,
									value_feed_dict = {self.model.get_layer('input_1').input : self.loaded_image}, 
									layers=layers, path_logdir='./vis/deconv/' + self.model_dict[self.model_no] + '/log/', 
									path_outdir='./vis/deconv/' + self.model_dict[self.model_no] + '/output/')

			if self.vis_method_no == 3:
				deepdream.deep_dream(self.model, self.name, self.image_path, self.target_size, self.preprocess_fun, result_prefix = './vis/deepdream/' + self.name + '/output/')

def main():
	ConvNet_Vis('./images/cat.jpg')

if __name__ == "__main__":
	main()
