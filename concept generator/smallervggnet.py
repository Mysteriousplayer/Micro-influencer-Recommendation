from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as k


class SmallerVGGNet:
	
	def build(width, height, depth, classes, final_act="softmax"):
		# initialize the model along with the input shape to be "channels last" and the channels dimension itself
		model = Sequential()
		input_shape = (height, width, depth)
		channel_dim = -1

		# if we are using "channels first", update the input shape and channels dimension
		if k.image_data_format() == "channels_first":
			input_shape = (depth, height, width)
			channel_dim = 1

		# CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Only set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# final classifier
		model.add(Dense(classes))
		model.add(Activation(final_act))

		# return the constructed network architecture
		return model
