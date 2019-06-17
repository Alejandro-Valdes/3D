from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from keras import layers
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical

from keras import backend as K
import tensorflow as tf

import numpy as np


def getModel():
	img_input = layers.Input(shape=(256,256,47,1))
	x = layers.Dropout(0.1)(img_input)
	x = layers.Conv3D(8,(7,7,5), activation='relu')(x)

	x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
	x = layers.Conv3D(16, (5,5,3), activation='relu')(x)

	x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
	x = layers.Conv3D(16, (5,5,3), activation='relu')(x)

	x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
	x = layers.Conv3D(32, (5,5,3), activation='relu')(x)
	x = layers.Conv3D(32, (3), activation='relu')(x)

	x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
	x = layers.Conv3D(64, (3,3,1), activation='relu')(x)
	x = layers.Conv3D(64, (3,3,1), activation='relu')(x)

	x = layers.Dropout(0.5)(x)
	x = layers.Flatten()(x)

	x = layers.Dense(512, activation='relu')(x)

	x = layers.Dropout(0.5)(x)
	x = layers.Dense(4, activation='softmax')(x)

	model = Model(input=img_input, output=x)

	model.compile(loss=categorical_crossentropy,
	              optimizer=optimizers.RMSprop(lr=1e-4),
	              metrics=['accuracy'])

	print(model.summary())
	return model

