from pathlib import Path
from medpy.io import load
import numpy as np
from sklearn.model_selection import KFold
from Model_3D_AD import getModel
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def normalize_0_1(img):
	x_min = img.min(axis=(0, 1), keepdims=True)
	x_max = img.max(axis=(0, 1), keepdims=True)

	img = (img - x_min)/(x_max-x_min)
	return img


def get_image(img_file):
	img, _ = load(img_file)
	img = normalize_0_1(img)
	img = img[:,:,:47]

	if img.shape[0] != 256:
		img = np.pad(img, ((14,14),(0,0), (0,0)), 'constant')
	img = np.expand_dims(img, axis=-1)

	return img
	

def generator(features, labels, batch_size = 1):
	# Create empty arrays to contain batch of features and labels#
	batch_features = np.zeros((batch_size, 256, 256, 47, 1))
	batch_labels = np.zeros((batch_size,4))
	j = 0

	while True:
		for i in range(batch_size):
			index = j
			if index >= len(features):
				j = 0

			batch_features[i] = get_image(features[index])
			batch_labels[i] = labels[index]
			j += 1

		yield batch_features, batch_labels

ALL = np.array([str(filename) for filename in Path('data/all').glob('*.nii')])
CLASS_ALL = np.array([file_class[file_class.rfind('_') + 1 
				: file_class.index('.nii')] 
					for file_class in ALL]) 

ids = {'Normal': 0, 'AD': 1, 'sMCI': 2, 'pMCI': 3}

CLASS_ALL = np.array([ids[c] for c in CLASS_ALL])
CLASS_ALL = to_categorical(CLASS_ALL)
print(CLASS_ALL[0])

skf = KFold(n_splits=10, shuffle=True, random_state=7)
skf.get_n_splits(ALL)

for train_index, test_index in skf.split(ALL):
	X_train, X_test = ALL[train_index], ALL[test_index]
	y_train, y_test = CLASS_ALL[train_index], CLASS_ALL[test_index]

	break


train_gen = generator(X_train, y_train)
test_gen = generator(X_test, y_test)

model = getModel()

history = model.fit_generator(train_gen,
    epochs=10,
    steps_per_epoch = len(X_train),
    validation_data = test_gen,
    validation_steps = len(X_test),
    verbose = True)


