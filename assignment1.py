#sarah pell
#assignment 1

import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
from keras_drop_block import DropBlock2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.client import device_lib

def main:

	batch_size = 128
	epochs = 1
	val_frac=0.2

	dir = "png/"

	labels = []
	images = []
	leave = 0

	for img_type in os.listdir(dir):
	    sub_dir = dir + img_type
	    for pic in os.listdir(sub_dir):
	        # leave += 1
	        
	        # if leave < 3:
	        img = image.load_img(sub_dir+'/'+pic)
	        img = image.img_to_array(img)
	        img = img/255
	        print(img_type)
	        images.append(img)
	        labels.append(img_type)
	#         print(images)
	#         print(labels)

	label_encoder = LabelEncoder()
	x = np.array(images)
	y = label_encoder.fit_transform(labels)

	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = .2)


	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(1111, 1111, 3)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='softmax'))

	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer='Adam',
		metrics=['accuracy'],
		)
	model.summary()

	gpu_count = len(available_gpus())
	if gpu_count > 1:
	        print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
	        parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
	else:
	    print("\n\nModel not parallelized over GPUs.\n\n")
	    parallel_model = model

	# parallel_model.compile(
	#     optimizer='adam',
	#     loss='categorical_crossentropy',
	#     metrics=['accuracy'],
	# )
	parallel_model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer='Adam',
		metrics=['accuracy'],
		)

	checkpoint = keras.callbacks.ModelCheckpoint(
	    '../output/weights.h5',
	    monitor='val_acc',
	    save_weights_only=True,
	    save_best_only=True,
	)

	parallel_model.fit(
	    x_train,
	    y_train,
	    batch_size=batch_size,
	    epochs=epochs,
	    verbose=1,
	    callbacks=[checkpoint],
	)

	parallel_model.load_weights('../output/weights.h5')
	score = parallel_model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
	print(f'Test score:    {score[0]: .4f}')
	print(f'Test accuracy: {score[1] * 100.:.2f}')

	preds = parallel_model.predict(x_test, batch_size=batch_size)

	c = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(preds, axis=-1))
	plot_confusion_matrix(
	    c,
	    list(range(10)),
	    normalize=False,
	    output_path='../output',
	)
	plot_confusion_matrix(
	    c,
	    list(range(10)),
	    normalize=True,
	    output_path='../output',
	)

	print(
	    classification_report(
	        np.argmax(y_test, axis=-1),
	        np.argmax(preds, axis=-1),
	        target_names=[str(x) for x in range(10)],
	    )
	)

def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def plot_confusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap='Blues',
        output_path='.',
):
    """
    Logs and plots a confusion matrix, e.g. text and image output.

    Adapted from:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tag = '_norm'
        print("Normalized confusion matrix:")
    else:
        tag = ''
        print('Confusion matrix:')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{output_path}/confusion{tag}.png')
    plt.close()


# model.fit(x_train, y_train, batch_size = 100, epochs=50, validation_data=(x_test, y_test))

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("test accuracy: ", test_acc)

if __name__ == '__main__':
    main(batch_size=2 ** 14, epochs=200)
