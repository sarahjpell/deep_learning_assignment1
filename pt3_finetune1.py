from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import vgg16
from keras.models import Model
import keras
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers



vgg = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

# vgg.trainable = False
# for layer in vgg.layers:
#     layer.trainable = False
set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
	layer.trainable = False


vgg.summary()


train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory("png/", target_size=(224, 224), class_mode = "categorical", batch_size = 15000, subset = "training")
val_generator = val_datagen.flow_from_directory("png/", target_size = (224,224), class_mode = "categorical", batch_size = 5000, subset = "validation")

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=(224,224,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(250, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])



checkpoint = keras.callbacks.ModelCheckpoint("pt3_weights.h5",monitor="val_acc",save_weights_only=True,save_best_only=True,)

history = model.fit_generator(train_generator, steps_per_epoch=2, epochs=30,validation_data=val_generator, validation_steps=50, verbose=1, callbacks=[checkpoint])



model.save('vgg16_finetune_pt3.h5')



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

# plt.figure()
plt.savefig('pt3_accplot.png')
plt.clf()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# plt.show()
plt.savefig('pt3_lossplot.png')









