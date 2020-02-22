data_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)

train_it = data_gen.flow_from_directory("png/", target_size=(224, 224), class_mode = "categorical", batch_size = 15000, subset = "training")
test_it = data_gen.flow_from_directory("png/", target_size = (224,224), class_mode = "categorical", batch_size = 5000, subset = "validation")

x_train, y_train = train_it.next()
x_test, y_test = test_it.next()

inds = np.random.permutation(len(x_train))
split_ind = int(len(x_train) * (1 - val_frac))
train_inds, val_inds = inds[:split_ind], inds[split_ind:]
x_val, y_val = x_train[val_inds], y_train[val_inds]
x_train, y_train = x_train[train_inds], y_train[train_inds]

print(f"Train shape:    {x_train.shape}")
print(f"Validation shape: {x_val.shape}")
print(f"Test shape:   {x_test.shape}")

train_feat = vgg.predict(x_train)
test_feat = vgg.predict(x_test)
val_feat = vgg.predict(x_val)


input_shape = vgg.output_shape[1]


model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(250, activation='softmax'))


model.compile(loss='categorical_crossentropy',
  optimizer=optimizers.RMSprop(lr=1e-4),
  metrics=['accuracy'])

model.summary()


checkpoint = keras.callbacks.ModelCheckpoint(
  "pt3_weights.h5",
  monitor="val_acc",
  save_weights_only=True,
  save_best_only=True,
)

history = model.fit(x=train_feat,
  y=y_train,
  validation_data=(val_feat, y_val),
  batch_size=batch_size,
  epochs=500,
  verbose=1,
  callbacks=[checkpoint],)

with open('pt2_vgg16.pickle', 'wb') as handle:
    pickle.dump(history, handle)

model.save('vgg16_pt2.h5')


score = model.evaluate(test_feat, y_test, verbose=1, batch_size=batch_size)
print(f"Test score:    {score[0]: .4f}")
print(f"Test accuracy: {score[1] * 100.:.2f}")



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















