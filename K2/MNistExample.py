from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Orginal train_images',
      type(train_images), train_images.shape, train_images.dtype)
print('Orginal train_labels',
      type(train_labels), train_labels.shape, train_labels.dtype)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('Transformed train_images',
      type(train_images), train_images.shape, train_images.dtype)
print('transformed train_labels',
      type(train_labels), train_labels.shape, train_labels.dtype)

model = models.Sequential()
model.add(layers.Input(shape=(28 * 28,)))
model.add(layers.Dense(512, activation='relu', name='input_layer'))
model.add(layers.Dense(10, activation='softmax', name='output_layer'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print('Image 0', predictions[0])
print('expected',test_labels[0].argmax(), 'found', predictions[0].argmax())