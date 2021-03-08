from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Original Images',
      type(train_images), train_images.shape, train_images.dtype)
print('Original Labels',
      type(train_labels), train_labels.shape, train_labels.dtype)
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('Transformed Images',
      type(train_images), train_images.shape, train_images.dtype)
print('Transformed Labels',
      type(train_labels), train_labels.shape, train_labels.dtype)

model = models.Sequential(name="myConvNet")
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Train model')
model.fit(train_images, train_labels, epochs=2, batch_size=128)
print('Test model')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print('Image 0', predictions[0])
print('Expected label',test_labels[0].argmax(), 'Found label', predictions[0].argmax())