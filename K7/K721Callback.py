import keras
import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb

class ActivationLogger(keras.callbacks.Callback):

    ### Called by parent model before training, to inform the callback of what model will be calling it.
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs = None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation data')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_' + str(epoch) + '.npz', 'wb')
        np.savez(f, *list(activations[:]))
        f.close()


def vectorize_sequences(sequences, dimension=10000):
    # create feature vectors of size dimension - one-hot encode
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # sequence is an int
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) =\
    imdb.load_data(num_words=10000)

# create feature vectors of size 10000 - one-hot encode
x_train = vectorize_sequences(train_data,10000)
x_test = vectorize_sequences(test_data,10000)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
epochs = 4
model.fit(x_train, y_train, epochs=epochs, batch_size=512,
          callbacks=[ActivationLogger()],
          validation_data= (x_test, y_test))
results = model.evaluate(x_test, y_test)
print(results)

predictions = model.predict(x_test)
print('End')
for epoch in range(epochs):
    with open ('activations_' + str(epoch) + '.npz', 'rb') as f:
        npzfile = np.load(f)
        print('Variables found',npzfile.files)
        for name in npzfile.files:
            print('Variable',name,'Value\n ',npzfile[name])

