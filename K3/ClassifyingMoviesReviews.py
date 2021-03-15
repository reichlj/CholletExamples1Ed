import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import imdb

def vectorize_sequences(sequences, dimension=10000):
    # create feature vectors of size dimension - one-hot encode
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # sequence is an int
        results[i, sequence] = 1.
    return results

# tf.keras.datasets.imdb.load_data(
#     path="imdb.npz",# where to cache the data (relative to ~/.keras/dataset)
#     num_words=None, # Words are ranked by how often they occur (in the training set) and only
#                     # the num_words most frequent words are kept. Any less frequent word will
#                     # appear as oov_char value in the sequence data.
#     skip_top=0, # skip the top N most frequently occurring words (which may not be informative)
#                 # These words will appear as oov_char value in the dataset. Defaults to 0, so no words are skipped.
#     maxlen=None,# Maximum sequence length
#     seed=113,
#     start_char=1, # The start of a sequence will be marked with this character. Defaults to 1
#     oov_char=2,   # The out-of-vocabulary character.
#     index_from=3, # Index actual words with this index and higher.
#     **kwargs
# )
(train_data, train_labels), (test_data, test_labels) =\
    imdb.load_data(num_words=10000)

# word_index            crime  ->  17       word -> int
# reverse_word_index    17     ->  crime    int -> word
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# create feature vectors of size 10000 - one-hot encode
x_train = vectorize_sequences(train_data,10000)
x_test = vectorize_sequences(test_data,10000)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',
              metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train,
                    epochs=10,batch_size=512,validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)

predictions = model.predict(x_test)

pass