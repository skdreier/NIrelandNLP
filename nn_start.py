








##################################################################
##################################################################
# Deep Neural Networks -- I got pretty lost here, but everything #
# seemed to work until the function
##################################################################
##################################################################

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu')) #relu is a uniform distribution
model.add(layers.Dense(1, activation='sigmoid')) #binary classification

model.compile(loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
    epochs=100,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)

##### Word embeddings

### One hot encoding for each category (we already did this same thing with simpler code):

# List justification categories into categorical integer values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
justification_labels = encoder.fit_transform(df_just['justification_cat'])
justification_labels

# Reshape array
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
justification_labels = justification_labels.reshape((len(justification_labels), 1))
encoder.fit_transform(justification_labels)

# Vectorize a text corpus into a list of integers
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(sentences_train[2])
print(X_train[2])

# choose some words and look at where they show up
for word in ['the', 'internment', 'hmg', 'faulkner', 'belfast']:
    print('{}: {}'.format(word, tokenizer.word_index[word]))

# pad sequence
from keras.preprocessing.sequence import pad_sequences
maxlen = 100 # max sentence length
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

 print(X_train[0, :]) # look at first sentence in corpus

# Keras embedding layer
from keras.models import Sequential
from keras import layers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#vocab_size*embedding_dim

# Train model

history = model.fit(X_train, y_train,
    epochs=20,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history) # plot still not working