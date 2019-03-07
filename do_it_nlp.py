import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
seed = 7
numpy.random.seed(seed)

train = pd.read_csv("C://Users//RA40024262//Downloads//word2vec-nlp-tutorial//labeledTrainData.tsv", sep='\t')
test = pd.read_csv("C://Users//RA40024262//Downloads//word2vec-nlp-tutorial//testData.tsv", sep='\t')

X_train, y_train = train.iloc[:, 2], train.iloc[:,1]
X_test = test.iloc[:, 1]
vocab_size = 10000
encoded_sentiments = [one_hot(d, vocab_size) for d in X_train.values]
padded_sentiments = pad_sequences(encoded_sentiments, maxlen = 5000, padding = 'post')



model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=5000))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(padded_sentiments, y_train, epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(padded_sentiments, y_train, verbose=0)
print(f'binary_crossentropy {scores[0]}, accuracy {scores[1]}')














