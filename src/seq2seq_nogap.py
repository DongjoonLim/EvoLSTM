
# coding: utf-8

# In[ ]:


from keras.utils.vis_utils import plot_model
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, LSTM, GRU
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
from attention_decoder import AttentionDecoder

nucleotide = ['A','C','G','T','-']
print('Build model...')
#TensorBoard(log_dir='./logs', histogram_freq=1,
#            write_graph=True, write_images=False)

tf.reset_default_graph()

X_train=np.load('prepData/X_train.npy')
X_val=np.load('prepData/X_val.npy')
X_test=np.load('prepData/X_test.npy')
y_train=np.load('prepData/y_train.npy')
y_val=np.load('prepData/y_val.npy')
y_test=np.load('prepData/y_test.npy')

data_dim = 5
hidden_size = len(X_train)
num_classes = 5
sequence_length = time_steps = 20
learning_rate = 0.1

batch_size = 1000#(len(X_train))


def vectorize(char_setx, char_sety, sequence_length):
    dataX = []
    dataY = []
    for i in range(len(char_setx) - sequence_length):
        x = char_setx[i:i + sequence_length]
        y = char_sety[i: i + sequence_length]
        print(i, decoder(x), '->', decoder(y))
        dataX.append(x)
        dataY.append(y)
    return dataX, dataY

model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
model.add(LSTM(64, batch_input_shape=(batch_size,time_steps, data_dim), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
#model2.add(AttentionDecoder(64, 5))

# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(time_steps))
# The decoder RNN could be multiple layers stacked or a single layer
#model2.add(LSTM(64, return_sequences=False))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))

model.add(TimeDistributed(Dense(5)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
model.fit(X_train, y_train, batch_size = batch_size, epochs=20, validation_data=(X_val, y_val), verbose = 1)
model.reset_states()


# In[ ]:


import difflib
def decoder(array):
    result = ""
    size, trash = array.shape
    for i in range(size):
        if list(array[i]) == [1, 0, 0, 0, 0]:
            result=result+"A" 
        elif list(array[i]) == [0, 1, 0, 0, 0]:
            result=result+"C"
        elif list(array[i]) == [0, 0, 1, 0, 0]:
            result=result+"G"
        elif list(array[i]) == [0, 0, 0, 1, 0]:
            result=result+"T"
        elif list(array[i]) == [0, 0, 0, 0, 1]:
            result=result+"-"
    return result

nucleotide = ['A','C','G','T','-']
X_test = X_test[:20000]
#X_test = X_test.reshape(-1,10,5)
print(X_test.shape)
x_true=[]
y_hat_list = []
y_true = []
predictions = model.predict(X_test, batch_size=1000, verbose=0)
for i, prediction in enumerate(predictions):
    # print(prediction)
    #x_index = np.argmax(testX[i], axis=1)
    #print(prediction[i])
    x_str = decoder(X_test[i])

    index = np.argmax(prediction, axis=1)
    result = [nucleotide[j] for j in index]

    print(''.join(x_str), ' -> ', ''.join(result),
          " true: ", ''.join(decoder(y_test[i])))
    x_true.append(''.join(x_str))
    y_hat_list.append(''.join(result))
    y_true.append(''.join(decoder(y_test[i])))
sm=difflib.SequenceMatcher(None,y_hat_list,y_true)
sm2=difflib.SequenceMatcher(None,y_hat_list,x_true)
print(sm.ratio())
print(sm2.ratio())


# In[ ]:




