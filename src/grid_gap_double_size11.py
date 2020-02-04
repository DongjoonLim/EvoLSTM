
# coding: utf-8

# In[5]:




# In[6]:


from keras.utils.vis_utils import plot_model
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, TimeDistributed, Dense, RepeatVector, CuDNNLSTM, GRU, Bidirectional, Input, CuDNNGRU
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
from keras import backend as K
from keras.models import Model
from keras.layers.core import Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers import concatenate
import difflib
from keras.models import load_model
import keras
from keras import losses
import matplotlib.pyplot as plt
import random
from random import choice
import re
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[7]:

seq_length = 11
os.environ["CUDA_VISIBLE_DEVICES"]="0"
val_loss_hist = []
K.clear_session()
keras.backend.clear_session()

X_train=np.load('prepData/X_train_camFam3_v3_chr2_size11.npy')[:3000000]
X_val=np.load('prepData/X_val_camFam3_v3_chr2_size11.npy')
X_test=np.load('prepData/X_test_camFam3_v3_chr2_size11.npy')
y_train=np.load('prepData/y_train_camFam3_v3_chr2_size11.npy')[:3000000]
y_val=np.load('prepData/y_val_camFam3_v3_chr2_size11.npy')
y_test=np.load('prepData/y_test_camFam3_v3_chr2_size11.npy')

y_train1 = np.load('prepData/y_train1_camFam3_v3_chr2_size11.npy')[:3000000]
y_val1 = np.load('prepData/y_val1_camFam3_v3_chr2_size11.npy')
y_test1 = np.load('prepData/y_test1_camFam3_v3_chr2_size11.npy')

def concat(input1, input2):
    result = []
    for x, y in zip(input1, input2):
        result.append(np.hstack((x, y)))
    
    return np.array(result)

y_train1 = concat(X_train, y_train1)
y_val1 = concat(X_val, y_val1)
y_test1 = concat(X_test, y_test1)
    
# X_train=np.load('prepData/X_train_camFam3_1mut.npy')
# X_val=np.load('prepData/X_val_camFam3_1mut.npy')
# X_test=np.load('prepData/X_test_camFam3_1mut.npy')
# y_train=np.load('prepData/y_train_camFam3_1mut.npy')
# y_val=np.load('prepData/y_val_camFam3_1mut.npy')
# y_test=np.load('prepData/y_test_camFam3_1mut.npy')

# X_train=np.load('prepData20/X_train_camFam3_1mut.npy')
# X_val=np.load('prepData20/X_val_camFam3_1mut.npy')
# X_test=np.load('prepData20/X_test_camFam3_1mut.npy')
# y_train=np.load('prepData20/y_train_camFam3_1mut.npy')
# y_val=np.load('prepData20/y_val_camFam3_1mut.npy')
# y_test=np.load('prepData20/y_test_camFam3_1mut.npy')
nucleotide = ['0', 'A', 'C', 'G', 'T', '-']
#model5 = load_model('model/seq2seq_nogap_camFam3_1mut.h5')
def decoder(array):
    result = ""
    size = len(array)
    for i in range(size):
        if array[i].tolist() == [1, 0, 0, 0, 0, 0]:
            result=result+"0" 
        elif array[i].tolist() == [0, 1, 0, 0, 0, 0]:
            result=result+"A"
        elif array[i].tolist() == [0, 0, 1, 0, 0, 0]:
            result=result+"C"
        elif array[i].tolist() == [0, 0, 0, 1, 0, 0]:
            result=result+"G"
        elif array[i].tolist() == [0, 0, 0, 0, 1, 0]:
            result=result+"T"
        elif array[i].tolist() == [0, 0, 0, 0, 0, 1]:
            result=result+"-"
    return result

#model5 = load_model('model/seq2seq_nogap_camFam3_1mut.h5')
def decoderY(array):
    result = ""
    size = len(array)
    
    if array.tolist() == [1, 0, 0, 0, 0, 0]:
        result=result+"0" 
    elif array.tolist() == [0, 1, 0, 0, 0, 0]:
        result=result+"A"
    elif array.tolist() == [0, 0, 1, 0, 0, 0]:
        result=result+"C"
    elif array.tolist() == [0, 0, 0, 1, 0, 0]:
        result=result+"G"
    elif array.tolist() == [0, 0, 0, 0, 1, 0]:
        result=result+"T"
    elif array.tolist() == [0, 0, 0, 0, 0, 1]:
        result=result+"-"
    return result


def printHitMiss(a,b):
    if a==b:
        return 'Hit'
    else:
        return 'Miss'
    
print(X_test.shape)
def accuracy(a, b):
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            count = count+1
    return count/len(a)

def accuracy2(a, b, c):
    count = 0
    count2 =0
    for i in range(len(a)):
        if a[i] != c[i]:
            count2 = count2 +1
        if a[i] != c[i] and b[i]==c[i]:
            count = count+1
    return count/count2

def isMutation(a, b):
    if a!= b:
        print("mutation")

def predict(model):
    x_true=[]
    y_hat_list = []
    y_true = []
    predictions = model.predict(X_test, batch_size=250, verbose=0)
    
    for i, prediction in enumerate(predictions):
        #print(prediction)
        # print(prediction)
        #x_index = np.argmax(testX[i], axis=1)
        #print(prediction[i])
        x_str = decoder(X_test[i])

        #index = np.argmax(prediction)
        index = np.random.choice(6, 3, p=prediction)
        result = [nucleotide[index]]

        print(''.join(x_str), ' -> ', ''.join(result),
              " true: ", ''.join(decoderY(y_test[i])), printHitMiss(''.join(result), ''.join(decoderY(y_test[i]))))
        x_true.append(''.join(x_str[5]))
        y_hat_list.append(''.join(result))
        y_true.append(''.join(decoderY(y_test[i])))
    sm=difflib.SequenceMatcher(None,y_hat_list,y_true)
    sm2=difflib.SequenceMatcher(None,y_hat_list,x_true)
    print()
    print("Percentage of target and prediction being identical: {}".format(accuracy(y_hat_list, y_true)))
    print("Percentage of training and prediction being identical: {}".format(accuracy(y_hat_list, x_true)))
    print("Accuracy given mutation happened : {}".format(accuracy2(x_true, y_hat_list, y_true)))
    return x_true, y_hat_list, y_true


# In[8]:


def lstm_model(latent_dim, half):
    batch_size = 1000  # Batch size for training.
    epochs = 45  # Number of epochs to train for.
#     latent_dim = 128  # Latent dimensionality of the encoding space.
#     half = 64
    num_samples = 10000  # Number of samples to train on.
    encoder_inputs = Input(shape=(None, 6))
    decoder_inputs = Input(shape=(None, 12))
    
    encoder = CuDNNLSTM(latent_dim, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    _, state_h2, state_c2 = LSTM(latent_dim, return_state=True)(encoder_outputs)
    encoder_states = [state_h, state_c, state_h2, state_c2]


    out_layer1 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    d_outputs, dh1, dc1 = out_layer1(decoder_inputs,initial_state= [state_h, state_c])
    out_layer2 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    final, dh2, dc2 = out_layer2(d_outputs, initial_state= [state_h2, state_c2])
    decoder_dense = Dense(5, activation='softmax')
    decoder_outputs = decoder_dense(final)
    
    # Set up the decoder.
    decoder_inputs = Input(shape=(None, 12))
    decoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_lstm2 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=[state_h, state_c])
    final , _, _= decoder_lstm2(decoder_outputs, initial_state= [state_h2, state_c2])
    decoder_dense = Dense(6, activation='softmax')
    decoder_outputs = decoder_dense(final)

    # Define the model that will turn
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    #inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_input_h1 = Input(shape=(latent_dim,))
    decoder_state_input_c1 = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, 
                             decoder_state_input_h1, decoder_state_input_c1]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs[:2])
    decoder_outputs, state_h1, state_c1 = decoder_lstm2(
        decoder_outputs, initial_state=decoder_states_inputs[-2:])
    decoder_states = [state_h, state_c, state_h1, state_c1]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)


    # Run training
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    return model, encoder_model, decoder_model


def modelFit(epoch, batchSize, latent_dim, half, X_train, y_train, y_train1):
    model1, encoder_model1, decoder_model1 = lstm_model(latent_dim, half)
    hist1 = model1.fit([X_train, y_train1], y_train,
          batch_size=batchSize,
          epochs=epoch,
          validation_data=([X_val,y_val1], y_val),
          verbose = 2
         )
    return hist1, model1, encoder_model1, decoder_model1

def grid_search(latent, half,train_size, X_train, y_train, y_train1):
    hist1, model1, encoder_model1, decoder_model1 = modelFit(1, 100, latent, half, X_train, y_train, y_train1)
    hist2 ,model2, encoder_model2, decoder_model2 = modelFit(2, 100, latent, half, X_train, y_train, y_train1)
    hist3 ,model3, encoder_model3, decoder_model3 = modelFit(10, 100, latent, half, X_train, y_train, y_train1)
    #hist4 ,model4, encoder_model4, decoder_model4 = modelFit(30, 100, latent, half, X_train, y_train, y_train1)
    #hist5 ,model5, encoder_model5, decoder_model5 = modelFit(50, 100, latent, half, X_train, y_train, y_train1)
    #hist6 ,model6, encoder_model6, decoder_model6 = modelFit(80, 100, latent, half, X_train, y_train, y_train1)
    #hist7 ,model7, encoder_model7, decoder_model7 = modelFit(100, 100, latent, half, X_train, y_train, y_train1)
    #hist8 ,model8, encoder_model8, decoder_model8 = modelFit(500, 100, latent, half)

    model1.save("models/gap_{}_{}_1_double.h5".format(train_size,half))
    model2.save("models/gap_{}_{}_2_double.h5".format(train_size,half))
    model3.save("models/gap_{}_{}_10_double.h5".format(train_size,half))
    #model4.save("models/_gap_{}_{}_30_double.h5".format(train_size,half))
    #model5.save("models/_gap_{}_{}_50_double.h5".format(train_size,half))
    #model6.save("models/_gap_{}_{}_80_double.h5".format(train_size,half))
    #model7.save("models/_gap_{}_{}_100_double.h5".format(train_size,half))
    #model8.save("_gap_{}_{}_500.h5".format(train_size,half))
    
    encoder_model1.save("models/Egap_{}_{}_1_double.h5".format(train_size,half))
    encoder_model2.save("models/Egap_{}_{}_2_double.h5".format(train_size,half))
    encoder_model3.save("models/Egap_{}_{}_10_double.h5".format(train_size,half))
    #encoder_model4.save("models/E_gap_{}_{}_30_double.h5".format(train_size,half))
    #encoder_model5.save("models/E_gap_{}_{}_50_double.h5".format(train_size,half))
    #encoder_model6.save("models/E_gap_{}_{}_80_double.h5".format(train_size,half))
    #encoder_model7.save("models/E_gap_{}_{}_100_double.h5".format(train_size,half))
    #encoder_model8.save("E_gap_{}_{}_500.h5".format(train_size,half))
    
    decoder_model1.save("models/Dgap_{}_{}_1_double.h5".format(train_size,half))
    decoder_model2.save("models/Dgap_{}_{}_2_double.h5".format(train_size,half))
    decoder_model3.save("models/Dgap_{}_{}_10_double.h5".format(train_size,half))
    #decoder_model4.save("models/D_gap_{}_{}_30_double.h5".format(train_size,half))
    #decoder_model5.save("models/D_gap_{}_{}_50_double.h5".format(train_size,half))
    #decoder_model6.save("models/D_gap_{}_{}_80_double.h5".format(train_size,half))
    #decoder_model7.save("models/D_gap_{}_{}_100_double.h5".format(train_size,half))
    #decoder_model8.save("D_gap_{}_{}_500.h5".format(train_size,half))
    
    count = [i for i in range(len(hist3.history['val_loss']))]
    val_loss_hist.append([hist3.history['val_loss'].index(min(hist3.history['val_loss'])),min(hist3.history['val_loss'])])
    print(val_loss_hist)
    for i, value in zip(count, hist3.history['val_loss']):
        print(i, value)

grid_search(2, 1, 000, X_train, y_train, y_train1)
# grid_search(16, 8, 000, X_train, y_train, y_train1)
# grid_search(32, 16, 000, X_train, y_train, y_train1)
# grid_search(64, 32, 000, X_train, y_train, y_train1)
# grid_search(128, 64, 000, X_train, y_train, y_train1)
grid_search(256, 128, 000, X_train, y_train, y_train1)
# grid_search(512, 256, 000, X_train, y_train, y_train1)
# grid_search(1024, 512, 000, X_train, y_train, y_train1)
# grid_search(8192, 4096, 000, X_train, y_train, y_train1)



with open('loss_hist_d.txt', 'wb') as fp:
    pickle.dump(val_loss_hist, fp)

