
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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[7]:


os.environ["CUDA_VISIBLE_DEVICES"]="0";
val_loss_hist = []
K.clear_session()
keras.backend.clear_session()

X_train=np.load('prepData/X_train_camFam3_1mutOnly_v3_chr2.npy')[:12000]
X_val=np.load('prepData/X_val_camFam3_1mutOnly_v3_chr2.npy')[:10000]
X_test=np.load('prepData/X_test_camFam3_1mutOnly_v3_chr2.npy')[:4000]
y_train=np.load('prepData/y_train_camFam3_1mutOnly_v3_chr2.npy')[:12000]
y_val=np.load('prepData/y_val_camFam3_1mutOnly_v3_chr2.npy')[:10000]
y_test=np.load('prepData/y_test_camFam3_1mutOnly_v3_chr2.npy')[:4000]

y_train1 = np.load('prepData/y_train1_camFam3_1mutOnly_v3_chr2.npy')[:12000]
y_val1 = np.load('prepData/y_val1_camFam3_1mutOnly_v3_chr2.npy')[:10000]
y_test1 = np.load('prepData/y_test1_camFam3_1mutOnly_v3_chr2.npy')[:4000]

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

# X_train=np.load('prepData20/X_train_camFam3_1mutOnly.npy')
# X_val=np.load('prepData20/X_val_camFam3_1mutOnly.npy')
# X_test=np.load('prepData20/X_test_camFam3_1mutOnly.npy')
# y_train=np.load('prepData20/y_train_camFam3_1mutOnly.npy')
# y_val=np.load('prepData20/y_val_camFam3_1mutOnly.npy')
# y_test=np.load('prepData20/y_test_camFam3_1mutOnly.npy')
nucleotide = ['0', 'A', 'C', 'G', 'T', '-']
#model5 = load_model('model/seq2seq_nogap_camFam3_1mutOnly.h5')
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

#model5 = load_model('model/seq2seq_nogap_camFam3_1mutOnly.h5')
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
          verbose = 1
         )
    return hist1, model1, encoder_model1, decoder_model1

def grid_search(latent, half,train_size, X_train, y_train, y_train1):
    hist1, model1, encoder_model1, decoder_model1 = modelFit(5, 100, latent, half, X_train, y_train, y_train1)
    hist2 ,model2, encoder_model2, decoder_model2 = modelFit(10, 100, latent, half, X_train, y_train, y_train1)
    hist3 ,model3, encoder_model3, decoder_model3 = modelFit(20, 100, latent, half, X_train, y_train, y_train1)
    hist4 ,model4, encoder_model4, decoder_model4 = modelFit(30, 100, latent, half, X_train, y_train, y_train1)
    hist5 ,model5, encoder_model5, decoder_model5 = modelFit(50, 100, latent, half, X_train, y_train, y_train1)
    hist6 ,model6, encoder_model6, decoder_model6 = modelFit(80, 100, latent, half, X_train, y_train, y_train1)
    hist7 ,model7, encoder_model7, decoder_model7 = modelFit(100, 100, latent, half, X_train, y_train, y_train1)
    #hist8 ,model8, encoder_model8, decoder_model8 = modelFit(500, 100, latent, half)

    model1.save("models/{}_{}_5_double.h5".format(train_size,half))
    model2.save("models/{}_{}_10_double.h5".format(train_size,half))
    model3.save("models/{}_{}_20_double.h5".format(train_size,half))
    model4.save("models/{}_{}_30_double.h5".format(train_size,half))
    model5.save("models/{}_{}_50_double.h5".format(train_size,half))
    model6.save("models/{}_{}_80_double.h5".format(train_size,half))
    model7.save("models/{}_{}_100_double.h5".format(train_size,half))
    #model8.save("{}_{}_500.h5".format(train_size,half))
    
    encoder_model1.save("models/E{}_{}_5_double.h5".format(train_size,half))
    encoder_model2.save("models/E{}_{}_10_double.h5".format(train_size,half))
    encoder_model3.save("models/E{}_{}_20_double.h5".format(train_size,half))
    encoder_model4.save("models/E{}_{}_30_double.h5".format(train_size,half))
    encoder_model5.save("models/E{}_{}_50_double.h5".format(train_size,half))
    encoder_model6.save("models/E{}_{}_80_double.h5".format(train_size,half))
    encoder_model7.save("models/E{}_{}_100_double.h5".format(train_size,half))
    #encoder_model8.save("E{}_{}_500.h5".format(train_size,half))
    
    decoder_model1.save("models/D{}_{}_5_double.h5".format(train_size,half))
    decoder_model2.save("models/D{}_{}_10_double.h5".format(train_size,half))
    decoder_model3.save("models/D{}_{}_20_double.h5".format(train_size,half))
    decoder_model4.save("models/D{}_{}_30_double.h5".format(train_size,half))
    decoder_model5.save("models/D{}_{}_50_double.h5".format(train_size,half))
    decoder_model6.save("models/D{}_{}_80_double.h5".format(train_size,half))
    decoder_model7.save("models/D{}_{}_100_double.h5".format(train_size,half))
    #decoder_model8.save("D{}_{}_500.h5".format(train_size,half))
    
    count = [i for i in range(len(hist7.history['val_loss']))]
    val_loss_hist.append([hist7.history['val_loss'].index(min(hist7.history['val_loss'])),min(hist7.history['val_loss'])])
    print(val_loss_hist)
    for i, value in zip(count, hist7.history['val_loss']):
        print(i, value)

X_train=np.load('prepData/X_train_camFam3_1mutOnly_v3_chr2.npy')[:48000]
y_train=np.load('prepData/y_train_camFam3_1mutOnly_v3_chr2.npy')[:48000]
y_train1 = np.load('prepData/y_train1_camFam3_1mutOnly_v3_chr2.npy')[:48000]
y_train1 = concat(X_train, y_train1)

#grid_search(2, 1, 48000, X_train, y_train, y_train1)
#grid_search(32, 16, 48000, X_train, y_train, y_train1)
#grid_search(64, 32, 48000, X_train, y_train, y_train1)
#grid_search(128, 64, 48000, X_train, y_train, y_train1)
#grid_search(256, 128, 48000, X_train, y_train, y_train1)
#grid_search(512, 256, 48000, X_train, y_train, y_train1)
#grid_search(1024, 512, 48000, X_train, y_train, y_train1)
grid_search(2048, 1024, 48000, X_train, y_train, y_train1)


# In[ ]:






with open('loss_hist_48d.txt', 'wb') as fp:
    pickle.dump(val_loss_hist, fp)


# In[ ]:


def decode_sequence(input_seq, model, encoder_model, decoder_model):
    nucleotide = ['0', 'A', 'C', 'G', 'T', '-']
    # Encode the input as state vectors.
    #print(input_seq[0,0])
    index = 0
    states_value = encoder_model.predict(input_seq)
    #print(len(states_value))
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 12))
    target_seq[0][0]= np.hstack((input_seq[0,index], np.array([1,0,0,0,0,0])))
    #print(target_seq)
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    probability = 1
    
    while not stop_condition:
        index = index +1
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_token_index = np.random.choice(6, 1, p=output_tokens[0, -1, :])[0]
        
        #print(output_tokens[0, -1, :])
        sampled_char = nucleotide[sampled_token_index]
        decoded_sentence += sampled_char
        #print(decoded_sentence)
        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) == 10):
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 12))
        temp = np.zeros((6))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[0, index], temp))
        # target_seq[0, 0, sampled_token_index] = 1
        
        
        # Update states
        states_value = [h, c]

    return decoded_sentence

def get_prob(input_seq, target, model, encoder_model, decoder_model):
    nucleotide = ['0', 'A', 'C', 'G', 'T', '-']
    # Encode the input as state vectors.
    index = 0
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, 12))
    target_seq[0][0]= np.hstack((input_seq[0,index], np.array([1,0,0,0,0,0])))

    stop_condition = False
    decoded_sentence = ''
    probability = 1.0
    
    while not stop_condition:
        index = index +1
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        #print(output_tokens[0, -1, :])
        sampled_token_index = np.argmax(target[index-1])
        #sampled_token_index = np.random.choice(6, 1, p=output_tokens[0, -1, :])[0]
        probability = probability * output_tokens[0, -1, :][sampled_token_index]
        #print(output_tokens[0, -1, :])
        sampled_char = nucleotide[sampled_token_index]
        decoded_sentence += sampled_char
        #print(decoded_sentence)
        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) == 10):
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 12))
        temp = np.zeros((6))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[0, index], temp))
        # target_seq[0, 0, sampled_token_index] = 1
        
        
        # Update states
        states_value = [h, c]

    return decoded_sentence, probability

def decode_gru(input_seq, model, encoder_model, decoder_model):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, 6))
    target_seq[0][0]= np.array([1,0,0,0,0,0])
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = nucleotide[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) == 10):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 6))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h]

    return decoded_sentence

def diffList(a, b):
    count = 0
    length = len(a)
    for i in range(length):
        if a[i] != b[i]:
            count = count+1
    return count
def predict2(X_test, y_test, model, encoder_model, decoder_model, gru=False):
    x_true =[]
    y_hat =[]
    y_true =[]
    probList=[]
    productProb = 0
    for seq_index in range(len(X_test)):
        input_seq = X_test[seq_index: seq_index + 1]
        #print(input_seq[0])
        if gru:
            decoded_sentence = decode_gru(input_seq, model, encoder_model, decoder_model)
        else :
            decoded_sentence = decode_sequence(input_seq, model, encoder_model, decoder_model)
        _, prob = get_prob(input_seq, y_test[seq_index], model, encoder_model, decoder_model)
        probList.append(prob)
        productProb = productProb+ math.log(prob)
        input_sen = decoder(input_seq[0])
        print(input_sen, ' -> ',
              decoded_sentence, 'True:', decoder(y_test[seq_index]), 
              printHitMiss(decoded_sentence, decoder(y_test[seq_index])), 
              diffList(input_sen, decoded_sentence)
             )
        print(input_sen, ' -> ',
              decoder(y_test[seq_index]), 'True:', decoder(y_test[seq_index]), 
              prob,
              printHitMiss(decoded_sentence, decoder(y_test[seq_index])), 
              diffList(input_sen, decoded_sentence)
             )
        print()
        x_true.append(input_sen)
        y_hat.append(decoded_sentence)
        y_true.append(decoder(y_test[seq_index]))
    print("Mean and std of probabilities : {} , {}  ".format(np.mean(probList), np.std(probList)))
    print("Sum of log probabilities : {}".format(productProb))
    print("Percentage of target and prediction being identical: {}".format(accuracy(y_hat, y_true)))
    print("Percentage of training and prediction being identical: {}".format(accuracy(y_hat, x_true)))
    print("Accuracy given mutation happened : {}".format(accuracy2(x_true, y_hat, y_true)))
    #print("Test loss : {}".format(keras.losses.categorical_crossentropy(y_true, y_hat)))
    return x_true, y_hat, y_true


# In[ ]:


#model1.summary()
#model2.summary()
def grid_predict(train_size, half, X_test, y_test):
    model1 = load_model("{}_{}_5.h5".format(train_size,half))
#     model1 =load_model("{}_{}_10.h5".format(train_size,half))
#     model1 =load_model("{}_{}_20.h5".format(train_size,half))
#     model1 =load_model("{}_{}_30.h5".format(train_size,half))
#     model1 =load_model("{}_{}_50.h5".format(train_size,half))
#     model1 =load_model("{}_{}_80.h5".format(train_size,half))
#     model1 =load_model("{}_{}_100.h5".format(train_size,half))
#     model1 =load_model("{}_{}_500.h5".format(train_size,half))

    encoder_model1 = load_model("E{}_{}_5.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_10.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_20.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_30.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_50.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_80.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_100.h5".format(train_size,half))
#     encoder_model1 =load_model("E{}_{}_500.h5".format(train_size,half))

    decoder_model1 =load_model("D{}_{}_5.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_10.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_20.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_30.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_50.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_80.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_100.h5".format(train_size,half))
#     decoder_model1 =load_model("D{}_{}_500.h5".format(train_size,half))
    #return model1, encoder_model1, decoder_model1
    x_true, y_hat, y_true = predict2(X_test, y_test, model1, encoder_model1, decoder_model1, gru=False)
model1, encoder_model1, decoder_model1 = grid_predict(12000, 128, X_test, y_test)

encoder_model1.summary()
decoder_model1.summary()


# In[ ]:


count = [i for i in range(len(hist7.history['val_loss']))]
for i, value in zip(count, hist7.history['val_loss']):
    print(i, value)


# In[ ]:



def plotgraph(hist1):
    plt.plot(hist1.history['val_acc'])
    plt.plot(hist1.history['acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation', 'train'], loc='upper left')
    plt.show()
    plt.plot(hist1.history['val_loss'])
    plt.plot(hist1.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'train'], loc='upper left')
    plt.show()
    

x_true, y_hat, y_true = predict2(X_test, y_test, model1, encoder_model1, decoder_model1, gru=False)
#plotgraph(hist1)


# In[ ]:


x_true2, y_hat2, y_true2 = predict2(X_test, y_test, model2, encoder_model2, decoder_model2)
plotgraph(hist2)


# In[ ]:


x_true3, y_hat3, y_true3 = predict2(X_test, y_test, model3, encoder_model3, decoder_model3)
plotgraph(hist3)


# In[ ]:


x_true4, y_hat4, y_true4 = predict2(X_test, y_test, model4, encoder_model4, decoder_model4)
plotgraph(hist4)


# In[ ]:


x_true5, y_hat5, y_true5 = predict2(X_test, y_test, model5, encoder_model5, decoder_model5)
plotgraph(hist5)


# In[ ]:


x_true6, y_hat6, y_true6 = predict2(X_test, y_test, model6, encoder_model6, decoder_model6)
plotgraph(hist6)


# In[ ]:


x_true7, y_hat7, y_true7 = predict2(X_test, y_test, model7, encoder_model7, decoder_model7)
plotgraph(hist7)


# In[ ]:


x_true8, y_hat8, y_true8 = predict2(X_test, y_test, model8, encoder_model8, decoder_model8)
plotgraph(hist8)


# ## Finding ratio of mutation on predicted sequences

# In[ ]:


def countMutation(ancestor, target, realDes): 
    AtoG = 0
    AtoC = 0
    AtoT= 0
    GtoA= 0
    GtoC= 0
    GtoT= 0
    CtoA= 0
    CtoG= 0
    CtoT= 0
    TtoA= 0
    TtoG= 0
    TtoC= 0
    CG = 0
    CA = 0
    CAGT = 0
    CAAT = 0
    transitionCorrect = 0
    transversionCorrect = 0
    for j in range(len(ancestor)):
        for i in range(10):

                if ancestor[j][i] == 'A' and target[j][i] == 'G':
                    AtoG = AtoG+1
                    if target[j][i] == realDes[j][i]:
                        transitionCorrect+=1
                elif ancestor[j][i] == 'A' and target[j][i] == 'C':
                    AtoC = AtoC+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'A' and target[j][i] == 'T':
                    AtoT = AtoT+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'G' and target[j][i] == 'A':
                    GtoA = GtoA+1
                    if target[j][i] == realDes[j][i]:
                        transitionCorrect +=1
                elif ancestor[j][i] == 'G' and target[j][i] == 'C':
                    GtoC = GtoC+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'G' and target[j][i] == 'T':
                    GtoT = GtoT+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'C' and target[j][i] == 'A':
                    CtoA = CtoA+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'C' and target[j][i] == 'G':
                    CtoG = CtoG+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'C' and target[j][i] == 'T':
                    CtoT = CtoT+1
                    if target[j][i] == realDes[j][i]:
                        transitionCorrect +=1
                elif ancestor[j][i] == 'T' and target[j][i] == 'A':
                    TtoA = TtoA+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'T' and target[j][i] == 'G':
                    TtoG = TtoG+1
                    if target[j][i] == realDes[j][i]:
                        transversionCorrect +=1
                elif ancestor[j][i] == 'T' and target[j][i] == 'C':
                    TtoC = TtoC+1
                    if target[j][i] == realDes[j][i]:
                        transitionCorrect +=1
                else :
                    transitionCorrect = transitionCorrect
                if i<9 and ancestor[j][i] == 'C' and ancestor[j][i+1] == 'G' :
                    CG = CG+1
                    if target[j][i] == 'C' and target[j][i+1] == 'A' :
                        CA = CA+1
                if i<6 and ancestor[j][i] == 'C' and ancestor[j][i+1] == 'A' and ancestor[j][i+2] == 'A' and ancestor[j][i+3] == 'T':
                    CAAT = CAAT+1
#                     print(ancestor[j])
#                     print(target[j])
                    if target[j][i] == 'C' and target[j][i+1] == 'A'and target[j][i+2] == 'G' and target[j][i+3] == 'T':
                        CAGT = CAGT+1
                        #print(CAAT)
                #print(CAAT)

    print('This is A-->G :{}'.format(AtoG))
    print('This is A-->C :{}'.format(AtoC))
    print('This is A-->T :{}'.format(AtoT))
    print('This is G-->A :{}'.format(GtoA))
    print('This is G-->C :{}'.format(GtoC))
    print('This is G-->T :{}'.format(GtoT))
    print('This is C-->A :{}'.format(CtoA))
    print('This is C-->G :{}'.format(CtoG))
    print('This is C-->T :{}'.format(CtoT))
    print('This is T-->A :{}'.format(TtoA))
    print('This is T-->G :{}'.format(TtoG))
    print('This is T-->C :{}'.format(TtoC))
    numMutation = AtoG+AtoC+AtoT+GtoA+GtoC+GtoT+CtoA+CtoG+CtoT+TtoA+TtoG+TtoC
    transition = AtoG+GtoA+CtoT+TtoC
    transversion = numMutation-transition
    print('Total number of mutations: {}'.format(numMutation))
    print('Number of transitions: {}'.format( transition))
    print('Number of transversions: {}'.format(transversion))
    if transversion !=0 :
        print('Ratio of transition/transversion : {}'.format(transition/transversion))
    print('Percentage of mutation : {}'.format(numMutation/(10*(len(X_test)))))
    print('Percentage of transition predicted correct : {}'.format(transitionCorrect/transition))
    if transversion !=0 :
        print('Percentage of transversion predicted correct : {}'.format(transversionCorrect/transversion))
    
    print('This is the ratio of CG-->CA :{}'.format(CA/CG))
    print('This is the ratio of CAAT-->CAGT :{}'.format(CAGT/CAAT))
    
def positionAccuracy(x_true, y_hat, y_true):
    n = len(x_true[0])
    totalMut = [0] * n
    correctMut = [0] * n
    posAccuracy = [0] * n
    for i in range(n):
        count = 0
        countCorrect = 0
        for j in range(len(x_true)):
            if x_true[j][i] != y_true[j][i]:
                count = count+1
                if y_hat[j][i]== y_true[j][i]:
                    countCorrect = countCorrect+1
        #print(countCorrect, count)
        totalMut[i]=count
        correctMut[i]=countCorrect
        posAccuracy[i]=countCorrect/float(count)
        
        
    print ('This is the accuracy for each position : {}'.format(posAccuracy))

#countMutation(x_true6, y_hat6, y_true6)
#countMutation(x_true8, y_hat8, y_true8)


# ## Mutation Ratio

# In[ ]:


countMutation(x_true3, y_hat3, y_true3)
#positionAccuracy(x_true, y_hat, y_true)
countMutation(x_true4, y_hat4, y_true4)


# In[ ]:


countMutation(x_true8, y_hat8, y_true8)
#positionAccuracy(x_true2, y_hat2, y_true2)


# In[ ]:


countMutation(x_true, y_true, y_true)


# In[ ]:


X_test5=np.load('prepData/X_test_camFam3_1mutOnly_v3.npy')
y_test5=np.load('prepData/y_test_camFam3_1mutOnly_v3.npy')
x_true5, y_hat5, y_true5 = predict2(X_test5, y_test5, model1, encoder_model1, decoder_model1)


# In[ ]:


x_true6, y_hat6, y_true6 = predict2(X_test5, y_test5, model2, encoder_model2, decoder_model2)


# In[ ]:


countMutation(x_true5, y_hat5, y_true5)
countMutation(x_true6, y_hat6, y_true6)
countMutation(x_true5, y_true5, y_true5)


# In[ ]:


#K.clear_session()

