
# coding: utf-8

# In[1]:


# !pip3 install -U scikit-learn
# !pip3 install keras
# !pip3 install cudnnenv
# !pip3 install tensorflow-gpu
# !pip3 install matplotlib

# !conda uninstall -c anaconda cudatoolkit
#!nvidia-smi

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
import math

seq_length = 10
test_size = 50000
val_size = 30000
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
        output_tokens, h, c, h1, c1 = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print(output_tokens)
        #sampled_token_index = np.random.choice(6, 1, p=output_tokens[0, -1, :])[0]
        
        #print(output_tokens[0, -1, :])
        sampled_char = nucleotide[sampled_token_index]
        decoded_sentence += sampled_char
        #print(decoded_sentence)
        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) == seq_length):
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 12))
        temp = np.zeros((6))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[0, index], temp))
        # target_seq[0, 0, sampled_token_index] = 1
        
        
        # Update states
        states_value = [h, c, h1, c1]

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
    probability = []
    
    while not stop_condition:
        index = index +1
        output_tokens, h, c, h1, c1 = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        #print(output_tokens[0, -1, :])
        sampled_token_index = np.argmax(target[index-1])
        #sampled_token_index = np.random.choice(6, 1, p=output_tokens[0, -1, :])[0]
        probability.append(output_tokens[0, -1, :][sampled_token_index])
        #print(output_tokens[0, -1, :])
        sampled_char = nucleotide[sampled_token_index]
        decoded_sentence += sampled_char
        #print(decoded_sentence)
        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) == seq_length):
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 12))
        temp = np.zeros((6))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[0, index], temp))
        # target_seq[0, 0, sampled_token_index] = 1
        
        
        # Update states
        states_value = [h, c, h1, c1]

    return decoded_sentence, probability

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
    productProb = [0]*seq_length
    for seq_index in range(len(X_test)):
        input_seq = X_test[seq_index: seq_index + 1]
        #print(input_seq[0])
        if gru:
            decoded_sentence = decode_gru(input_seq, model, encoder_model, decoder_model)
        else :
            decoded_sentence = decode_sequence(input_seq, model, encoder_model, decoder_model)
        _, prob = get_prob(input_seq, y_test[seq_index], model, encoder_model, decoder_model)
        probList.append(prob)
        prob = [math.log(x) for x in prob]
        productProb = [sum(x) for x in zip(productProb, prob)]
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
    productProb = [x/test_size for x in productProb]
    print("Mean and std of probabilities : {} , {}  ".format(np.mean(probList), np.std(probList)))
    print("Sum of log probabilities : {}".format(productProb))
    print("Percentage of target and prediction being identical: {}".format(accuracy(y_hat, y_true)))
    print("Percentage of training and prediction being identical: {}".format(accuracy(y_hat, x_true)))
    print("Accuracy given mutation happened : {}".format(accuracy2(x_true, y_hat, y_true)))
    #print("Test loss : {}".format(keras.losses.categorical_crossentropy(y_true, y_hat)))
    #return x_true, y_hat, y_true


def grid_predict(train_size, half, epoch, X_test, y_test):
    model1 = load_model("models/{}_{}_{}_double.h5".format(train_size,half,epoch))

    encoder_model1 = load_model("models/E{}_{}_{}_double.h5".format(train_size,half, epoch))

    decoder_model1 =load_model("models/D{}_{}_{}_double.h5".format(train_size,half, epoch))

    predict2(X_test, y_test, model1, encoder_model1, decoder_model1, gru=False)
    



# In[3]:
def concat(input1, input2):
    result = []
    for x, y in zip(input1, input2):
        result.append(np.hstack((x, y)))
    
    return np.array(result)

def get_data(trainInd, valInd, testInd):
    X_train=np.load('prepData/X_train_camFam3_1mut_v3_chr2.npy')[:trainInd]
    X_val=np.load('prepData/X_val_camFam3_1mut_v3_chr2.npy')[:valInd]
    X_test=np.load('prepData/X_test_camFam3_1mut_v3_chr2.npy')[:testInd]
    y_train=np.load('prepData/y_train_camFam3_1mut_v3_chr2.npy')[:trainInd]
    y_val=np.load('prepData/y_val_camFam3_1mut_v3_chr2.npy')[:valInd]
    y_test=np.load('prepData/y_test_camFam3_1mut_v3_chr2.npy')[:testInd]

    y_train1 = np.load('prepData/y_train1_camFam3_1mut_v3_chr2.npy')[:trainInd]
    y_val1 = np.load('prepData/y_val1_camFam3_1mut_v3_chr2.npy')[:valInd]
    y_test1 = np.load('prepData/y_test1_camFam3_1mut_v3_chr2.npy')[:testInd]

    y_train1 = concat(X_train, y_train1)
    y_val1 = concat(X_val, y_val1)
    y_test1 = concat(X_test, y_test1)
    return X_test, y_test


train_size = 0
hidden = [32,64,128,256,512]
epoch = [5,5,5,5,5]


X_test, y_test = get_data(train_size, val_size, test_size)
for h, e in zip(hidden, epoch):
    print("Train size = {}, hidden_size = {}, epoch = {}".format(train_size, h, e))
    grid_predict(train_size, h, e, X_test, y_test)
    print("The end of Train size = {}, hidden_size = {}, epoch = {}".format(train_size, h, e))
