import math
import itertools
import operator
from tqdm import tqdm

from scipy import stats
import tensorflow as tf
import math
import itertools
import operator
from tensorflow.python.keras import backend as k
from tqdm import tqdm, tqdm_notebook, notebook
import numpy as np
from tensorflow.keras import layers
import os
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import random
from random import choice
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

from sklearn.preprocessing import LabelEncoder
from bio import AlignIO

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing import sequence
import sklearn
import numpy as np
import re
import pickle
import itertools
import random
import string
from joblib import dump, load
import pickle
import sys
import os


# In[3]:


ancName = sys.argv[1]
desName = sys.argv[2]
train_size = int(sys.argv[3])
seq_length = int(sys.argv[4])
print(seq_length)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#seq_length = 11
val_loss_hist = []
epoch = 80

K.clear_session()
keras.backend.clear_session()

anc = str(np.load('prepData/insert2Anc_{}_{}.npy'.format(ancName, desName)))
des = str(np.load('prepData/insert2Des_{}_{}.npy'.format(ancName, desName)))


# In[ ]:
anc = np.array(list(anc+'0'))
des = np.array(list(des+'0'))
label_encoder = LabelEncoder()
label_encoder.fit(des)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded_des = label_encoder.transform(des)
integer_encoded_anc = label_encoder.transform(anc)
#one hot the sequence
integer_des = integer_encoded_des.reshape(len(integer_encoded_des), 1)

onehot_encoder.fit(integer_des)
encoded_des =onehot_encoder.transform(integer_des)

integer_anc = integer_encoded_anc.reshape(len(integer_encoded_anc), 1)
encoded_anc = onehot_encoder.transform(integer_anc)

print(encoded_des)
print(encoded_anc)

print(len(encoded_des[0]))
encode_dimension= len(encoded_des[0])

print(label_encoder.inverse_transform(range(encode_dimension)))
print(onehot_encoder.transform(np.ones(1).reshape(-1,1)))
key = list(label_encoder.inverse_transform(range(encode_dimension)))

dump(label_encoder, 'label_encoder.joblib') 
dump(onehot_encoder, 'onehot_encoder.joblib')
# In[ ]:


# def splice(input, pad):
#     result = []
#     if pad == False:
#         for i in range(len(input)-seq_length-1):
#             result.append(input[i:i+seq_length])
#     else :
#         for i in range(len(input)-seq_length-1):
#             # print(len(input)-seq_length-1)
#             # print(np.concatenate((onehot_encoder.transform(np.ones(1).reshape(-1,1)), 
#             #                              input[i:i+seq_length-1]), 
#             #                              axis = 0))
#             result.append(np.concatenate((onehot_encoder.transform(np.ones(1).reshape(-1,1)), 
#                                          input[i:i+seq_length-1]), 
#                                          axis = 0)
#                          )
#     return np.array(result)
def splice(input, pad):
    result = []
    if pad == False:
        for i in tqdm(range(int(len(input)/seq_length))):
            result.append(input[i*seq_length:(i+1)*seq_length])
    else :
        for i in tqdm(range(int(len(input)/seq_length))):
            result.append(np.concatenate((onehot_encoder.transform((np.ones(1)*key.index('0')).reshape(-1,1)), 
                                         input[i*seq_length:(i+1)*seq_length-1]), 
                                         axis = 0)
                         )
    return np.array(result)

def decoder(input):
    nucleotide = label_encoder.inverse_transform(range(encode_dimension))
    decoded = ''
    for i in range(len(input)):
        # decoded= decoded+(nucleotide[np.argmax(onehot_encoder.inverse_transform(input[i].reshape(-1, 1)))])
        #print(np.argmax(input[i]))
        #print(nucleotide)
        decoded= decoded+nucleotide[np.argmax(input[i])]
    return decoded

y = splice(encoded_des, False)
y1 = splice(encoded_des, pad = True)
X = splice(encoded_anc, False)

print(y1.shape)
print(X.shape)
print(X)

onehot_encoder.transform(np.ones(1).reshape(-1,1))
for i in range (100):
    print(decoder(X[i]), decoder(y1[i]), decoder(y[i]))


# In[ ]:
def concat(input1, input2):
    result = []
    for x, y in tqdm(zip(input1, input2)):
        result.append(np.concatenate((x, y), axis=1))
        # print(decoder(x), decoder(y), decoder(np.concatenate((x, y))))
    
    return np.array(result)

# X_train, X_test, y_train1, y_test1, y_train, y_test = train_test_split(
#     X, y1, y, test_size=0.2, random_state=42)
X_train, X_test, y_train1, y_test1, y_train, y_test = train_test_split(
    X, y1, y, test_size=0.2, random_state=42)

y_train1 = concat(X_train, y_train1)
# y_val1 = concat(X_val, y_val1)
y_test1 = concat(X_test, y_test1)
    
nucleotide = label_encoder.inverse_transform(range(encode_dimension))

def decoder(input):
    nucleotide = label_encoder.inverse_transform(range(encode_dimension))
    decoded = ''
    for i in range(len(input)):
        # decoded= decoded+(nucleotide[np.argmax(onehot_encoder.inverse_transform(input[i].reshape(-1, 1)))])
        #print(np.argmax(input[i]))
        #print(nucleotide)
        decoded= decoded+nucleotide[np.argmax(input[i])]
    return decoded


# In[ ]:


onehot_encoder.transform(np.ones(1).reshape(-1,1))
for i in range (100):
    print(decoder(X_train[i]), decoder(y_train1[i]), decoder(y_train[i]))
print(y_train1[1])


# In[ ]:


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


# In[ ]:


# In[ ]:

def lstm_model(latent_dim, half):
    batch_size = 1000  # Batch size for training.
    epochs = 45  # Number of epochs to train for.
#     latent_dim = 128  # Latent dimensionality of the encoding space.
#     half = 64
    num_samples = 10000  # Number of samples to train on.
    encoder_inputs = layers.Input(shape=(None, encode_dimension))
    
    encoder = layers.Bidirectional(layers.LSTM(half, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    state_h = layers.concatenate([forward_h, backward_h])
    state_c = layers.concatenate([forward_c, backward_c])
    
    
    # only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder
    decoder_inputs = layers.Input(shape=(None, 2*encode_dimension))
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = layers.Dense(encode_dimension, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # inference
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = layers.Input(shape=(latent_dim,))
    decoder_state_input_c = layers.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.Model(
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
          #validation_data=([X_test,y_test1], y_test),
          validation_split=0.2,
          verbose = 1
         )
    return hist1, model1, encoder_model1, decoder_model1

def grid_search(latent, half,train_size, X_train, y_train, y_train1):
    hist1, model1, encoder_model1, decoder_model1 = modelFit(1, 256, latent, half, X_train, y_train, y_train1)
    hist2 ,model2, encoder_model2, decoder_model2 = modelFit(2, 256, latent, half, X_train, y_train, y_train1)
    hist3 ,model3, encoder_model3, decoder_model3 = modelFit(10, 256, latent, half, X_train, y_train, y_train1)
    #hist4 ,model4, encoder_model4, decoder_model4 = modelFit(30, 1000, latent, half, X_train, y_train, y_train1)
    #hist5 ,model5, encoder_model5, decoder_model5 = modelFit(50, 100, latent, half, X_train, y_train, y_train1)
    #hist6 ,model6, encoder_model6, decoder_model6 = modelFit(80, 100, latent, half, X_train, y_train, y_train1)
    #hist7 ,model7, encoder_model7, decoder_model7 = modelFit(100, 100, latent, half, X_train, y_train, y_train1)
    #hist8 ,model8, encoder_model8, decoder_model8 = modelFit(500, 100, latent, half)

    model1.save("models/insert2_{}_{}_{}_{}_1.h5".format(ancName, desName,ancName, desName))
    model2.save("models/insert2_{}_{}_{}_{}_2.h5".format(ancName, desName,ancName, desName))
    model3.save("models/insert2_{}_{}_{}_{}_10.h5".format(ancName, desName,ancName, desName))
    #model4.save("models/insert2_{}_{}_30_double.h5".format(train_size,half))
    #model5.save("models/_gap_hg38_{}_{}_50_double.h5".format(train_size,half))
    #model6.save("models/_gap_hg38_{}_{}_80_double.h5".format(train_size,half))
    #model7.save("models/_gap_hg38_{}_{}_100_double.h5".format(train_size,half))
    #model8.save("_gap_hg38_{}_{}_500.h5".format(train_size,half))
    
    encoder_model1.save("models/E_insert2_{}_{}_{}_{}_1.h5".format(ancName, desName,ancName, desName))
    encoder_model2.save("models/E_insert2_{}_{}_{}_{}_2.h5".format(ancName, desName,ancName, desName))
    encoder_model3.save("models/E_insert2_{}_{}_{}_{}_10.h5".format(ancName, desName,ancName, desName))
    #encoder_model4.save("models/E_insert2_{}_{}_30_double.h5".format(train_size,half))
    #encoder_model5.save("models/E_gap_hg38_{}_{}_50_double.h5".format(train_size,half))
    #encoder_model6.save("models/E_gap_hg38_{}_{}_80_double.h5".format(train_size,half))
    #encoder_model7.save("models/E_gap_hg38_{}_{}_100_double.h5".format(train_size,half))
    #encoder_model8.save("E_gap_hg38_{}_{}_500.h5".format(train_size,half))
    
    decoder_model1.save("models/D_insert2_{}_{}_{}_{}_1.h5".format(ancName, desName,ancName, desName))
    decoder_model2.save("models/D_insert2_{}_{}_{}_{}_2.h5".format(ancName, desName,ancName, desName))
    decoder_model3.save("models/D_insert2_{}_{}_{}_{}_10.h5".format(ancName, desName,ancName, desName))
    #decoder_model4.save("models/D_insert2_{}_{}_30_double.h5".format(train_size,half))
    #decoder_model5.save("models/D_gap_hg38_{}_{}_50_double.h5".format(train_size,half))
    #decoder_model6.save("models/D_gap_hg38_{}_{}_80_double.h5".format(train_size,half))
    #decoder_model7.save("models/D_gap_hg38_{}_{}_100_double.h5".format(train_size,half))
    #decoder_model8.save("D_gap_hg38_{}_{}_500.h5".format(train_size,half))
    
#     count = [i for i in range(len(hist3.history['val_loss']))]
#     val_loss_hist.append([hist3.history['val_loss'].index(min(hist3.history['val_loss'])),min(hist3.history['val_loss'])])
#     print(val_loss_hist)
#     for i, value in zip(count, hist3.history['val_loss']):
#         print(i, value)



# grid_search(2, 1, train_size, X_train, y_train, y_train1)
# grid_search(16, 8, train_size, X_train, y_train, y_train1)        
# grid_search(32, 16, train_size, X_train, y_train, y_train1)
# grid_search(64, 32, train_size, X_train, y_train, y_train1)
# grid_search(128, 64, train_size, X_train, y_train, y_train1)
# grid_search(256, 128, train_size, X_train, y_train, y_train1)
# grid_search(512, 256, train_size, X_train, y_train, y_train1)
# grid_search(1024, 512, train_size, X_train, y_train, y_train1)
#grid_search(8192, 4096, train_size, X_train, y_train, y_train1)

# with open('loss_hist.txt', 'wb') as fp:
#     pickle.dump(val_loss_hist, fp)


# In[ ]:


# get_ipython().run_line_magic('precision', '2')
test_size = len(y_test)
val_size = 30000

key = ['-', '0', 'A', 'B', 'C', 'G', 'I', 'L', 'N', 'O', 'P', 'T', 'V' ,'X' ,'b', 'c', 'f', 'g', 'h', 'i',
       'o', 'p' ,'r']

mapDict = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', '-': '-', 'AA': 'O', 'AC': 'h', '0': '0',
           'AT': 'b', 'AG': 'V', 'CA': 'r', 'CC': 'p', 'CT': 'o', 'CG': 'i', 'TA': 'g', 
           'TC': 'I', 'TT': 'f', 'TG': 'L', 'GA': 'B', 'GC': 'c', 'GT': 'X', 'GG': 'P'}

rev_dict = {v: k for k, v in mapDict.items()}
#print(rev_dict.keys())
rev_key = []
for item in key:
    #print(item)
    if item in list(rev_dict.keys()):
        rev_key.append(rev_dict[item])
        #print('hi')
    else :
        rev_key.append(item)
print(rev_key)

nucleotide = label_encoder.inverse_transform(range(encode_dimension))

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
    nucleotide = label_encoder.inverse_transform(range(encode_dimension))
    # Encode the input as state vectors.
    #print(input_seq[0,0])
    index = 0
    states_value = encoder_model.predict(input_seq)
    #print(len(states_value))
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, encode_dimension*2))
    target_seq[0][0]= np.hstack((input_seq[0,index], onehot_encoder.transform(np.ones(1).reshape(-1,1))[0]))
    #print(target_seq)
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_seq = ''
    probability = 1
    
    while not stop_condition:
        index = index +1
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.random.choice(encode_dimension, 1, p=output_tokens[0, -1, :])[0]
        #list(mapDict.keys())
        # for i in range(len(output_tokens[0])):
        #     print(i, dict(zip(rev_key, output_tokens[0][i])))
        sampled_nucleotide = nucleotide[np.random.choice(encode_dimension, 1, p=output_tokens[0, -1, :])[0]]
        
        decoded_seq += sampled_nucleotide
        #print(sampled_nucleotide, decoded_seq)
        #print(decoded_sentence)
        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_seq) == seq_length):
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, encode_dimension*2))
        temp = np.zeros((encode_dimension))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[0, index], temp))
        # target_seq[0, 0, sampled_token_index] = 1
        
        
        # Update states
        states_value = [h, c]

    return decoded_seq

def get_prob(input_seq, target, model, encoder_model, decoder_model):
    # Encode the input as state vectors.
    nucleotide = label_encoder.inverse_transform(range(encode_dimension))
    index = 0
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, encode_dimension*2))
    target_seq[0][0]= np.hstack((input_seq[0,index], onehot_encoder.transform(np.ones(1).reshape(-1,1))[0]))

    stop_condition = False
    decoded_seq = ''
    probability = []
    
    while not stop_condition:
        index = index +1
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(target[index-1])
        
        probability.append(output_tokens[0, -1, :][sampled_token_index])
        sampled_nucleotide = nucleotide[np.random.choice(encode_dimension, 1, p=output_tokens[0, -1, :])[0]]
        decoded_seq += sampled_nucleotide
        if (len(decoded_seq) == seq_length):
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, encode_dimension*2))
        temp = np.zeros((encode_dimension))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[0, index], temp))
        # target_seq[0, 0, sampled_token_index] = 1
        
        
        # Update states
        states_value = [h, c]

    return decoded_seq, probability

def diffList(a, b):
    count = 0
    length = len(a)
    for i in range(length):
        if a[i] != b[i]:
            count = count+1
    return count

def decoder(input):
    nucleotide = label_encoder.inverse_transform(range(encode_dimension))
    decoded = ''
    for i in range(len(input)):
        # decoded= decoded+(nucleotide[np.argmax(onehot_encoder.inverse_transform(input[i].reshape(-1, 1)))])
        #print(np.argmax(input[i]))
        #print(nucleotide)
        decoded= decoded+nucleotide[np.argmax(input[i])]
    return decoded

#for seq_index in range(1):
def predict2(X_test, y_test, model, encoder_model, decoder_model, gru=False):
    x_true =[]
    y_hat =[]
    y_true =[]
    probList=[]
    generator_output = []
    productProb = [0]*seq_length

    for seq_index in tqdm(range(len(X_test))):
        input_seq = X_test[seq_index: seq_index + 1]
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
              printHitMiss(decoded_sentence, decoder(y_test[seq_index]))
              #diffList(input_sen, decoded_sentence)
             )
        print(input_sen, ' -> ',
              decoder(y_test[seq_index]), 'True:', decoder(y_test[seq_index]), 
              prob,
              printHitMiss(decoded_sentence, decoder(y_test[seq_index]))
              #diffList(input_sen, decoded_sentence)
             )
        x_true.append(input_sen)
        y_hat.append(decoded_sentence)
        y_true.append(decoder(y_test[seq_index]))
    #generator_output.append(input_sen+decoded_seq)
    productProb = [x/test_size for x in productProb]
    print("Mean and std of probabilities : {} , {}  ".format(np.mean(probList), np.std(probList)))
    print("Sum of log probabilities : {} ---- {}".format(productProb, np.mean(productProb)))
    print("Percentage of target and prediction being identical: {}".format(accuracy(y_hat, y_true)))
    print("Percentage of training and prediction being identical: {}".format(accuracy(y_hat, x_true)))
    print("Accuracy given mutation happened : {}".format(accuracy2(x_true, y_hat, y_true)))

    cross_entropy_str = "Sum of log probabilities {} {} {} {} : {} ---- {}".format(ancName, desName, train_size, seq_length, productProb, np.mean(productProb))
    #np.save('data/hg38_output.npy', generator_output)
    
    return x_true, y_hat, y_true, cross_entropy_str

def grid_predict(train_size, half, epoch, X_test, y_test):
    model1 = load_model("models/insert2_{}_{}_{}_{}_{}.h5".format(ancName, desName, ancName, desName, epoch))

    encoder_model1 = load_model("models/E_insert2_{}_{}_{}_{}_{}.h5".format(ancName, desName, ancName, desName, epoch))

    decoder_model1 =load_model("models/D_insert2_{}_{}_{}_{}_{}.h5".format(ancName, desName, ancName, desName, epoch))

    inputAll, predAll, outputAll, cross_entropy_str = predict2(X_test, y_test, model1, encoder_model1, decoder_model1, gru=False)

    file = open("cross_entropy_loss.txt","w")
    file.write(cross_entropy_str)
    file.close()
    
    return inputAll, predAll, outputAll
    



# In[3]:
def concat(input1, input2):
    result = []
    for x, y in zip(input1, input2):
        result.append(np.hstack((x, y)))
    
    return np.array(result)




# hidden = [16, 32, 64,128,256,512]
# epoch = [10, 10, 2, 2, 2, 1]
hidden = [512]
epoch = [2]
#X_test, y_test = get_data(train_size, val_size, test_size)
for h, e in zip(hidden, epoch):
    print("Anc : {}, Des : {}, Train size = {}, hidden_size = {}, epoch = {}".format(ancName, desName, train_size, h, e))
    inputAll, predAll, outputAll = grid_predict(train_size, h, e, X_test, y_test)
    print("The end of Anc : {}, Des : {}, Train size = {}, hidden_size = {}, epoch = {}".format(ancName, desName,train_size, h, e))
    

    




onehot_encoder.transform(np.ones(1).reshape(-1,1))
for i in range (100):
    print(decoder(X_train[i]), decoder(y_train[i]), decoder(y_train1[i]))
    
print(y_train[1].shape)
