#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
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
# from Bio.Align import MultipleSeqAlignment
# from Bio.SeqRecord import SeqRecord
# from Bio import SeqIO
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
import sys

ancName = sys.argv[1]
desName = sys.argv[2]
sample_size = int(sys.argv[3])
gpu = sys.argv[4]
chromosome = sys.argv[5]


os.environ["CUDA_VISIBLE_DEVICES"]=gpu
context_length = 15
val_loss_hist = []

# K.clear_session()
# keras.backend.clear_session()



anc = str(np.load('prepData/insert2Anc_{}_hg38_{}.npy'.format(ancName, chromosome)))[:sample_size]
des = str(np.load('prepData/insert2Des_{}_hg38_{}.npy'.format(ancName, chromosome)))[:sample_size]
anc = np.array(list(anc+'0'))
des = np.array(list(des+'0'))

# with open('label_encoder.pickle', 'rb') as f:
#     label_encoder = pickle.load(f)
# with open('onehot_encoder.pickle', 'rb') as f:
#     onehot_encoder = pickle.load(f)

label_encoder = load('label_encoder.joblib') 
onehot_encoder = load('onehot_encoder.joblib') 

integer_encoded_des = label_encoder.transform(des)
integer_encoded_anc = label_encoder.transform(anc)
integer_des = integer_encoded_des.reshape(len(integer_encoded_des), 1)
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

nucleotide = label_encoder.inverse_transform(range(encode_dimension))


# In[2]:


# def splice(input, pad):
#     result = []
#     if pad == False:
#         for i in tqdm(range(len(input)-seq_length-1)):
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
                          
# sliced_anc = splice(encoded_anc, False)


# In[3]:


def decode_sequence(input_seq, model, encoder_model, decoder_model):
    length = len(input_seq)
    nucleotide = label_encoder.inverse_transform(range(encode_dimension))
    index = 0
    initial_context = np.expand_dims(input_seq[0: context_length], axis=0)
    states_value = encoder_model.predict(initial_context)
    target_seq = np.zeros((1, 1, encode_dimension*2))
    target_seq[0][0]= np.hstack((input_seq[0], onehot_encoder.transform(np.ones(1).reshape(-1,1))[0]))
    decoded_seq = ''
    for i in tqdm(range(1, length)):
        if i%context_length == 0 :
            context = np.expand_dims(input_seq[i: i+context_length], axis=0)
            states_value = encoder_model.predict(context)
            
#         stop_condition = False
        
#         while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.random.choice(encode_dimension, 1, p=output_tokens[0, -1, :])[0]
        sampled_nucleotide = nucleotide[sampled_token_index]
        if (sampled_nucleotide == '') or (not sampled_nucleotide.isprintable()) or (sampled_nucleotide.isspace()):
            decoded_seq += '0'
        else :
            decoded_seq += sampled_nucleotide
#         if (len(decoded_seq) == context_length):
#             break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, encode_dimension*2))
        temp = np.zeros((encode_dimension))
        temp[sampled_token_index] = 1
        target_seq[0][0]= np.hstack((input_seq[i], temp))
        
        if i == length -1:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
            sampled_token_index = np.random.choice(encode_dimension, 1, p=output_tokens[0, -1, :])[0]
            sampled_nucleotide = nucleotide[sampled_token_index]

            decoded_seq += sampled_nucleotide
            

        # Update states
        states_value = [h, c]

    return decoded_seq


model = load_model("models/insert2_{}_hg38_{}_hg38_10.h5".format(ancName, ancName))
encoder_model = load_model("models/E_insert2_{}_hg38_{}_hg38_10.h5".format(ancName, ancName))
decoder_model = load_model("models/D_insert2_{}_hg38_{}_hg38_10.h5".format(ancName, ancName))
decoded_seq = decode_sequence(encoded_anc, model, encoder_model, decoder_model)
print(decoded_seq)


# In[4]:


np.save('simulated_{}_{}_{}.npy'.format(ancName, sample_size, chromosome), decoded_seq)
