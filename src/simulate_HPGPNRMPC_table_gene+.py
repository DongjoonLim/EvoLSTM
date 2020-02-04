#!/usr/bin/env python
# coding: utf-8

# In[1]:

import seaborn as sns
sns.set()
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


# In[2]:



os.environ["CUDA_VISIBLE_DEVICES"]="0"
context_length = 15
val_loss_hist = []

# K.clear_session()
# keras.backend.clear_session()

ancName = '_HPGPNRMPC'
desName = 'hg38'


anc = str(np.load('prepData/insert2Anc_{}_{}_gene+_chr2.npy'.format(ancName, desName)))
des = str(np.load('prepData/insert2Des_{}_{}_gene+_chr2.npy'.format(ancName, desName)))
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
print(len(anc))


# In[ ]:


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


model = load_model("models/insert2__HPGPNRMPC_hg38__HPGPNRMPC_hg38_+_10.h5")
encoder_model = load_model("models/E_insert2__HPGPNRMPC_hg38__HPGPNRMPC_hg38_+_10.h5")
decoder_model = load_model("models/D_insert2__HPGPNRMPC_hg38__HPGPNRMPC_hg38_+_10.h5")
decoded_seq = decode_sequence(encoded_anc, model, encoder_model, decoder_model)
print(decoded_seq)


# In[ ]:





# In[12]:


def load_seq(chromList):
    inputAll = ''
    predAll = ''
    outputAll = ''
    for chromosome in chromList:
        try:
            inputAll += str(np.load('prepData/insert2Anc_{}_{}_gene+_chr{}.npy'.format(ancName,chromosome)))#[:10000000]
            outputAll += str(np.load('prepData/insert2Anc_{}_{}_gene+_chr{}.npy'.format(ancName,chromosome)))#[:10000000]
            predAll += str(np.load('prepData/simulated_{}_-1_chr{}.npy'.format(ancName, chromosome)))#[:10000000]
        except FileNotFoundError:
            print(chromosome)
            continue
        print(len(inputAll), len(outputAll), len(predAll))
        print(inputAll[-10:], outputAll[-10:], predAll[-10:])
    return [inputAll], [outputAll], [predAll]

# def load_seq(chromList):
#     inputAll = ''
#     predAll = ''
#     outputAll = ''
#     for chromosome in chromList:
#         inputAll += str(np.load('prepData/insert2Anc_{}_hg38_chr{}.npy'.format(ancName,chromosome)))[:10000000]
#         outputAll += str(np.load('prepData/insert2Des_{}_hg38_chr{}.npy'.format(ancName,chromosome)))[:10000000]
#         predAll += str(np.load('simulated_{}_10000000_chr{}.npy'.format(ancName, chromosome)))[:10000000]
#     return [inputAll], [outputAll], [predAll]
    
mut_dict = np.load('mut_dict_insert2.npy',allow_pickle=True).item()
inv_dict = {v: k for k, v in mut_dict.items()}
print(inv_dict)

inputAll, outputAll, predAll = load_seq([12,16,17,19,20,21,22])


# In[41]:



# decoded_seq = str(np.load('simulated_{}_10000000_chr{}.npy'.format(ancName, chromosome)))
# inputAll =[''.join(anc)[:-1]]
# predAll = [decoded_seq[:-1]]
# outputAll = [''.join(des)[:-1]

lstm_inputAll = [''.join(anc)[:2000000]]
lstm_predAll = [str(np.load('simulated_{}_lstm.npy'.format(ancName)))]
lstm_outputAll = [''.join(des)[:2000000]]
contextLen = 2
numBin = 10
def contextMut(size, ancNuc, desNuc, anc, des, pred, evol, lstm = False, table='', tableCon = 0):
    cont = list(itertools.product('ACGT', repeat=size))
    cont1 = list(itertools.product('ACGT', repeat=size))
    context_dict = {}
    count_dict = {}
    for i in cont1:
        for j in cont1:
            #context_dict[(''.join(i)+'A'+''.join(j) , ''.join(i)+'G'+''.join(j))] = 0
            context_dict[(''.join(i)+ancNuc+''.join(j))] = 0
            count_dict[(''.join(i)+ancNuc+''.join(j))] = 0
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            if a[i+size:i+size+len(ancNuc)] == ancNuc :
                count_dict[(a[i:i+size*2+len(ancNuc)])] += 1
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            if a[i+size:i+size+len(ancNuc)] == ancNuc and b[i+size+int(len(ancNuc)/2)] == desNuc:
                context_dict[(a[i:i+size*2+len(ancNuc)])] += 1
    for key in context_dict.keys():
        if count_dict[key] !=0:
            context_dict[key] = context_dict[key]/count_dict[key] 
        else :
            continue
    sorted_context = sorted(context_dict.items(), key=operator.itemgetter(1), reverse = 1)
    sorted_context = dict(sorted_context)
    if lstm ==False:
        if pred == True and evol ==False:
            np.save('data/pred{}_context{}->{}_{}_{}.npy'.format(table,ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred == True and evol == True:
            np.save('data/pred{}_evol_context{}->{}_{}_{}.npy'.format(table,ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred == False and evol ==False:
            np.save('data/true_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred == False and evol ==True:
            np.save('data/true_evol_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
        return sorted_context
    elif lstm ==True :
        if pred == True:
            np.save('data/pred_lstm_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred ==False:
            np.save('data/true_lstm_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
    return sorted_context
            
def contextMutInsert(size, ancNuc, desNuc, anc, des, pred, evol, lstm = False, table = '',tableCon = 0):
    cont1 = list(itertools.product('ACGT', repeat=size-1))
    cont = list(itertools.product('ACGT', repeat=size))
    context_dict = {}
    count_dict = {}
    for i in cont:
        for j in cont:
            #context_dict[(''.join(i)+'A'+''.join(j) , ''.join(i)+'G'+''.join(j))] = 0
            context_dict[(''.join(i)+ancNuc+''.join(j))] = 0
            count_dict[(''.join(i)+ancNuc+''.join(j))] = 0
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            count_dict[(a[i:i+size]+ancNuc+a[i+size+len(ancNuc):i+size*2+len(ancNuc)])] += 1
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            inserted_nuc = inv_dict[b[i+size+int(len(ancNuc)/2)-1]]
            if len(inserted_nuc) >1 and inserted_nuc[1] == desNuc:
                context_dict[(a[i:i+size]+ancNuc+a[i+size+len(ancNuc):i+size*2+len(ancNuc)])] += 1
    for key in context_dict.keys():
        if count_dict[key] !=0:
            context_dict[key] = context_dict[key]/count_dict[key] 
        else :
            continue
    sorted_context = sorted(context_dict.items(), key=operator.itemgetter(1), reverse = 1)
    sorted_context = dict(sorted_context)
    if lstm ==False:
        if pred == True and evol ==False:
            np.save('data/pred{}_context{}->{}_{}_{}.npy'.format(table,ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred == True and evol == True:
            np.save('data/pred{}_evol_context{}->{}_{}_{}.npy'.format(table,ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred == False and evol ==False:
            np.save('data/true_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred == False and evol ==True:
            np.save('data/true_evol_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
        return sorted_context
    elif lstm ==True :
        if pred == True:
            np.save('data/pred_lstm_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
        elif pred ==False:
            np.save('data/true_lstm_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), sorted_context)
    return sorted_context
            
def decodeList(inputAll, predAll, outputAll):
    inp =[]
    inp2 = []
    pre = []
    out = []
    for i, p, o in tqdm(zip(inputAll, predAll, outputAll)):
        input, pred = decodeDictSeq(i, p, mut_dict)
        input2, output = decodeDictSeq(i,o, mut_dict)
        inp.append(input)
        inp2.append(input2)
        pre.append(pred)
        out.append(output)
    return inp, inp2, pre, out

def valueFloat(data_list):
    newDict = dict(zip(data_list.keys(), [float(value) for value in data_list.values()]))
    return newDict
def plotPointMut(n_groups,ancNuc, desNuc):
    predSeq = np.load('data/pred_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()

    trueSeq = np.load('data/true_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()

#     evolSeq = np.load('data/true_evol_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()
    
    lstmSeq = np.load('data/pred_lstm_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()

    true = list(trueSeq.values())[:n_groups]
    true_context = list(trueSeq.keys())[:n_groups]
    pred = []
    evol = []
    lstm = []
    for i in true_context:
        pred.append(predSeq[i])
#         evol.append(evolSeq[i])
        lstm.append(lstmSeq[i])

    # create plot
    fig, ax = plt.subplots(figsize=(20, 10))
    index = np.arange(n_groups)
    bar_width = 0.05
    opacity = 0.8

    rects1 = plt.bar(index, pred, bar_width,
    alpha=opacity,
    color='b',
    label='seq2seq_pred')
    
    rects2 = plt.bar(index + bar_width, lstm, bar_width,
    alpha=opacity,
    color='m',
    label='lstm_pred')
    
    rects3 = plt.bar(index + bar_width*2, true, bar_width,
    alpha=opacity,
    color='g',
    label='true')
    
#     rects3 = plt.bar(index + bar_width*3, evol, bar_width,
#     alpha=opacity,
#     color='r',
#     label='evol_pred')

    plt.xlabel('context')
    plt.ylabel('rate')
    plt.title('{} to {} point mutation'.format(ancNuc, desNuc))
    plt.xticks(index + bar_width, list(trueSeq.keys())[:n_groups])
    plt.legend()

    #plt.tight_layout()
    plt.show()
    plt.close()

def plotScatter(n_groups, ancNuc, desNuc,k, tableCon):
    predSeq = np.load('data/pred_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()

    tableSeq = np.load('data/predTable_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()
    
    trueSeq = np.load('data/true_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()

#     evolSeq = np.load('data/true_evol_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()
    
    lstmSeq = np.load('data/pred_lstm_context{}->{}_{}_{}.npy'.format(ancNuc,desNuc,ancName,tableCon), allow_pickle = True).item()

    true = list(trueSeq.values())
    true_context = list(trueSeq.keys())
    pred = []
    evol = []
    lstm = []
    table = []
    for i in true_context:
        pred.append(predSeq[i])
#         evol.append(evolSeq[i])
        lstm.append(lstmSeq[i])
        table.append(tableSeq[i])
        
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15,5))
    f.text(0.5, 0.0, 'Observed', ha='center', va='center', fontsize = 16)
    f.text(0.05, 0.5, 'Predicted', ha='center', va='center', rotation='vertical', fontsize = 16)
    
    ax1.scatter(true, table, color = 'y', label = 'freqTable')

#     ax1.axis('scaled')
#     ax1.axis('square')
    ax1.set_xlim([0, 1.2 * max(max(table), max(true))])
    ax1.set_ylim([0, 1.2 * max(max(table), max(true))])
    ax1.text(0.5,-0.1, "r = {}".format(stats.pearsonr(table, true)[0]), size=12, ha="center", 
                         transform=ax1.transAxes)
    ax1.set_title('xy{}zw to xy{}zw point mutation'.format(ancNuc, desNuc))
    for i, txt in enumerate(list(trueSeq.keys())):
        if i%30 == 0:
            ax1.annotate(txt, (true[i], table[i]))

    ax1.legend()
    
    
    ax2.scatter(true, lstm, color = 'k', label = 'lstm')
    for i, txt in enumerate(list(lstmSeq.keys())):
        if i%30 == 0:
            ax2.annotate(txt, (true[i], lstm[i]))

#     ax2.axis('scaled')
#     ax2.axis('square')
    ax2.set_xlim([0, 1.2 * max(max(lstm), max(true))])
    ax2.set_ylim([0, 1.2* max(max(lstm), max(true))])
    ax2.text(0.5,-0.1, "r = {}".format(stats.pearsonr(lstm, true)[0]), size=12, ha="center", 
                             transform=ax2.transAxes)
    ax2.legend()
    ax2.set_title('xy{}zw to xy{}zw point mutation'.format(ancNuc, desNuc))
    
    ax3.scatter(true, pred, color = 'm', label = 'DeepEvoLstm')

#     ax3.axis('scaled')
#     ax3.axis('square')
    ax3.set_xlim([0, 1.2 * max(max(pred), max(true))])
    ax3.set_ylim([0, 1.2 * max(max(pred), max(true))])
    ax3.text(0.5,-0.1, "r = {}".format(stats.pearsonr(pred, true)[0]), size=12, ha="center", 
                         transform=ax3.transAxes)
    ax3.set_title('xy{}zw to xy{}zw point mutation'.format(ancNuc, desNuc))
    for i, txt in enumerate(list(trueSeq.keys())):
        if i%30 == 0:
            ax3.annotate(txt, (true[i], pred[i]))

    ax3.legend()
    
    f.savefig('figures/scatter_{}_{}_{}->{}_{}.png'.format(ancName, desName,  ancNuc, desNuc,k, tableCon))
    f.show()
    plt.show()
    plt.close()
    
    return ('{} $rightarrow$ {} & {:.3f} & {:.3f} & {:.3f}'.format(ancNuc, desNuc,stats.pearsonr(pred, true)[0], stats.pearsonr(lstm, true)[0], stats.pearsonr(table, true)[0] ))
#     print(stats.pearsonr(pred, true), stats.pearsonr(lstm, true))
            

# np.save('inputAll_{}_{}'.format(ancName, desName), inputAll)
# np.save('predAll_{}_{}'.format(ancName, desName), predAll)
# np.save('outputAll_{}_{}'.format(ancName, desName), outputAll)


# In[72]:


# inputAll, inputAll2, predAll, outputAll= decodeList(inputAll, predAll, outputAll)
# lstm_inputAll, lstm_inputAll2, lstm_predAll, lstm_outputAll = decodeList(lstm_inputAll, lstm_predAll, lstm_outputAll)
def plotCombine(k, tableCon):
#     contextLen = k
#     predTable = [str(np.load('predTable_{}.npy'.format(tableCon)))]
#     ancCase = ['A','C','G','T']
#     desCase = ['A','C','G','T','-']
#     for i in tqdm_notebook(ancCase):
#         for j in desCase:
#             contextMut(contextLen, i, j, inputAll, predAll, pred = True, evol = False, tableCon= tableCon)
#             contextMut(contextLen, i, j, inputAll, predTable, pred = True, evol = False, table = 'Table', tableCon= tableCon)
#             contextMut(contextLen, i, j, inputAll, outputAll, pred = False, evol = False, tableCon= tableCon)
#             contextMut(contextLen, i, j, lstm_inputAll, lstm_predAll, pred = True, evol = False, lstm = True, tableCon= tableCon)
#     ancCase = ['-']
#     desCase = ['A','C','G','T']
#     for i in tqdm_notebook(ancCase):
#         for j in desCase:
#             contextMutInsert(contextLen, i, j, inputAll, predAll, pred = True, evol = False, tableCon= tableCon)
#             contextMutInsert(contextLen, i, j, inputAll, predTable, pred = True, evol = False, table = 'Table', tableCon= tableCon)
#             contextMutInsert(contextLen, i, j, inputAll, outputAll, pred = False, evol = False, tableCon= tableCon)
#             contextMutInsert(contextLen, i, j, lstm_inputAll, lstm_predAll, pred = True, evol = False, lstm = True, tableCon= tableCon)

#     ancCase = ['A','C','G','T','-']
#     desCase = ['A','C','G','T','-']


    values = []
    for i in tqdm_notebook(ancCase):
        for j in desCase:
            if i != j:
                values.append(plotScatter(numBin,i, j, k, tableCon))
    for item in values:
        print(item)


# In[ ]:


plotCombine(2, 1)


# In[73]:


plotCombine(2, 5)
# ancCase = ['C','A','C','T','-']
# desCase = ['T','G','A','-','T']
# # for i in ancCase:
# #     for j in desCase:
# #         if i != j:
# #             plotPointMut(numBin,i, j)



# for i, j in zip(ancCase, desCase):
#         if i != j:
#             plotScatter(numBin,i, j)


# In[43]:


plotCombine(2, 15)


# In[64]:


plotCombine(3, 1)


# In[65]:


plotCombine(3, 5)


# In[66]:


plotCombine(3, 15)


# In[67]:


plotCombine(4, 5)


# In[68]:


plotCombine(4, 15)


# In[69]:


plotCombine(5, 5)


# In[70]:


plotCombine(5, 15)


# In[ ]:


def analyzeMut(ancNuc, desNuc):
    predSeq = np.load('data/pred_context{}{}_{}.npy'.format(ancNuc,desNuc, ancName), allow_pickle = True).item()
    trueSeq = np.load('data/true_context{}{}_{}.npy'.format(ancNuc,desNuc, ancName), allow_pickle = True).item()
#     evolSeq = np.load('data/true_evol_context{}{}_{}.npy'.format(ancNuc,desNuc, ancName), allow_pickle = True).item()
    lstmSeq = np.load('data/pred_lstm_context{}{}_{}.npy'.format(ancNuc,desNuc, ancName), allow_pickle = True).item()

#     print(list(predSeq.keys())[:256])
#     print(list(trueSeq.keys())[:256])
    print(predSeq)
#     print(trueSeq)
    
analyzeMut('A','G')


# In[62]:


contextLength = 1
numBin = 16
def contextMut(size, ancNuc, desNuc, anc, des, pred, evol, lstm = False):
    cont = list(itertools.product('ACGT', repeat=size))
    cont1 = list(itertools.product('ACGT', repeat=size))
    context_dict = {}
    count_dict = {}
    for i in cont1:
        for j in cont1:
            #context_dict[(''.join(i)+'A'+''.join(j) , ''.join(i)+'G'+''.join(j))] = 0
            context_dict[(''.join(i)+ancNuc+''.join(j))] = 0
            count_dict[(''.join(i)+ancNuc+''.join(j))] = 0
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            if a[i+size:i+size+len(ancNuc)] == ancNuc :
                count_dict[(a[i:i+size*2+len(ancNuc)])] += 1
    for a,b in tqdm_notebook(zip(anc, des)):
        for i in range(len(a)-size*2-len(ancNuc)):
            if a[i+size:i+size+len(ancNuc)] == ancNuc and b[i+size+int(len(ancNuc)/2)] == desNuc:
                context_dict[(a[i:i+size*2+len(ancNuc)])] += 1
    for key in context_dict.keys():
        if count_dict[key] !=0:
            context_dict[key] = context_dict[key]/count_dict[key] 
        else :
            continue
    sorted_context = sorted(context_dict.items(), key=operator.itemgetter(1), reverse = 1)
    sorted_context = dict(sorted_context)
    if lstm ==False:
        if pred == True and evol ==False:
            np.save('data/pred_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred == True and evol == True:
            np.save('data/pred_evol_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred == False and evol ==False:
            np.save('data/true_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred == False and evol ==True:
            np.save('data/true_evol_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        return sorted_context
    elif lstm ==True :
        if pred == True:
            np.save('data/pred_lstm_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred ==False:
            np.save('data/true_lstm_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
    return sorted_context
            
def contextMutInsert(size, ancNuc, desNuc, anc, des, pred, evol, lstm = False):
    cont1 = list(itertools.product('ACGT', repeat=size-1))
    cont = list(itertools.product('ACGT', repeat=size))
    context_dict = {}
    count_dict = {}
    for i in cont:
        for j in cont:
            #context_dict[(''.join(i)+'A'+''.join(j) , ''.join(i)+'G'+''.join(j))] = 0
            context_dict[(''.join(i)+ancNuc+''.join(j))] = 0
            count_dict[(''.join(i)+ancNuc+''.join(j))] = 0
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            count_dict[(a[i:i+size]+ancNuc+a[i+size+len(ancNuc):i+size*2+len(ancNuc)])] += 1
    for a,b in zip(anc, des):
        for i in range(len(a)-size*2-len(ancNuc)):
            inserted_nuc = inv_dict[b[i+size+int(len(ancNuc)/2)-1]]
            if len(inserted_nuc) >1 and inserted_nuc[1] == desNuc:
                context_dict[(a[i:i+size]+ancNuc+a[i+size+len(ancNuc):i+size*2+len(ancNuc)])] += 1
    for key in context_dict.keys():
        if count_dict[key] !=0:
            context_dict[key] = context_dict[key]/count_dict[key] 
        else :
            continue
    sorted_context = sorted(context_dict.items(), key=operator.itemgetter(1), reverse = 1)
    sorted_context = dict(sorted_context)
    if lstm ==False:
        if pred == True and evol ==False:
            np.save('data/pred_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred == True and evol == True:
            np.save('data/pred_evol_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred == False and evol ==False:
            np.save('data/true_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred == False and evol ==True:
            np.save('data/true_evol_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        return sorted_context
    elif lstm ==True :
        if pred == True:
            np.save('data/pred_lstm_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
        elif pred ==False:
            np.save('data/true_lstm_context{}->{}_{}.npy'.format(ancNuc,desNuc,ancName), sorted_context)
    return sorted_context
            
def calculateR(contextLen, ancCase, desCase):
    sorted_context = contextMut(contextLen, ancCase, desCase, inputAll, predAll, pred = True, evol = False)
    contextMut(contextLen, ancCase, desCase, inputAll, outputAll, pred = False, evol = False)
    contextMut(contextLen, ancCase, desCase, lstm_inputAll, lstm_predAll, pred = True, evol = False, lstm = True)

    sorted_word = list(sorted_context.keys())
    top = sorted_word[0]
    mid = sorted_word[int(len(sorted_word)/2)]
    bot = sorted_word[-1]
    return top, mid, bot

def calculateRInsert(contextLen, ancCase, desCase):
    sorted_context = contextMutInsert(contextLen, ancCase, desCase, inputAll, predAll, pred = True, evol = False)
    contextMutInsert(contextLen, ancCase, desCase, inputAll, outputAll, pred = False, evol = False)
    contextMutInsert(contextLen, ancCase, desCase, lstm_inputAll, lstm_predAll, pred = True, evol = False, lstm = True)

    sorted_word = list(sorted_context.keys())
    top = sorted_word[0]
    mid = sorted_word[int(len(sorted_word)/2)]
    bot = sorted_word[-1]
    return top, mid, bot
    
def plotContextChange(contLen, ancCase, desCase, insert = False):
    contextLen = contLen
    if insert == True:
        top, mid, bot = calculateRInsert(contextLen, ancCase, desCase)
        
        tt, tm, tb = calculateRInsert(contextLen, top, desCase)
        mt, mm, mb = calculateRInsert(contextLen, mid, desCase)
        bt, bm, bb = calculateRInsert(contextLen, bot, desCase)

        ttt, ttm, ttb = calculateRInsert(contextLen, tt, desCase)
        mmt, mmm, mmb = calculateRInsert(contextLen, mm, desCase)
        bbt, bbm, bbb = calculateRInsert(contextLen, bb, desCase)
    else :
        top, mid, bot = calculateR(contextLen, ancCase, desCase)
        tt, tm, tb = calculateR(contextLen, top, desCase)
        mt, mm, mb = calculateR(contextLen, mid, desCase)
        bt, bm, bb = calculateR(contextLen, bot, desCase)

        ttt, ttm, ttb = calculateR(contextLen, tt, desCase)
        mmt, mmm, mmb = calculateR(contextLen, mm, desCase)
        bbt, bbm, bbb = calculateR(contextLen, bb, desCase)
    
    
    topList = [ancCase, top, tt]
    midList = [ancCase, mid, mm]
    botList = [ancCase, bot, bb]
    ancWords = [topList, midList, botList]
    

    print(topList)
    print(midList)
    print(botList)
    print(ancWords)
#     topList =['A', 'CAT', 'CCATG']
#     midList =['A', 'AAC', 'AAACT']
#     botList =['A', 'GAA', 'TGAAA']
#     ancWords =[['A', 'CAT', 'CCATG'], ['A', 'AAC', 'AAACT'], ['A', 'GAA', 'TGAAA']]
    
    f, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20,20))
    axs = axs.flatten()
    f.text(0.5, 0.08, 'Observed', ha='center', va='center', fontsize=40)
    f.text(0.05, 0.5, 'Predicted', ha='center', va='center', rotation='vertical', fontsize=40)
    f.suptitle('Effect of adding flanking base contexts to {}->{} mutation'.format(ancCase, desCase), fontsize =30)

    index = 0
    for words in tqdm(ancWords):
        for j in desCase:
            for n, i in enumerate(words):
                if i != j:
                    predSeq = np.load('data/pred_context{}->{}_{}.npy'.format(i,j,ancName), allow_pickle = True).item()
                    trueSeq = np.load('data/true_context{}->{}_{}.npy'.format(i,j,ancName), allow_pickle = True).item()
                    true = list(trueSeq.values())
                    true_context = list(trueSeq.keys())
                    pred = []
                    for x in true_context:
                        pred.append(predSeq[x])
                    axs[index].scatter(true, pred, color = 'm')
#                     axs[index].axis('scaled')
#                     axs[index].axis('square')
                    axs[index].set_xlim([0, 1.1 * max(max(pred), max(true))])
                    axs[index].set_ylim([0, 1.1 * max(max(pred), max(true))])
                    axs[index].text(0.5,-0.1, "r = {}".format(stats.pearsonr(pred, true)[0]), size=20, ha="center", 
                             transform=axs[index].transAxes)
                    axs[index].set_title('x{}y to {} point mutation'.format(i, j), fontsize=25)
                    for i, txt in enumerate(list(predSeq.keys())):
                        if i %3 ==0 or i%(int(len(true)/2))==0 or i%(int(len(true)-1))==0:
                            axs[index].annotate(txt, (true[i], pred[i]))

                    index += 1
                    f.savefig('figures/scatter_conChange_{}_{}_{}->{}.png'.format(ancName, desName,  ancCase, desCase))
#                     print('pearson corr: ', stats.pearsonr(pred, true)[0])
    plt.show()
    plt.close()
#                     plotPointMut(numBin,i, j)
#                     plotScatter(numBin,i, j)
#     analyzeMut(ancCase[0],desCase[0])


# In[63]:


plotContextChange(1, 'A', 'G', insert = False)
plotContextChange(1, 'C', 'T', insert = False)
plotContextChange(1, 'A', '-', insert = False)
plotContextChange(1, '-', 'T', insert = True)


# In[ ]:


analyzeMut('TAG','G')
contextMut(contextLen, 'TAG', 'G', inputAll, predAll, pred = True, evol = False)

