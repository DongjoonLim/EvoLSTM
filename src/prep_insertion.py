#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm import tqdm


# In[2]:


seq_length = 11
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt0]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

def readmaf(start, end, filename):
    count = 0
    oldlist =[]
    align = AlignIO.parse(filename, "maf")

    for multiple_alignment in tqdm(align):
        count = count+1
        if count >start and count<=end:
            oldlist.append(multiple_alignment)
        elif count>end:
            break
    return oldlist

#oldlist stores MSAs
oldlist = readmaf(0, 1000000, "data/chr2.anc.maf")
print(oldlist[1][1])
print(dir(oldlist[1]))
print(oldlist[1][0].id)
#print(oldlist[2])


# In[3]:


print(len(oldlist))


# In[4]:


#alignList stores alignments between direct ancestor and descendent
ancSeq = ['_HP', '_HPG', '_HPGP', '_HPGPN', '_HPGPNRMPC']
desSeq = ['hg38']

# def getAlign(inputList):
#     alignList = []
#     zeros = ''
#     anc = ''
#     des = ''
#     for i in range(len(inputList)):
#         # index for the dog
#         indexDes = -1
#         indexAnc = -1

#         for j in range(len(inputList[i])):
#             if len(inputList[i][j])>=10 and 'canFam3' in inputList[i][j].id  :
#                 indexDes = j
#             elif len(inputList[i][j])>=10 and inputList[i][j].id == '_CMAOL': #'_RMPC' :
#                 indexAnc = j
#         if indexDes!=-1 and indexAnc!=-1:
#             des = des+str(inputList[i][indexDes].seq)
#             anc = anc+str(inputList[i][indexAnc].seq)

#     return anc, des


# anc, des = getAlign(oldlist)
# print(anc[:100])
# print(des[:100])
# print(anc[7]=='-')


# In[ ]:


keys = []
alphabet = string.ascii_letters
alphabet = alphabet.replace('A','').replace('C', '').replace('G', '').replace('T', '').replace('-', '')
print(alphabet)
#print(random.sample(string.ascii_letters, 5))
a = 'ACTG'
for output in itertools.product(a, repeat=2):
    keys.append(''.join(output))
value = random.sample(alphabet, len(keys))
alphaDict = {'A' : 'A', 'C' :'C', 'G':'G','T':'T', '-':'-'}
for k, v in zip(keys, value):
    alphaDict[k] = v
    
print(alphaDict)
np.save('mut_dict',alphaDict)

def ungap(anc, des):
    a = ''
    d = ''
    for i in tqdm(range(len(anc))):
        if anc[i] == '-' and des[i] =='-':
            continue
        else:
            a = a+ anc[i]
            d = d + des[i]
            
    return a, d

def coded(anc, des):
    newAnc = ''
    newDes = ''
    if len(anc) != len(des):
        print('Lenghts mismatch!')
        return
    i = 0
    while i < len(anc)-1:

        if anc[i] != '-' and anc[i+1] == '-' and anc[i+2] != '-' and des[i+1] != '-' and des[i] != '-':
            newAnc = newAnc + anc[i]
            newDes = newDes + alphaDict[des[i:i+2]]
            i = i+2
        else:
            newAnc = newAnc + anc[i]
            newDes = newDes + des[i]
            i= i+1

    return [newAnc, newDes]

def ungapAnc(anc, des):
    newAnc = ''
    newDes = ''
    for i in tqdm(range(len(anc))):
        if anc[i] == '-':
            continue
        else:
            newAnc = newAnc + anc[i]
            newDes = newDes + des[i]
    return [newAnc, newDes]
            
def getAlign(inputList, ancSeq, desSeq):
    alignDict = {}
    nucSet = set(['A','C','G','T','-'])
    for a in tqdm(ancSeq):
        for b in desSeq:
            print(a,b)
            alignList = []
            zeros = ''
            anc = ''
            des = ''
            for i in range(len(inputList)):
                # index for the dog
                indexDes = -1
                indexAnc = -1
                for j in range(len(inputList[i])):
                    if set(inputList[i][j].seq.upper()).issubset(nucSet) and (b in inputList[i][j].id ) :
                        indexDes = j
                    elif set(inputList[i][j].seq.upper()).issubset(nucSet) and (inputList[i][j].id == a): #'_RMPC' :
                        indexAnc = j
                if indexDes!=-1 and indexAnc!=-1:
                    des = des+str(inputList[i][indexDes].seq)
                    anc = anc+str(inputList[i][indexAnc].seq)
            anc = anc.upper()
            des = des.upper()
            anc, des = ungap(anc, des)
            anc, des = coded(anc, des)
            anc, des = ungapAnc(anc, des)
            
            np.save('prepData/insert1Anc_{}_{}'.format(a,b), anc)
            np.save('prepData/insert1Des_{}_{}'.format(a,b), des)
            alignDict[(a,b)] = (anc,des)
    return alignDict           

getAlign(oldlist, ancSeq, desSeq)


# In[6]:


def oneEncode(input):
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(input)

    #one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False)
    #reshape because that's what OneHotEncoder likes
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    return onehot_encoded_seq

oneEncode(np.array(list(des)))


# In[ ]:


def tokenize(anc):
    token = []
    for i in range(len(anc)-seq_length):
        token.append(list(anc[i:i+seq_length]))
    return token


# with open('ancestral', 'wb') as fp:
#     pickle.dump(anc, fp)
    
# with open('descendant', 'wb') as f:
#     pickle.dump(des, f)


# In[ ]:


for i in range(1000):
    print(''.join(anc[i]), ''.join(des[i]))
    if ''.join(anc[i]) == ''.join(des[i]):
        print(True)


# In[ ]:


label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','0', 'z']))
encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(6)])
def one_hot_encoder(my_array, encoder):
    integer_encoded = label_encoder.transform(my_array)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = encoder.fit_transform(integer_encoded)
    #onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded, encoder

def alignBoth(alignList):
    zeros = '00000'
    h, w = len(alignList), 2;
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(alignList)):
        for j in range(len(alignList[i])):
            Matrix[i][j] = one_hot_encoder(string_to_array(str(alignList[i][j].seq).upper()), encoder)
    return Matrix

# [Index of alignment][1 for Ancester, 0 for Descendant][Always 0]
one_hot_aligned = alignBoth(alignList)
#one_hot_aligned = one_hot_aligned[:][:][0]
one_hot_aligned = np.array(one_hot_aligned)
#print(one_hot_aligned.shape)
#ignorethis, encoder = one_hot_encoder(string_to_array(str(alignList[1][1].seq)),encoder)


print(type(one_hot_aligned))
print(len(one_hot_aligned[126][1]))
print(one_hot_aligned[126][1][0])
#print(encoder.inverse_transform(one_hot_aligned[126][1]))
print(type(one_hot_aligned[126][1]))
# A = [1 0 0 0], C = [0 1 0 0], G = [0 0 1 0], T = [0 0 0 1]
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

def decoderY(array):
    result = ""    
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
    
print(decoder(one_hot_aligned[126][1][0]))
print(len(one_hot_aligned))


# In[ ]:


def vectorize(char_setx, char_sety, sequence_length):
    dataX = []
    dataY = []
    dataX1 = []
    dataY1 = []
    result=[]
    char_setx, char_sety = mutation_with_gap(char_setx, char_sety)
    #print(type(char_sety[0]))
    for i in range(len(char_setx) - sequence_length):
        
        '''for i in char_sety[i: i + sequence_length-1]:
            y1.append(i)'''
        #x1 = char_setx1[i: i + sequence_length]
        x = char_setx[i:i + sequence_length]
        y = char_sety[i: i + sequence_length]
        y1 = [[1, 0, 0, 0, 0, 0]]
        temp = y[:-1]
        #temp = temp.tolist()
        for i in temp:
            y1.append(i)
        y1=np.array(y1)
        dataX.append(x)
        dataY.append(y)
        dataY1.append(y1)

    return dataX, dataY, dataY1

def tokenize(one_hot_aligned, sequence_length):
    X = []
    Y = []
    X1 = []
    Y1 = []
    for i in range(len(one_hot_aligned)):
        dataX, dataY, dataY1 = vectorize(one_hot_aligned[i][1][0], one_hot_aligned[i][0][0],  sequence_length)
        X.extend(dataX)
        Y.extend(dataY)
        #X1.extend(dataX1)
        Y1.extend(dataY1)
    return X, Y, Y1

def truncate(one_hot_aligned):
    X = []
    Y = []
    for i in range(len(one_hot_aligned)):
        X.append(one_hot_aligned[i][1][0])
        Y.append(one_hot_aligned[i][0][0])
    X = sequence.pad_sequences(X,
                                 maxlen=30,
                                 truncating='post',
                                 padding='post',
                                 value=0)
    Y = sequence.pad_sequences(Y,
                                 maxlen=30,
                                 truncating='post',
                                 padding='post',
                                 value=0)
    return X,Y

# function for checking if there are at least one mutation and there are no gaps.
def mutation_with_nogap(a, b):
    a = a.tolist()
    b = b.tolist()
    if a.count([0,0,0,0,0,1])>0 or b.count([0,0,0,0,0,1])>0:
        return False
    else:
        return True
    
def mutation_with_gap(a, b):
    a = a.tolist()
    b = b.tolist()
    a_new = []
    b_new = []
    for i, j in zip(a,b):
        if i == [0,0,0,0,0,1] and j == [0,0,0,0,0,1]:
            continue
        else:
            a_new.append(i)
            b_new.append(j)
    return a_new, b_new
        


def diffList(a, b):
    count = 0
    length = len(a)
    for i in range(length):
        if a[i] != b[i]:
            count = count+1
    return count

def deleteGap(one_hot_aligned):
    result = one_hot_aligned
    for i in range(len(one_hot_aligned)):
        if one_hot_aligned[i][1][0] == '-' and one_hot_aligned[i][0][0] == '-':
            result.pop(i)
    return result

# Used for non 0 padding
#one_hot_aligned = deleteGap(one_hot_aligned)
X, Y, Y1=tokenize(one_hot_aligned, seq_length)

# Used for 0 padding
#X, Y = truncate(one_hot_aligned)
X = np.array(X)
Y = np.array(Y)
#X1 = np.array(X1)
Y1 = np.array(Y1)

print(Y.shape)
print(len(X))
pt1 = 9600000
pt2 = 9650000
pt3 = 9700000
X_train = X[:pt1]
X_val = X[pt1:pt2]
X_test = X[pt2:pt3]
y_train = Y[:pt1]
y_val =  Y[pt1:pt2]
y_test = Y[pt2:pt3]
# X_train1 = X1[:pt1]
# X_val1 = X1[pt1:pt2]
# X_test1 = X1[pt2:pt3]
y_train1 = Y1[:pt1]
y_val1 =  Y1[pt1:pt2]
y_test1 = Y1[pt2:pt3]

# X_train = X[:20000]
# X_val = X[20000:23000]
# X_test = X[23000:26000]
# y_train = Y[:20000]
# y_val =  Y[20000:23000]
# y_test = Y[23000:26000]

np.save('prepData/X_train_camFam3_v3_chr2_size11', X_train)
np.save('prepData/X_val_camFam3_v3_chr2_size11', X_val)
np.save('prepData/X_test_camFam3_v3_chr2_size11', X_test)
np.save('prepData/y_train_camFam3_v3_chr2_size11', y_train)
np.save('prepData/y_val_camFam3_v3_chr2_size11', y_val)
np.save('prepData/y_test_camFam3_v3_chr2_size11', y_test)

np.save('prepData/y_train1_camFam3_v3_chr2_size11', y_train1)
np.save('prepData/y_val1_camFam3_v3_chr2_size11', y_val1)
np.save('prepData/y_test1_camFam3_v3_chr2_size11', y_test1)

# np.save('prepData/X_train1_camFam3_1mutOnly_v2', X_train1)
# np.save('prepData/X_val1_camFam3_1mutOnly_v2', X_val1)
# np.save('prepData/X_test1_camFam3_1mutOnly_v2', X_test1)


# np.save('prepData20/X_train_camFam3_1mutOnly', X_train)
# np.save('prepData20/X_val_camFam3_1mutOnly', X_val)
# np.save('prepData20/X_test_camFam3_1mutOnly', X_test)
# np.save('prepData20/y_train_camFam3_1mutOnly', y_train)
# np.save('prepData20/y_val_camFam3_1mutOnly', y_val)
# np.save('prepData20/y_test_camFam3_1mutOnly', y_test)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(X_train[0])
print(y_train[0])
print(y_train1[0])


# In[ ]:


# import difflib
# sm=difflib.SequenceMatcher(None,X_test,X_train)
# sm2=difflib.SequenceMatcher(None,y_test,y_train)
for i in range(100):
    print(i, decoder(X[i]))


# In[ ]:


for i in range(100):
    print(i, decoder(Y[i]))


# In[ ]:


for i in range(100):
    print(i, decoder(Y1[i]))

