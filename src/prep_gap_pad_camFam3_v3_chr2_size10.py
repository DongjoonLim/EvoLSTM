#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.preprocessing import LabelEncoder
from bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing import sequence
import sklearn
import numpy as np
import re


# In[ ]:


seq_length = 10
pad_length = 5
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt0]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

def readmaf(start, end, filename):
    count = 0
    oldlist =[]
    align = AlignIO.parse(filename, "maf")
    for multiple_alignment in align:
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


# In[ ]:


print(len(oldlist))


# In[ ]:


#alignList stores alignments between direct ancestor and descendent
def getAlign(inputList, keyword):
    alignList = []
    zeros = ''
    for i in range(len(inputList)):
        # index for the dog
        indexDes = -1
        indexAnc = -1

        for j in range(len(inputList[i])):
            if len(inputList[i][j])>=10 and 'canFam3' in inputList[i][j].id  :
                indexDes = j
            elif len(inputList[i][j])>=10 and inputList[i][j].id == '_CMAOL': #'_RMPC' :
                indexAnc = j
        if indexDes!=-1 and indexAnc!=-1:
            alignment = MultipleSeqAlignment([
                    SeqRecord(inputList[i][indexDes].seq+zeros, id=inputList[i][indexDes].id),
                    SeqRecord(inputList[i][indexAnc].seq+zeros, id=inputList[i][indexAnc].id),
                    #SeqRecord(zeros+inputList[i][indexDes].seq, id=inputList[i][indexDes].id)
                ]
            )
            alignList.append(alignment)
    return alignList

def zeros(length):
    return "0"*length
    
FinalList = []    
anc = []
des = []
keyword = ["_HP","_RM","_FC","_CS","_BO","_PB","_TO","_VC", "_PP"]
alignList = getAlign(oldlist, keyword)
print(len(alignList))
print(alignList[126])
print(len(alignList[126][1].seq))
print(len(alignList[126][1]))
for i in range (100):
    print('printing new alignment -------------------------------------------\n')
    print(alignList[i])

#print(str(alignList[62][1]))
#print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[ ]:


label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','0', 'z']))
encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(6)])

def ungap(strX, strY):
    new_X = ''
    new_Y = ''
    for i in range(len(strX)):
        if strX[i] == '-' and strY[i] == '-':
            continue
        else :
            new_X = new_X+strX[i]
            new_Y = new_Y+strY[i]

    return new_X, new_Y
    

def getXY(alignList):
    listX = []
    listY = []
    for i in range(len(alignList)):
        X, Y = ungap(str(alignList[i][1].seq).upper(), str(alignList[i][0].seq).upper())
        listX.append(X)
        listY.append(Y)
    return listX, listY

def zeroPad(A):
    zeros = '0'*(pad_length+seq_length-len(A))
    A = A+zeros
    return A

def tokenize(listX, listY):
    X_token =[]
    X1_token = []
    Y_token =[]
    Y1_token = []
    for i in range(len(listX)):
        if len(listX[i])==seq_length:
            X_token.append(listX[i])
            Y_token.append(listY[i])
        elif len(listX[i])>seq_length:
            for j in range(len(listX[i])-seq_length):
                X = listX[i][j:j+seq_length].replace('-','')
                Y = listY[i][j:j+seq_length]
                X = zeroPad(X)
                Y = zeroPad(Y)
                X1 = '0'+X[:seq_length+pad_length-1]
                Y1 = '0'+Y[:seq_length+pad_length-1]
                X_token.append(X)
                Y_token.append(Y)
                X1_token.append(X1)
                Y1_token.append(Y1)
    return X_token, Y_token, X1_token, Y1_token

def one_hot_encoder(my_array, encoder):
    integer_encoded = label_encoder.transform(my_array)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = encoder.fit_transform(integer_encoded)
    #print(np.array(onehot_encoded))
    return onehot_encoded

def encode_list(X, Y, encoder):
    new_X =[]
    new_Y =[]
    for x, y in zip(X,Y):
        #new_X = np.append(new_X, one_hot_encoder(string_to_array(x), encoder))
        #new_Y = np.append(new_Y, one_hot_encoder(string_to_array(y), encoder))
        new_X.extend(one_hot_encoder(string_to_array(x), encoder))
        new_Y.extend(one_hot_encoder(string_to_array(y), encoder))
    
    return np.array(new_X).reshape((-1,seq_length+pad_length,6)), np.array(new_Y).reshape((-1,seq_length+pad_length,6))

X, Y=getXY(alignList)
X, Y, X1, Y1= tokenize(X,Y)
#print(X,Y)
X, Y= encode_list(X, Y, encoder)
X1, Y1= encode_list(X1, Y1, encoder)
for i in range(100):
    print(X[i], Y[i])

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
    
#print(Y.shape)
#print(len(X))


# In[ ]:


pt1 = 9900000
pt2 = 9950000
pt3 = 10000000
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

np.save('prepData/X_train_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), X_train)
np.save('prepData/X_val_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), X_val)
np.save('prepData/X_test_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), X_test)
np.save('prepData/y_train_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), y_train)
np.save('prepData/y_val_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), y_val)
np.save('prepData/y_test_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), y_test)

np.save('prepData/y_train1_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), y_train1)
np.save('prepData/y_val1_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), y_val1)
np.save('prepData/y_test1_gap_pad_camFam3_v3_chr2_size{}'.format(seq_length), y_test1)


# In[ ]:



X = np.array(X)
Y = np.array(Y)
#X1 = np.array(X1)
Y1 = np.array(Y1)
print(X)
print(X.shape)
print(X[0].shape)
print(X[0][0].shape)
print(type(X))
#print(X)


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

