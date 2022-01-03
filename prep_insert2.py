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
import string
from tqdm import tqdm
import sys

seq_length = 15
chromosome = sys.argv[1]
ancestor = sys.argv[2]
descendant = sys.argv[3]
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
oldlist = readmaf(0, 1000000, "data/{}.anc.maf".format(chromosome))
print(oldlist[1][1])
print(dir(oldlist[1]))
print(oldlist[1][0].id)

# ancSeq = ['_HPGPNRMPC', '_HPGPNRMPCCS', '_HPGPNRMPCCSO']
# desSeq = ['hg38']
ancSeq = [ancestor]
desSeq = [descendant]

alphaDict = np.load('mut_dict_insert2.npy',allow_pickle=True).item()
print(alphaDict)
np.save('mut_dict_insert2_chr3',alphaDict)

def ungap(anc, des):
    a = ''
    d = ''
    for i in range(len(anc)):
        if anc[i] == '-' and des[i] =='-':
            continue
        else:
            a = a + anc[i]
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
        if anc[i] != '-' and anc[i+1] == '-' and anc[i+2] == '-' and anc[i+3] != '-' and des[i+2] != '-' and des[i+1] != '-' and des[i] != '-':
            newAnc = newAnc + anc[i]
            newDes = newDes + alphaDict[des[i:i+3]]
            i = i+3
        elif anc[i] != '-' and anc[i+1] == '-' and anc[i+2] != '-' and des[i+1] != '-' and des[i] != '-':
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
    for i in range(len(anc)):
        if anc[i] == '-':
            continue
        else:
            newAnc = newAnc + anc[i]
            newDes = newDes + des[i]
    return [newAnc, newDes]
            
            
            
# anc, des = ungap(anc, des)
# anc, des = coded(anc, des)
# anc, des = ungapAnc(anc, des)


# for i in range(20):
#     print(anc[i*100:(i+1)*100])
#     print(des[i*100:(i+1)*100])
#     print()
    
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
            
            for i in range(20):
                print(anc[i*100:(i+1)*100])
                print(des[i*100:(i+1)*100])
                print()
            
            np.save('prepData/insert2Anc_{}_{}_{}'.format(a,b, chromosome), anc)
            np.save('prepData/insert2Des_{}_{}_{}'.format(a,b, chromosome), des)
            alignDict[(a,b)] = (anc,des)
    return alignDict           

getAlign(oldlist, ancSeq, desSeq)
