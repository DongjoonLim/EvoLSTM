import os
import difflib
import matplotlib.pyplot as plt
import random
from random import choice
import re
import itertools
from sklearn.preprocessing import LabelEncoder
from bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import numpy as np
import re
import pickle
import pandas as pd

mutList = []
mutList2 = []
mutDict = {}
seqLength = 15
a = 'ACTG'
for output in itertools.product(a, repeat=seqLength):
    mutList.append(''.join(output))
    mutList2.append(''.join(output))

for i in range(len(mutList)):
    for j in range(len(mutList2)):
        mutDict[mutList[i], mutList2[j]] = 0

#print(mutDict)
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

mutSum = [0] * len(mutList)

keyword = ["_HP","_RM","_FC","_CS","_BO","_PB","_TO","_VC", "_PP"]
alignList = getAlign(oldlist, keyword)
print(len(alignList))
#print(alignList[126])
for i in range (len(alignList)):
    for j in range(len(str(alignList[i][1].seq))-seqLength+1):
        print('printing new alignment -------------------------------------------\n')
        ancest = str(alignList[i][1][j:j+5].seq).upper()
        descent = str(alignList[i][0][j:j+5].seq).upper()
        for k in range(len(mutList)):
            if mutList[k].upper() == ancest:
                mutSum[k] = mutSum[k]+1
        if ('-' not in ancest) and ('-' not in descent):
            mutDict[ancest, descent] = mutDict[ancest, descent] +1

print(mutDict)
print(mutSum)
with open('mutDict.txt', 'wb') as f:
    pickle.dump(mutDict, f)
with open('mutSum.txt', 'wb') as f:
    pickle.dump(mutSum, f)
