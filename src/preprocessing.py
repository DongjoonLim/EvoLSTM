from sklearn.preprocessing import LabelEncoder
from bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re

def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
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
oldlist = readmaf(1200, 2000, "data/chr1.anc.maf")
print(dir(oldlist[1]))
print(type(oldlist[1][0].id))
#print(oldlist[2])

#alignList stores alignments between direct ancestor and descendent
def getAlign(inputList, keyword):
    alignList = []
    for k in range(len(keyword)):
        for i in range(len(inputList)):
            for j in range(len(inputList[i])):
                if j<len(inputList[i])-1 and (keyword[k] in inputList[i][j].id and keyword[k] in inputList[i][j+1].id):
                    alignment = MultipleSeqAlignment([
                            SeqRecord(inputList[i][j].seq, id=inputList[i][j].id),
                            SeqRecord(inputList[i][j+1].seq, id=inputList[i][j+1].id)
                        ]
                    )
                    alignList.append(alignment)
    return alignList
     
keyword = ["_HP","_RM","_FC","_CS","_BO","_PB","_TO","_VC", "_PP"]
alignList = getAlign(oldlist, keyword)
print(len(alignList))
print(alignList[62])
print(alignList[62][1])
print(len(alignList[62][1]))

label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))
encoder = OneHotEncoder(sparse=False, dtype=int, n_values=5)
def one_hot_encoder(my_array, encoder):
    integer_encoded = label_encoder.transform(my_array)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded

def alignBoth(alignList):
    h, w = len(alignList), 2
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(alignList)):
        for j in range(len(alignList[i])):
            Matrix[i][j] = one_hot_encoder(string_to_array(str(alignList[i][j].seq)), encoder)
    return Matrix

one_hot_aligned = alignBoth(alignList)
one_hot_aligned = np.array(one_hot_aligned)
print(type(one_hot_aligned))
print(len(one_hot_aligned[62][1]))
print(one_hot_aligned[62][1])
# A = [1 0 0 0], C = [0 1 0 0], G = [0 0 1 0], T = [0 0 0 1]



#print(one_hot_aligned)
