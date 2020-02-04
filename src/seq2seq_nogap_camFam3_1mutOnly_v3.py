
# coding: utf-8

# In[1]:


from keras.utils.vis_utils import plot_model
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, CuDNNLSTM, GRU, Bidirectional, Input, CuDNNGRU
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
from attention_decoder import AttentionDecoder
from keras import backend as K
from keras.models import Model
from keras.layers.core import Dense, Reshape
from keras.layers.wrappers import TimeDistributed
import difflib
from keras.models import load_model
import keras

K.clear_session()
keras.backend.clear_session()

X_train=np.load('prepData/X_train_camFam3_1mutOnly_v3_chr2.npy')
X_val=np.load('prepData/X_val_camFam3_1mutOnly_v3_chr2.npy')
X_test=np.load('prepData/X_test_camFam3_1mutOnly_v3_chr2.npy')
y_train=np.load('prepData/y_train_camFam3_1mutOnly_v3_chr2.npy')
y_val=np.load('prepData/y_val_camFam3_1mutOnly_v3_chr2.npy')
y_test=np.load('prepData/y_test_camFam3_1mutOnly_v3_chr2.npy')

y_train1 = np.load('prepData/y_train1_camFam3_1mutOnly_v3_chr2.npy')
y_val1 = np.load('prepData/y_val1_camFam3_1mutOnly_v3_chr2.npy')
y_test1 = np.load('prepData/y_test1_camFam3_1mutOnly_v3_chr2.npy')

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

        index = np.argmax(prediction)
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



# In[2]:


batch_size = 1000  # Batch size for training.
epochs = 45  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

def lstm_model():
    encoder_inputs = Input(shape=(None, 6))
    encoder = CuDNNLSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, 6))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(6, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
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

def gru_model():
    encoder_inputs = Input(shape=(None, 6))
    encoder = CuDNNGRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(None, 6))
    decoder_gru = CuDNNGRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(6, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    # inference
    encoder_model = Model(encoder_inputs, state_h)

    decoder_state_input_h = Input(shape=(latent_dim,))
    #decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = decoder_state_input_h
    decoder_outputs, state_h = decoder_gru(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = state_h
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model

model1, encoder_model1, decoder_model1 = lstm_model()
model1.fit([X_train, y_train1], y_train,
          batch_size=1000,
          epochs=epochs,
          validation_data=([X_val,y_val1], y_val),
          verbose = 1
         )

model2, encoder_model2, decoder_model2 = lstm_model()
model2.fit([X_train, y_train1], y_train,
          batch_size=1000,
          epochs=150,
          validation_data=([X_val,y_val1], y_val),
          verbose = 1
         )
'''
model2, inference_model2 = gru_model()
model2.fit([X_train, y_train1], y_train,
          batch_size=1000,
          epochs=45,
          validation_data=([X_val,y_val1], y_val),
          verbose = 1
         )
'''


# In[27]:


# Save model
#model.save('s2s.h5')

def decode_sequence(input_seq, model, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 6))
    target_seq[0][0]= np.array([1,0,0,0,0,0])
    #print(target_seq)
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
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
        states_value = [h, c]

    return decoded_sentence

#for seq_index in range(1):
def predict2(X_test, y_test, model, encoder_model, decoder_model):
    x_true =[]
    y_hat =[]
    y_true =[]
    for seq_index in range(len(X_test)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = X_test[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, model, encoder_model, decoder_model)
        input_sen = decoder(input_seq[0])
        #print('-')
        print(input_sen, ' -> ', decoded_sentence, 'True:', decoder(y_test[seq_index]), printHitMiss(decoded_sentence, decoder(y_test[seq_index])))
        x_true.append(input_sen)
        y_hat.append(decoded_sentence)
        y_true.append(decoder(y_test[seq_index]))
    print("Percentage of target and prediction being identical: {}".format(accuracy(y_hat, y_true)))
    print("Percentage of training and prediction being identical: {}".format(accuracy(y_hat, x_true)))
    print("Accuracy given mutation happened : {}".format(accuracy2(x_true, y_hat, y_true)))
    return x_true, y_hat, y_true


# In[29]:


#model1.summary()
#model2.summary()

#encoder_model1.summary()
#decoder_model1.summary()


# In[28]:


x_true, y_hat, y_true = predict2(X_test, y_test, model1, encoder_model1, decoder_model1)


# In[22]:


x_true2, y_hat2, y_true2 = predict2(X_test, y_test, model2, encoder_model2, decoder_model2)


# ## Finding ratio of mutation on predicted sequences

# In[7]:


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
    print('Ratio of transition/transversion : {}'.format(transition/transversion))
    print('Percentage of mutation : {}'.format(numMutation/(10*(len(X_test)))))
    print('Percentage of transition predicted correct : {}'.format(transitionCorrect/transition))
    print('Percentage of transversion predicted correct : {}'.format(transversionCorrect/transversion))

#countMutation(x_true6, y_hat6, y_true6)
#countMutation(x_true8, y_hat8, y_true8)


# ## Mutation Ratio

# In[8]:


countMutation(x_true, y_hat, y_true)


# In[9]:


countMutation(x_true2, y_hat2, y_true2)


# In[10]:


countMutation(x_true, y_true, y_true)


# In[30]:


X_test5=np.load('prepData/X_test_camFam3_1mutOnly_v3.npy')
y_test5=np.load('prepData/y_test_camFam3_1mutOnly_v3.npy')
x_true5, y_hat5, y_true5 = predict2(X_test5, y_test5, model1, encoder_model1, decoder_model1)


# In[31]:


x_true6, y_hat6, y_true6 = predict2(X_test5, y_test5, model2, encoder_model2, decoder_model2)


# In[13]:


countMutation(x_true5, y_hat5, y_true5)
countMutation(x_true6, y_hat6, y_true6)
countMutation(x_true5, y_true5, y_true5)


# In[14]:


#K.clear_session()

