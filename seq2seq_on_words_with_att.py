from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense,concatenate,GRU, merge,multiply,Embedding, Bidirectional
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import re
import random
#константы
max_in_len=4 #максимальная длина входной последовательности
max_out_len=max_in_len+2
alphabet={'a':'c','b':'d'}
max_examples=100000
lstm_dim=2
#описание модели
input_token_index=dict([(char,i) for i,char in enumerate(list(alphabet.keys()))])
target_token_index=dict([(char,i) for i,char in enumerate(list(alphabet.values())+['\t','\n'])])
inputs=Input(shape=(None,len(input_token_index)))
encoder,state_h,state_c=LSTM(lstm_dim,return_sequences=True,return_state=True, go_backwards=True)(inputs)
encoder_states=[state_h,state_c]
decoder_inputs=Input(shape=(None,len(target_token_index)))
decoder_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True)
decoder, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
outputs=Dense(units=len(target_token_index),activation='softmax')(decoder)
model=Model([inputs,decoder_inputs],outputs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#подготовка данных для обучения и теста
examples={}
for x in range(max_examples*2):
    in_str=''
    for i in range(max_in_len):
        in_str=in_str+random.choice(list(alphabet.keys()))
    out_str=in_str
    for a,b in alphabet.items():
        out_str=re.sub(a,b,out_str)
    examples[in_str]='\t'+out_str+'\n'
examples=pd.DataFrame(list(examples.items()),columns=['IN','OUT'])
examples_train=examples[:max_examples]
examples_test=examples[max_examples:]
del(examples)
#обучение
def get_batch(examples):
    while True:
        encoder_input_data = np.zeros((max_examples, max_in_len, len(input_token_index)), dtype='float32')
        decoder_input_data = np.zeros((max_examples, max_out_len, len(target_token_index)), dtype='float32')
        decoder_target_data = np.zeros((max_examples,max_out_len, len(target_token_index)), dtype='float32')
        for i, (input_text, target_text) in enumerate(zip(examples['IN'], examples['OUT'])):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_target_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.        
        yield [encoder_input_data,decoder_input_data],decoder_target_data
h=model.fit_generator(get_batch(examples_train),epochs=10000,steps_per_epoch=1,validation_data=get_batch(examples_test),validation_steps=1)
