<div class="cell markdown" id="FD6BOhCOv0OK">

# **Hebrew Accents Recognition ML Model**

## Idea 
Jewish people read every week a special weekly Torah portion.
Except Hebrew letters and punctuation symbols there are hebrew symbols,
called in unicode as HEBREW ACCENT. Those symbols represent vocal
accents when reading Torah Scroll. In different Jewish diaspors there
are different vocal traditions. This repository represents Deep Learning
Convolution-Recurrent neural sequence to sequence network, which will
get audio wav data as input and will output recognized hebrew accents
symbols.

This model, with appropriate user interface, will help people, who is
responsible to read Torah on public, to prepare themselves better.

The following model is inspired by two TensorFlow official tutorials,
one about encoder decoder translation model(TensorFlow, n.d.) and
convolutional audio recognition(TensorFlow, n.d.).

The main idea is to use encoder decoder seq2seq model, with adding
`TimeDistributed` convolutional layer to proccess the input audio set of
spectograms as images, then passing the processed sequence in to `GRU`
RNN layer to encode the sequence. The attention layer is not neccessary
here, so we go straight to the decoder, which is `GRU` RNN decoder with
num-of-hebrew-acents shape `Dense` output.

Then will be built the TensorFlow `Module` with audio input and text
output processing, using TensorFlow built-in functionality in order to
save the `Module`, so it would be able to be deployed on TensorFlow C++
or TFLite.

</div>

<div class="cell markdown" id="eWus3kiS5fZl">

## Data 
For now we will train the Model to recognize hebrew accents as
they are in Ashkenazi(European Jewish) tradition. For this we will
retrieve data from Yuval Barak's(n.d.) YouTube channel for audio files.
For output we retrieved Torah text with different metadata in
jholybooks\_data repository from tora.ws(n.d.) website.

First, we wanted to automaticly download and process audio files using
`pytube` library, but the code crushed in the middle of execution with
`404 ERROR`. YouTube changed their code, so untill new update, pytube is
not able to work with YouTube. So we just downloaded audios MP3 and now
will process them to WAV format.

</div>

<div class="cell code" data-execution_count="1" id="_xn1OeDKsiM1">

``` python
#Import dependencies
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Dropout, Reshape, Flatten, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy

import numpy as np
import librosa
import soundfile as sf
import unicodedata
import random
import string
import csv
import json
import os
```

</div>

<div class="cell markdown" id="W2eTezXNPE10">

Since our data set is not so big, we will use Data Augumentation, for
this we will declare some functions:

</div>

<div class="cell code" data-execution_count="2" id="u2gpBYyu3OSr">

``` python
#Data augumentation functions
def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def change_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def change_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)
```

</div>

<div class="cell markdown" id="bY0NOtp4PtKL">

Now we will define a function which will process our input and output
data and return pairs of input-output

</div>

<div class="cell code" data-execution_count="3" id="XosT51FhQ_BJ">

``` python
#Helper function to deal with hebrew numerology, transforms hebrew letters to numbers

def gimatria(letters):
    alphabet = 'אבגדהוזחטיכלמנסעפצקרשת'
    result = 0
    try:
        for l in letters:
            n = alphabet.index(l) + 1
            d = int(n/10)
            result += (int(n%10) + d) * (10 ** d)
    except:
        print('Wrong parameter, hebrew letters only expected! Got: ', letters)
    return result

#Actual function that processes data
def process_data(dataset_dirname, accent_names):
    books = [
             'בראשית',
             'שמות',
             'ויקרא',
             'במדבר',
             'דברים',
    ]
    aliyot = {
        'ראשון' : 1,
        'שני': 2,
        'שלישי': 3,
        'רביעי': 4,
        'חמישי': 5,
        'שישי': 6,
        'שביעי': 7,
    }
    EXPECTED_SAMPLE_RATE = 16000
    inputs = []
    outputs = []

    for book_name in books:
        chapters_in_book = []
        with open(os.path.join('outputs', book_name + '.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)
        pasuk_counter = 0
        for chap in data['chapters']:
            psukim = data['psukim'][pasuk_counter:pasuk_counter + data['chapters'][chap]]
            chapters_in_book.append(psukim)
            pasuk_counter += data['chapters'][chap]
                  
        for w_chap in data['weekly_chaps']:
            if w_chap in data['double_chaps']: continue
            for alia in data['weekly_chaps'][w_chap]['aliyot']:
                alia_p = []
                f = data['weekly_chaps'][w_chap]['aliyot'][alia][0].split('-')
                to = data['weekly_chaps'][w_chap]['aliyot'][alia][1].split('-')
                f_chap = gimatria(f[0])
                f_pasuk = gimatria(f[1][1:-1])
                to_chap = gimatria(to[0])
                to_pasuk = gimatria(to[1][1:-1])
                
                if f_chap == to_chap:
                    alia_p = chapters_in_book[f_chap - 1][f_pasuk - 1:to_pasuk]
                else:
                    for c in range(f_chap - 1, to_chap):
                        if c == f_chap - 1:
                            for i in range(f_pasuk - 1, len(chapters_in_book[c])):
                                alia_p.append(chapters_in_book[c][i])
                        elif c == to_chap - 1:
                            for i in range(to_pasuk):
                                alia_p.append(chapters_in_book[c][i])
                        else:
                            for i in chapters_in_book[c]:
                                alia_p.append(i)
                alia_accents = []
                for pasuk in alia_p:
                    for char in pasuk:
                        if unicodedata.name(char) in accent_names:
                            accent_vector = np.zeros(len(accent_names))
                            accent_vector[accent_names.index(unicodedata.name(char))] = 1.
                            alia_accents.append(accent_vector)
                alia_accents = np.asarray(alia_accents)
                for i in range(6):
                    np_file = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                    np_path = os.path.join(dataset_dirname, 'outputs', np_file+'.npy')
                    np.save(np_path, alia_accents)
                    outputs.append(np_path)

                audio_file = os.path.join('inputs', book_name, w_chap[5:], str(aliyot[alia]) + '.mp3')
                y, sr = librosa.load(audio_file, EXPECTED_SAMPLE_RATE)
                if len(y.shape) == 2:
                    y = y.mean(1)
                y = y.astype(np.float32)
                all_audio_data_after_augumentation = [
                                                      y,
                                                      add_noise(y, 0.05),
                                                      change_pitch(y, sr, 4),
                                                      change_pitch(y, sr, -6),
                                                      change_speed(y, 1.5),
                                                      change_speed(y, .75),
                ]
                for aud_data in all_audio_data_after_augumentation:
                    wav_file = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                    wav_path = os.path.join(dataset_dirname, 'inputs', wav_file+'.wav')
                    sf.write(wav_path, y, sr)
                    inputs.append(wav_path)

    if len(inputs) == len(outputs):
        csv_data = [i for i in zip(inputs, outputs)]
        with open(os.path.join(dataset_dirname, 'dataset.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(csv_data)
        return inputs, outputs
    else:
        raise ValueError("Something has gone wrong! Inputs and Outputs must be the same length!")
```

</div>

<div class="cell markdown" id="3ZdZbxw66y_C">

After processing our audio files and building our dataset directory, now
we need to define data preprocessing functions

</div>

<div class="cell code" data-execution_count="4" id="2suY_Bw17OIM">

``` python
def decode_audio(file_path):
    file_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(file_binary)
    return tf.squeeze(audio, axis=-1)

#This function slices audio in sequence of 1-second audios for our seq2seq model
def slice_audio_to_time_series(wavform):
    if tf.cast(tf.shape(wavform)[0] % [16000], dtype=tf.bool):
        zero_padding = tf.zeros([16000] - (tf.shape(wavform)[0] % [16000]))
        wavform = tf.concat([wavform, zero_padding], 0)
    wav_seq = tf.reshape(wavform, (tf.shape(wavform)[0] // 16000, 16000))
    return wav_seq

#Function that converts audio in to sequence of 1-second spectograms
def get_spectrograms_sequence(file_path):
    wavform = decode_audio(file_path)
    wav_seq = slice_audio_to_time_series(wavform)
    seq_of_specs = []
    for frame in wav_seq:
        spectrogram = tf.signal.stft(frame, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.reshape(spectrogram, spectrogram.shape + (1))
        seq_of_specs.append(np.asarray(spectrogram))
    return tf.constant(seq_of_specs)

#Function that returns output for model
def get_output(output_file_path):
    return tf.constant(np.load(output_file_path), dtype=tf.float32)
```

</div>

<div class="cell markdown" id="PhIHWX3iGPh3">

Now we will finally prepare a `Dataset` for our model

</div>

<div class="cell code" data-execution_count="5" id="lj1PCkZrGPEc">

``` python
accent_names = [
                "HEBREW ACCENT ETNAHTA",
                "HEBREW ACCENT SEGOL",
                "HEBREW ACCENT SHALSHELET",
                "HEBREW ACCENT ZAQEF QATAN",
                "HEBREW ACCENT ZAQEF GADOL",
                "HEBREW ACCENT TIPEHA",
                "HEBREW ACCENT REVIA",
                "HEBREW ACCENT ZARQA",
                "HEBREW ACCENT PASHTA",
                "HEBREW ACCENT YETIV",
                "HEBREW ACCENT TEVIR",
                "HEBREW ACCENT GERESH",
                "HEBREW ACCENT GERESH MUQDAM",
                "HEBREW ACCENT GERSHAYIM",
                "HEBREW ACCENT QARNEY PARA",
                "HEBREW ACCENT TELISHA GEDOLA",
                "HEBREW ACCENT PAZER",
                "HEBREW ACCENT ATNAH HAFUKH",
                "HEBREW ACCENT MUNAH",
                "HEBREW ACCENT MAHAPAKH",
                "HEBREW ACCENT MERKHA",
                "HEBREW ACCENT MERKHA KEFULA",
                "HEBREW ACCENT DARGA",
                "HEBREW ACCENT QADMA",
                "HEBREW ACCENT TELISHA QETANA",
                "HEBREW ACCENT YERAH BEN YOMO",
                "HEBREW ACCENT OLE",
                "HEBREW ACCENT ILUY",
                "HEBREW ACCENT DEHI",
                "HEBREW ACCENT ZINOR",
                "HEBREW POINT METEG",
                "HEBREW PUNCTUATION PASEQ",
]
#inp, targ = process_data('askenaz_accents_data', accent_names)

inp = []
targ = []
with open("/content/askenaz_accents_data/dataset.csv", 'r') as f:
    r = csv.reader(f)
    for row in r:
        inp.append(row[0])
        targ.append(row[1])

num_lables = len(accent_names)
X_train = inp[:-tf.shape(inp)[0]//10]
Y_train = targ[:-tf.shape(targ)[0]//10]
X_test = inp[-tf.shape(inp)[0]//10:]
Y_test = targ[-tf.shape(targ)[0]//10:]

input_shape = get_spectrograms_sequence(inp[0])
input_shape = input_shape.shape[1:]

#Functions generator to pass into training

def train_dataset_generator():
    for i in range(len(X_train)):
        #For decoder model input we will use an empty arr in length of expected output
        yield {'input': get_spectrograms_sequence(X_train[i]), 
               'decoder_empty_input': tf.zeros(get_output(Y_train[i]).shape[0])}, get_output(Y_train[i])

def test_dataset_generator():
    for i in range(len(X_test)):
        yield {'input': get_spectrograms_sequence(X_test[i]), 
               'decoder_empty_input': tf.zeros(get_output(Y_test[i]).shape[0]) + 1}, get_output(Y_test[i])

train_dataset = tf.data.Dataset.from_generator(
    train_dataset_generator,
    output_signature=(
            {'input': tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
             'decoder_empty_input': tf.TensorSpec(shape=(None,), dtype=tf.float32)},
            tf.TensorSpec(shape=(None, num_lables), dtype=tf.float32)
    )
)
test_dataset = tf.data.Dataset.from_generator(
    test_dataset_generator,
    output_signature=(
            {'input': tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
             'decoder_empty_input': tf.TensorSpec(shape=(None,), dtype=tf.float32)},
            tf.TensorSpec(shape=(None, num_lables), dtype=tf.float32)
    )
)

BATCH_SIZE = 1
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
```

</div>

<div class="cell markdown" id="53qRE0n-hRV_">

And, finaly we will built our training model

</div>

<div class="cell code" data-execution_count="6" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="uwxDdPcy8Exd" data-outputId="81b8c16f-c4c0-46d6-9544-b14489e7ffce">

``` python
print('Input shape:', input_shape)

enc_input = Input(
    shape=(None,) + input_shape, 
    name='input', 
    dtype='float32'
    )

enc_output = TimeDistributed(
    Conv2D(
        32,
        3,
        activation='relu'
    )
)(enc_input)

enc_output = TimeDistributed(
    Conv2D(
        64,
        3,
        activation='relu'
    )
)(enc_input)

enc_output = TimeDistributed(MaxPool2D())(enc_output)

enc_output = TimeDistributed(
    Dropout(0.25)
)(enc_output)

enc_output = TimeDistributed(
    Flatten()
)(enc_output)

enc_output = Dense(32)(enc_output)

enc_output = Dropout(0.5)(enc_output)

enc_output, enc_output_state = GRU(32, return_state=True)(enc_output)

print(enc_output_state)

dec_input = Input(
    shape=(None,1),
    name='decoder_empty_input',
    dtype='float32'
)

decoder = GRU(32, return_sequences=True)(dec_input, initial_state=enc_output_state)
decoder = Dense(32)(decoder)

model = Model([enc_input, dec_input], decoder)
model.summary()
```

<div class="output stream stdout">

    Input shape: (124, 129, 1)
    KerasTensor(type_spec=TensorSpec(shape=(None, 32), dtype=tf.float32, name=None), name='gru/PartitionedCall:2', description="created by layer 'gru'")
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input (InputLayer)              [(None, None, 124, 1 0                                            
    __________________________________________________________________________________________________
    time_distributed_1 (TimeDistrib (None, None, 122, 12 640         input[0][0]                      
    __________________________________________________________________________________________________
    time_distributed_2 (TimeDistrib (None, None, 61, 63, 0           time_distributed_1[0][0]         
    __________________________________________________________________________________________________
    time_distributed_3 (TimeDistrib (None, None, 61, 63, 0           time_distributed_2[0][0]         
    __________________________________________________________________________________________________
    time_distributed_4 (TimeDistrib (None, None, 245952) 0           time_distributed_3[0][0]         
    __________________________________________________________________________________________________
    dense (Dense)                   (None, None, 32)     7870496     time_distributed_4[0][0]         
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, None, 32)     0           dense[0][0]                      
    __________________________________________________________________________________________________
    decoder_empty_input (InputLayer [(None, None, 1)]    0                                            
    __________________________________________________________________________________________________
    gru (GRU)                       [(None, 32), (None,  6336        dropout_1[0][0]                  
    __________________________________________________________________________________________________
    gru_1 (GRU)                     (None, None, 32)     3360        decoder_empty_input[0][0]        
                                                                     gru[0][1]                        
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, None, 32)     1056        gru_1[0][0]                      
    ==================================================================================================
    Total params: 7,881,888
    Trainable params: 7,881,888
    Non-trainable params: 0
    __________________________________________________________________________________________________

</div>

</div>

<div class="cell code" data-execution_count="7" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="_AkwFz1vv0QL" data-outputId="53af2683-97e2-48ca-d33e-8ffade16f6b7">

``` python
adam = tf.keras.optimizers.Adam()
model.compile(adam, 'binary_crossentropy', metrics=['binary_accuracy'])

hist = model.fit(
    x=train_dataset,
    validation_data=test_dataset
    )
```

<div class="output stream stdout">

    2035/2035 [==============================] - 6875s 3s/step - loss: 0.1854 - binary_accuracy: 0.9687

</div>

</div>

<div class="cell markdown" id="S7muNEoGE3op">

This is still very rare project, we will continue to work on it,
optimize data, etc.

For now we just save the weights for future work

</div>

<div class="cell code" data-execution_count="8" id="tcbDlp2qayFE">

``` python
model.save_weights('askenaz_taamim_model.h5')
```

</div>

<div class="cell markdown" id="ufVnw22P2tQB">

### References: 
TensorFlow(n.d.).*Neural machine translation with
attention*.
<https://www.tensorflow.org/text/tutorials/nmt_with_attention>

TensorFlow(n.d.).*Simple audio recognition: Recognizing keywords*.
<https://www.tensorflow.org/tutorials/audio/simple_audio>

11yvl(n.d.).*Yuval Barak\[YouTube channel\].*YouTube. Retrieved May 3,
2020, from <https://www.youtube.com/user/11yvl>

ובלכתך בדרך. (n.d.). <http://mobile.tora.ws/>.

</div>
