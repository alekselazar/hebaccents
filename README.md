# **Hebrew Accents Recognition ML Model**
## Idea
Jewish people read every week a special weekly Torah portion. Except Hebrew letters and punctuation symbols there are hebrew symbols, called in unicode as HEBREW ACCENT. Those symbols represent vocal accents when reading Torah Scroll. In different Jewish diaspors there are different vocal traditions. This repository represents Deep Learning Convolution-Recurrent neural sequence to sequence network, which will get audio wav data as input and will output recognized hebrew accents symbols.

This model, with appropriate user interface, will help people, who is responsible to read Torah on public, to prepare themselves better.

The following model is inspired by two TensorFlow official tutorials, one about encoder decoder translation model(TensorFlow, n.d.) and convolutional audio recognition(TensorFlow, n.d.).

Then will be built the TensorFlow `Module` with audio input and text output processing, using TensorFlow built-in functionality in order to save the `Module`, so it would be able to be deployed on TensorFlow C++ or TFLite.
## Data
For now we will train the Model to recognize hebrew accents as they are in Ashkenazi(European Jewish) tradition. For this we will retrieve data from Yuval Barak's(n.d.) YouTube channel for audio files. First, we manually downloaded audio files. We aguamented our data with speciall functions in notebook.
## ConvRNN 
The main idea is to use encoder decoder seq2seq model, with adding convolutional layer to proccess the input audio set of spectograms as images, then passing the processed sequence in to `LSTM` RNN layer to encode the sequence. The attention layer is not neccessary here, so we go straight to the decoder, which is `LSTM` RNN decoder with num-of-hebrew-acents shape `Dense` output.
## TODO
Now it is neccessary to make manul data cleaninng, contains slicing large audios in short audios so the new dataset will contain psukim
## References:
TensorFlow(n.d.).*Neural machine translation with attention*. https://www.tensorflow.org/text/tutorials/nmt_with_attention

TensorFlow(n.d.).*Simple audio recognition: Recognizing keywords*. https://www.tensorflow.org/tutorials/audio/simple_audio 

11yvl(n.d.).*Yuval Barak[YouTube channel].*YouTube. Retrieved May 3, 2020, from https://www.youtube.com/user/11yvl

ובלכתך בדרך. (n.d.). http://mobile.tora.ws/. 