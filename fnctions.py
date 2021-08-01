import tensorflow as tf 
'''
Functions use Tensorflow API for future building Module for exporting model with pure Tenserflow pipeline
'''
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