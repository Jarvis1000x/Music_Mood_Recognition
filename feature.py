import os
import librosa
from pathlib import Path
import itertools
import pickle
from collections import Counter
import allennlp as nlp
import pandas as pd
import numpy as np
from lyrics_preprocess import Vocab, split_sentence
import config


def load_lyrics(set_name):
    df = pd.read_csv("data/" + set_name + ".csv").loc[:, ["lyric"]]
    lyrics_list = []
    label_list = []

    for index in range(len(df)):
        lyrics = df.iloc[index]["lyric"]
        lyrics_list.append(lyrics)

    return lyrics_list


def preprocessing(sentence):
    new_sentence = []
    for i in sentence.split():
        i.isalpha()
        new_sentence.append(i)
    return new_sentence


def melspectrogram(file_name, config):
    y, sr = librosa.load(file_name, config.SAMPLE_RATE)
    mel_s = librosa.feature.melspectrogram(y, config.SAMPLE_RATE, n_fft=config.FFT_SIZE, hop_length=config.HOP_SIZE, win_length=config.WIN_SIZE)
    return mel_s


def resize_array(array, length):
    resized_array = np.zeros((length, array.shape[1]))
    if array.shape[0] >= length:
        resized_array = array[:length]
    else:
        resized_array[:array.shape[0]] = array

    return resized_array


def save_audio(dataset_path, feature_path):
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    audios_path = os.path.join(dataset_path, "MSD")
    audios = [path for path in os.listdir(audios_path)]

    for audio in audios:
        audio_path = os.path.join(audios_path, audio)
        try:
            feature = melspectrogram(audios_path, config)
            if len(feature) < 500:
                print("Feature length is less than 500")

        except:
            print("Cannot load audio {}".format(audio))
            continue
        feature = resize_array(feature, config.FEATURE_LENGTH)
        fn = audio.split(".")[0]
        print(Path(feature_path)/(fn + ".npy"))
        np.save(Path(feature_path)/(fn + ".npy"), feature)


def build_vocab(config, types="fasttext", source="wiki.sample", min_freq=10):
    lyrics_train = load_lyrics("train")
    lyrics_valid = load_lyrics("valid")
    lyrics_test = load_lyrics("test")

    # Extract tokens
    total_vocab = lyrics_train + lyrics_valid + lyrics_test
    list_of_tokens = []
    for i in total_vocab:
        list_of_tokens.append(preprocessing(i))

    token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
    tmp_vocab = nlp.Vocab(
        counter=token_counter, min_freq=10, bos_token=None, eos_token=None
    )

    ptr_embedding = nlp.embedding.create(types, source=source)
    tmp_vocab.set_embedding(ptr_embedding)
    array = tmp_vocab.embedding.idx_to_vec.asnumpy()

    vocab = Vocab(
        padding_token="<pad>",
        unknown_token="<unk>",
        bos_token=None,
        eos_token=None
    )

    vocab.embedding = array

    with open(config.DATA_PATH + "/vocab.pkl", "wb") as io:
        pickle.dump(vocab, io)








