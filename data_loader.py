from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, List, Callable
import pickle
import torch
import numpy as np
import pandas as pd
import config
from lyrics_preprocess import Vocab, split_sentence, PadSequence, Tokenizer


# Tokenizer
with open(config.DATA_PATH + "/vocab.pkl", "rb") as fp:
    vocab = pickle.load(fp)
pad_sequence = PadSequence(length=config.MAX_LEN, pad_val=vocab.to_indices(vocab.padding_token))
tokenizer = Tokenizer(vocab=vocab, split_fn=split_sentence, pad_fn=pad_sequence)


class Dataset(Dataset):

    def __init__(self, audio, lyrics, labels):
        self.audio = audio
        self.lyrics = lyrics
        self.labels = labels

    def __getitem__(self, index):
        return self.audio[index], self.lyrics[index], self.labels[index]

    def __len__(self):
        return self.audio.shape[0]


def load_dataset(config, set_name, transform_fn: Callable[[str], List[int]]):
    audio_list = []
    lyrics_list = []
    label_list = []
    df = pd.read_csv("data/" + set_name + ".csv").loc[:, ["MSD_track_id", "lyric", config.LABEL_NAME]]
    for index in range(len(df)):
        audio_name = df.iloc[index]["MSD_track_id"].split(".")[0]+".npy"
        try:
            audio = np.load(config.FEATURE_PATH + audio_name)
        except:
            continue
        lyrics = transform_fn(df.iloc[index]["lyric"])
        label = df.iloc[index][config.LABEL_NAME]
        audio_list.append(lyrics)
        lyrics_list.append(lyrics)
        label_list.append(label)

    if config.LABEL_NAME == "arousal" or config.LABEL_NAME == "valence":
        label_list = torch.FloatTensor(label_list)
    else:
        label_list = torch.LongTensor(label_list)

    return audio_list, lyrics_list, label_list


def get_dataloader(config):
    audio_train, lyrics_train, label_train = load_dataset(config, "train", transform_fn=tokenizer.split_and_transform)
    audio_valid, lyrics_valid, label_valid = load_dataset(config, "valid", transform_fn=tokenizer.split_and_transform)
    audio_test, lyrics_test, label_test = load_dataset(config, "test", transform_fn=tokenizer.split_and_transform)

    mean = np.mean(audio_train)
    std = np.std(audio_train)
    audio_train = (audio_train - mean)/std
    audio_valid = (audio_valid - mean)/std
    audio_test = (audio_test - mean)/std

    lyrics_train = np.stack(lyrics_train)
    lyrics_valid = np.stack(lyrics_valid)
    lyrics_test = np.stack(lyrics_test)

    label_train = np.stack(label_train)
    label_valid = np.stack(label_valid)
    label_test = np.stack(label_test)

    audio_train = np.stack(audio_train)
    audio_valid = np.stack(audio_valid)
    audio_test = np.stack(audio_test)

    train_set = Dataset(audio_train, lyrics_train, label_train)
    valid_set = Dataset(audio_valid, lyrics_valid, label_valid)
    test_set = Dataset(audio_test, lyrics_test, label_test)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader


