import torch
import torch.nn as nn
from torch.autograd import Variable
from lyrics_preprocess import Vocab


class MultiNet(nn.Module):

    def __init__(self, config, vocab: Vocab, modal="multi"):
        super().__init__()
        self.modal = modal
        if modal == "audio":
            self.audionet = AudioNet(config)
        elif modal == "lyrics":
            self.lyricsnet = LyricsNet(config, vocab)
        elif modal == "multi":
            self.audionet = AudioNet(config)
            self.lyricsnet = LyricsNet(config, vocab)
        else:
            raise ValueError("Mode should be specified")

        if config.LABEL_NAME == 'arousal' or config.LABEL_NAME == 'valence':
            self.num_output = 1
        elif config.LABEL_NAME == 'Label_kMean_4' or config.LABEL_NAME == 'Label_abs_4':
            self.num_output = 4
        elif config.LABEL_NAME == 'Label_kMean_16' or config.LABEL_NAME == 'Label_kMean_16':
            self.num_output = 16
        else:
            self.num_output = 64

        self._classifier = nn.Sequential(nn.Linear(in_features=1456, out_features=100),
                                         nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                         nn.Dropout(),
                                         nn.Linear(in_features=100, out_features=self.num_output))

    def forward(self, audio, lyrics):

        if self.modal == "audio":
            score, _ = self.audionet(audio)
        elif self.modal == "lyrics":
            score, _ = self.lyricsnet(lyrics)
        elif self.modal == "multi":
            _, audioflat = self.audionet(audio)
            _, lyricsflat = self.lyricsnet(lyrics)
            concat = torch.cat((audioflat, lyricsflat), dim=1)
            score = self._classifier(concat)
        return score


class AudioNet(nn.Module):

    def __init__(self, config):
        super(AudioNet, self).__init__()

        if config.LABEL_NAME == 'arousal' or config.LABEL_NAME == 'valence':
            self.num_output = 1
        elif config.LABEL_NAME == 'Label_kMean_4' or config.LABEL_NAME == 'Label_abs_4':
            self.num_output = 4
        elif config.LABEL_NAME == 'Label_kMean_16' or config.LABEL_NAME == 'Label_kMean_16':
            self.num_output = 16
        else:
            self.num_output = 64

        self.conv0 = nn.Sequential(
            nn.Conv1d(config.NUM_MELS, 32, kernel_size=8, stride=1, padding=0),
            nn.MaxPool1d(4, stride=4),
            nn.BatchNorm1d(32)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(config.NUM_MELS, 32, kernel_size=8, stride=1, padding=0),
            nn.MaxPool1d(4, stride=4),
            nn.BatchNorm1d(16),
        )

        self._clasifier = nn.Sequential(nn.Linear(in_features=976, out_features=64),
                                        nn.Tanh(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=64, out_features=self.num_output))
        self.apply(self._init_weights)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.conv0(x)
            x = self.conv1(x)
            flatten = x.view(x.size(0), -1)
            score = self._classifier(flatten)
            return score, flatten

        def _init_weights(self, layer) -> None:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight)

            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


class LyricsNet(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super(LyricsNet, self).__init__()

        if config.LABEL_NAME == 'arousal' or config.LABEL_NAME == 'valence':
            self.num_output = 1
        elif config.LABEL_NAME == 'Label_kMean_4' or config.LABEL_NAME == 'Label_abs_4':
            self.num_output = 4
        elif config.LABEL_NAME == 'Label_kMean_16' or config.LABEL_NAME == 'Label_kMean_16':
            self.num_output = 16
        else:
            self.num_output = 64

        self._lstm_hid_dim = config.LSTM_LATENT_DIM
        self._batch_size = config.BATCH_SIZE

        self._embedding = nn.Embedding(len(vocab), config.EMBD_DIM, vocab.to_indices(vocab.padding_token))
        self._cnn_permute = CNN_Permute()
        self._conv0 = nn.Sequential(nn.Conv1d(in_channels=config.EMBD_DIM, out_channels=16, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2))
        self._flatten = Flatten()
        self._rnn_permute = RNN_Permute()
        self._lstm = nn.LSTM(16, config.LSTM_LATENT_DIM, batch_first=False)
        self.hidden_state = self.init_hidden()

        self._classifier = nn.Sequential(nn.Linear(in_features=480, out_features=64),
                                         nn.Tanh(),
                                         nn.Dropout(0.5),
                                         nn.Linear(in_features=64, out_features=self.num_output))
        self.apply(self._initialize)

    def forward(self, x):
        x = self._embedding(x)
        x = self._cnn_permute(x)
        x = self._conv0(x)
        x = x.permute(2, 0, 1)
        lstm_out, _ = self._lstm(x, self.hidden_state)
        lstm_out = lstm_out.permute(1, 2, 0)
        flatten = self._flatten(lstm_out)
        score = self._classifier(flatten)
        return score, flatten

    def init_hidden(self):
        h0 = Variable(torch.zeros(1, self._batch_size, self._lstm_hid_dim))
        c0 = Variable(torch.zeros(1, self._batch_size, self._lstm_hid_dim))
        h0, c0 = h0.cuda(), c0.cuda()
        return (h0, c0)

    def _initialize(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(layer.weight)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.contiguous()
        return x.view(x.size(0), -1)


class CNN_Permute(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.permute(0, 2, 1)


class RNN_Permute(nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.permute(2, 0, 1)
        x = x.view(x.size(0), x.size(1), -1)
        return x




