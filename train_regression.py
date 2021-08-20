import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from sklearn.metrics import r2_score
import data_loader
import config
from lyrics_preprocess import Vocab, split_sentence, PadSequence, Tokenizer
from model import MultiNet


floatTensor = torch.FloatTensor
longTensor = torch.LongTensor


class Runner(object):

    def __init__(self, config, vocab:Vocab):
        self.model = MultiNet(config, vocab, modal=config.MODE)
        self.optimizer = torch.optim.SGD(self.model_parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=config.FACTOR, patience=config.PATIENCE, verbose=True)
        self.learning_rate = config.LEARNING_RATE
        self.stopping_rate = config.STOPPING_RATE
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()

    def run(self, dataloader, mode="train"):
        self.model.train() if mode is "train" else self.model.eval()
        epoch_loss = 0

        all_prediction = []
        all_label = []

        for batch, (audio, lyrics, labels) in enumerate(dataloader):

            audio = audio.type(floatTensor)
            lyrics = lyrics.type(longTensor)
            labels = labels.type(floatTensor)

            prediction = self.model(audio, lyrics)
            prediction = torch.squeeze(prediction, 1)
            loss = self.criterion(prediction, labels)

            all_prediction.extend(prediction.cpu().detach().numpy())
            all_label.extend(labels.cpu().detach().numpy())

            if mode is "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()

        epoch_loss = epoch_loss/len(dataloader.dataset)

        return all_prediction, all_label, epoch_loss

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        stop = self.learning_rate < self.stopping_rate
        return stop


def main():

    with open(config.DATA_PATH + "/vocab.pkl", "rb") as fp:
        vocab = pickle.load(fp)
    pad_sequences = PadSequence(length=config.MAX_LEN, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_sentence, pad_fn=pad_sequences)

    train_loader, valid_loader, test_loader = data_loader.get_dataloader(config)
    runner = Runner(config, vocab=tokenizer.vocab)

    for epoch in range(config.NUM_EPOCHS):
        train_pred, train_label, train_loss = runner.run(train_loader, "train")
        valid_pred, valid_label, valid_loss = runner.run(valid_loader, "eval")

        train_score = r2_score(train_label, train_pred)
        valid_score = r2_score(valid_label, valid_pred)

        if runner.early_stop(valid_loss, epoch+1):
            break

    test_pred, test_label, test_loss = runner.run(test_loader, "eval")
    test_score = r2_score(test_label, test_pred)



