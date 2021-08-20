import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import config
from lyrics_preprocess import Vocab, split_sentence, PadSequence, Tokenizer
from model import MultiNet
import data_loader


floatTensor = torch.FloatTensor
longTensor = torch.LongTensor


class Runner(object):

    def __init__(self, config, vocab: Vocab):
        self.model = MultiNet(config, vocab, modal=config.MODE)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=config.FACTOR, patience=config.PATIENCE, verbose=True)
        self.learning_rate = config.LEARNING_RATE
        self.stopping_rate = config.STOPPING_RATE
        self.device = config.DEVICE
        self.criterion = torch.nn.CrossEntropyLoss()

    def accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.long().cpu()
        correct = (source == target).sum().item()
        return correct/float(source.size(0))

    def run(self, dataloader, mode="train"):
        self.model.train() if mode is "train" else self.model.eval()
        epoch_loss = 0
        epoch_acc = 0

        for batch, (audio, lyrics, labels) in enumerate(dataloader):

            audio = audio.type(floatTensor)
            lyrics = lyrics.type(longTensor)
            labels = labels.type(longTensor)

            prediction = self.model(audio, lyrics)
            loss = self.criterion(prediction, labels)
            acc = self.accuracy(prediction, labels)

            if mode is "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        return epoch_loss, epoch_acc

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        stop = self.learning_rate < self.stopping_rate
        return stop


def main():

    with open(config.DATA_PATH + "vocab.pkl", "rb") as fp:
        vocab = pickle.load(fp)
    pad_sequence = PadSequence(length=config.MAX_LEN, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_sentence, pad_fn=pad_sequence)

    train_loader, valid_loader, test_loader = data_loader.get_dataloader(config)
    runner = Runner(config, vocab=tokenizer.vocab)

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = runner.run(train_loader, "train")
        valid_loss, valid_acc = runner.run(valid_loader, "eval")

        if runner.early_stop(valid_loss, epoch + 1):
            break

    test_loss, test_acc = runner.run(test_loader, "eval")



