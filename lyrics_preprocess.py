from nltk.tokenize import TweetTokenizer
from typing import List, Callable, Union, Dict

# instantiate tokeniser
tok = TweetTokenizer()
split_sentence = tok.tokenize()


# Vocab Class
class Vocab:

    def __init__(
            self,
            list_of_tokens: List[str] = None,
            padding_token: str = "<pad>",
            unknown_token: str = "<unk>",
            bos_token: str = "<bos>",
            eos_token: str = "<eos>",
            reserved_tokens: List[str] = None,
            token_to_idx: Dict[str, int] = None
    ):
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._reserved_tokens = reserved_tokens
        self._special_tokens = []

        for tkn in [
            self._unknown_token,
            self._padding_token,
            self._bos_token,
            self._eos_token,
        ]:
            if tkn:
                self._special_tokens.append(tkn)

        if self._reserved_tokens:
            self._special_tokens.extend(self._reserved_tokens)

        if list_of_tokens:
            self._special_tokens.extend(
                list(
                    filter(lambda elm: elm not in self._special_tokens, list_of_tokens)
                )
            )

        self._token_to_idx, self._idx_to_token = self._build(self._special_tokens)

        if token_to_idx:
            self._sort_index_acc_to_user(token_to_idx)

        self._embedding = None

    def to_indices(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, list):
            return [
                self._token_to_idx[tkn]
                if tkn in self._token_to_idx
                else self._token_to_idx[self._unknown_token]
                for tkn in tokens
            ]
        else:
            return (
                self._token_to_idx[tokens]
                if tokens in self._token_to_idx
                else self._token_to_idx[self._unknown_token]
            )

    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(indices, list):
            return [self._idx_to_token[idx] for idx in indices]
        else:
            return self._idx_to_token[indices]

    def _build(self, list_of_tokens):
        token_to_idx = {tkn: idx for idx, tkn in enumerate(list_of_tokens)}
        idx_to_token = list_of_tokens
        return token_to_idx, idx_to_token

    def _sort_index_acc_to_user(self, token_to_idx):
        for token, new_idx in token_to_idx.items():
            old_idx = self._token_to_idx[token]
            ousted_token = self._idx_to_token[new_idx]

            self._token_to_idx[token] = new_idx
            self._token_to_idx[ousted_token] = old_idx
            self._idx_to_token[old_idx] = ousted_token
            self._idx_to_token[new_idx] = token

    def __len(self):
        return len(self._token_to_idx)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, array):
        self._embedding = array


# Tokenizer Class
class Tokenizer:

    def __init__(
            self,
            vocab: Vocab,
            split_fn: Callable[[str], List[str]],
            pad_fn: Callable[[List[int]], List[int]] = None,) -> None:
        self._vocab = vocab
        self._split = split_fn
        self._pad = pad_fn

    def split(self, string: str) -> List[str]:
        list_of_tokens = self._split(string)
        return list_of_tokens

    def transform(self, list_of_tokens: List[str]) -> List[int]:
        list_of_indices = self._vocab.to_indices(list_of_tokens)
        list_of_indices = self._pad(list_of_indices) if self._pad else list_of_indices
        return list_of_indices

    def split_and_transform(self, string: str) -> List[int]:
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab


class PadSequence:

    def __init__(self, length: int, pad_val: int = 0, clip: bool = True) -> None:
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample):
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[: self._length]
            else:
                return sample
        else:
            return sample + [self._pad_val for _ in range(self._length - sample_length)]


