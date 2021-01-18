import torch
from typing import List

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SummarizerDataset(Dataset):

    def __init__(self,
                 abstracts: List[str],
                 texts: List[str],
                 tokenizer: AutoTokenizer,
                 max_tokens=512,
                 sep_word='|') -> None:
        self.abstracts = abstracts
        self.texts = texts
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.encode(sep_word)
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        abstract = self.abstracts[idx]
        abstract = self.tokenizer.encode(abstract)
        text = self.texts[idx]
        text = self.tokenizer.encode(text)
        tokens = abstract + self.sep_token + text
        tokens = tokens[:self.max_tokens]
        tokens = torch.tensor(tokens)
        return {'tokens': tokens, 'abstract_len': len(abstract) + 1}


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/german-gpt2')
    dataset = SummarizerDataset(['abstract'], ['text'], tokenizer)