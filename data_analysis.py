import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('dbmdz/german-gpt2')
    tokens = tokenizer.encode('Transformerpipeline')
    print(tokenizer.sep_token)