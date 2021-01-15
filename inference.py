import re
import json
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")
    model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")
    chef = pipeline(task='text-generation',
                    model=model,
                    tokenizer='anonymous-german-nlp/german-gpt2',
                    config={'max_length': 800})

    result = chef('Ein')[0]['generated_text']
    print(result)