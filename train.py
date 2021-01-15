# adapted from https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb#scrollTo=laDp891gO25V

import re
import json
from torch.nn import CrossEntropyLoss
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from sklearn.model_selection import train_test_split


def build_text_files(text_list, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for text in text_list:
        summary = str(text.split('|')[1]).strip()
        data += f'{summary}\n'
    f.write(data)


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    return train_dataset, test_dataset, data_collator


if __name__ == '__main__':
    with open('/Users/cschaefe/datasets/ASVoice4_breathing_francesco/metadata_clean.csv') as f:
        data = f.readlines()
    train_path = 'data/train_dataset.txt'
    test_path = 'data/test_dataset.txt'
    train, test = train_test_split(data, test_size=0.1)
    build_text_files(train, train_path)
    build_text_files(test, test_path)
    print(f'Train dataset length: {len(train)}')
    print(f'Test dataset length: {len(test)}')

    tokenizer = AutoTokenizer.from_pretrained('anonymous-german-nlp/german-gpt2')
    train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

    model = AutoModelWithLMHead.from_pretrained('anonymous-german-nlp/german-gpt2')

    training_args = TrainingArguments(
        output_dir='models/gpt2-model', #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=1, # number of training epochs
        per_device_train_batch_size=32, # batch size for training
        per_device_eval_batch_size=32,  # batch size for evaluation
        eval_steps=400, # Number of update steps between two evaluations.
        save_steps=800, # after # steps model is saved
        warmup_steps=500)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset)

    trainer.train()
    trainer.save_model()

    pipe = pipeline(task='text-generation',
                    model='models/gpt2-model',
                    tokenizer='anonymous-german-nlp/german-gpt2',
                    config={'max_length': 800})

    result = pipe('Zuerst Hähnchen')[0]['generated_text']

    print(result)