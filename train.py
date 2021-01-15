# adapted from https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb#scrollTo=laDp891gO25V

import re
import json

from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from sklearn.model_selection import train_test_split



def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for texts in data_json:
        summary = str(texts['Instructions']).strip()
        summary = re.sub(r'\s', ' ', summary)
        data += summary + '  '
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
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


class SummaryTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = CrossEntropyLoss()

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs[0]
        logits = logits.transpose(1, 2)
        loss = self.loss_func(logits[:, :, :-1], labels[:, 1:])
        print(f'loss: {loss}')
        return loss


if __name__ == '__main__':

    train_path = 'data/train_dataset.txt'
    test_path = 'data/test_dataset.txt'
    with open('data/recipes.json') as f:
        data = json.load(f)

    train, test = train_test_split(data, test_size=0.15)
    build_text_files(train, 'data/train_dataset.txt')
    build_text_files(test, 'data/test_dataset.txt')

    print('Train dataset length: '+str(len(train)))
    print('Test dataset length: '+ str(len(test)))

    tokenizer = AutoTokenizer.from_pretrained('anonymous-german-nlp/german-gpt2')


    train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

    model = AutoModelWithLMHead.from_pretrained('anonymous-german-nlp/german-gpt2')

    training_args = TrainingArguments(
        output_dir='models/gpt2-model', #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=1, # number of training epochs
        per_device_train_batch_size=32, # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        eval_steps=400, # Number of update steps between two evaluations.
        save_steps=800, # after # steps model is saved
        warmup_steps=500,# number of warmup steps for learning rate scheduler
        )

    trainer = SummaryTrainer(
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