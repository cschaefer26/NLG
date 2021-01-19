from pathlib import Path

import torch
from typing import List

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tnrange, tqdm
from torch.nn.utils.rnn import pad_sequence


class SummarizerDataset(Dataset):

    def __init__(self,
                 abstracts: List[str],
                 texts: List[str],
                 tokenizer: AutoTokenizer,
                 max_tokens=128,
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
        return {'tokens': tokens, 'abstract_len': len(abstract)}


def collate_dataset(batch: List[dict]) -> dict:
    tokens = [b['tokens'] for b in batch]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    abstract_len = [b['abstract_len'] for b in batch]
    return {'tokens': tokens, 'abstract_len': abstract_len}


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def train(model: AutoModelWithLMHead,
          train_dataset: Dataset,
          val_dataset: Dataset,
          batch_size=32) -> None:

    summary_writer = SummaryWriter('logs/gpt2-training')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler,
                              batch_size=batch_size, num_workers=0,
                              collate_fn=collate_dataset)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            num_workers=0, collate_fn=collate_dataset)
    loss_func = CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_step = 0
    for epoch in range(10):
        epoch_iterator = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            total_step += 1
            inputs, labels = batch['tokens'].to(device), batch['tokens'].to(device)
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)[0]
            loss = 0
            norm = 0
            for b, idx in enumerate(batch['abstract_len']):
                shift_logits = logits[b:b+1, idx:-1, :]
                shift_labels = labels[b:b+1, idx+1:]
                b_loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss += b_loss
                norm += shift_labels.size(1)
            loss = loss / norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            summary_writer.add_scalar('Loss/train', loss, global_step=total_step)

            # EVALUATION
            if total_step % 1000 == 0:
                model.eval()
                val_loss = 0
                val_norm = 0
                for val_batch in val_loader:
                    inputs, labels = batch['tokens'].to(device), batch['tokens'].to(device)
                    with torch.no_grad():
                        logits = model(inputs)[0]
                    for b, idx in enumerate(val_batch['abstract_len']):
                        shift_logits = logits[b:b + 1, idx:-1, :]
                        shift_labels = labels[b:b + 1, idx + 1:]
                        b_loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        val_loss += b_loss
                        val_norm += shift_labels.size(1)
                val_loss = val_loss / val_norm
                summary_writer.add_scalar('Loss/val', val_loss, global_step=total_step)
                model.train()

            # GENERATION
            texts = [
                'Deutsche Bank sehr schwach nach Aussagen zum Konzernumbau',
                'Mann nach Sturz in Brunnen schwer verletzt',
                'Unwetterwarnung: Sturm zieht über Bayern',
                'Bayern verliert klar im Pokalfinale gegen Liverpool'
            ]
            if total_step % 1000 == 0:
                model.eval()
                for text in texts:
                    inp = tokenizer.encode(text) + tokenizer.encode('|')
                    gen = generate(model, context=inp, length=100, device=device)
                    gen = tokenizer.decode(gen[0])
                    summary_writer.add_text('Text/Prediction', '    ' + gen,
                                            global_step=total_step)
                    print(f'step {step}, gen: {gen}')
                model.train()

            if total_step % 50000 == 0:
                torch.save(model.state_dict(), f'models/gpt2_step_{total_step}.pt')

    return None


if __name__ == '__main__':
    Path('logs').mkdir(exist_ok=True, parents=True)
    Path('models').mkdir(exist_ok=True, parents=True)
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/german-gpt2')
    df_train = pd.read_csv('data/train.csv')
    train_dataset = SummarizerDataset(abstracts=df_train['seoTitle'].tolist(),
                                      texts=df_train['body'].tolist(),
                                      tokenizer=tokenizer)
    df_test = pd.read_csv('data/test.csv')
    test_dataset = SummarizerDataset(abstracts=df_test['seoTitle'].tolist()[:10],
                                      texts=df_test['body'].tolist()[:10],
                                      tokenizer=tokenizer)

    model = AutoModelWithLMHead.from_pretrained('dbmdz/german-gpt2')
    #state_dict = torch.load('models/gpt2-checkpoint', map_location=torch.device('cpu'))
    #model.load_state_dict(state_dict)
    train(model, train_dataset, test_dataset)
    #gen = generate(model, tokenizer.encode('Das Wetter ist heute sonnig,'),
    #               length=10, device=torch.device('cpu'), top_k=5)
    #print(tokenizer.decode(gen[0]))