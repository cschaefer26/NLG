from transformers import pipeline

if __name__ == '__main__':

    pipe = pipeline(task='text-generation',
                    model='models/gpt2-model',
                    tokenizer='anonymous-german-nlp/german-gpt2',
                    config={'max_length': 800})

    result = pipe('Zuerst Knoblauch schneiden, dann')[0]['generated_text']
    print(result)