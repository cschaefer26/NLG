from transformers import pipeline, AutoModelWithLMHead

if __name__ == '__main__':
    pipe = pipeline(task='text-generation',
                    model='models/gpt2-model',
                    tokenizer='dbmdz/german-gpt2')

    result = pipe('Giengen an der Brenz - Im Mordfall Maria Bögerl findet ab heute ein zweiter', max_length=100)[0]['generated_text']
    print(result)