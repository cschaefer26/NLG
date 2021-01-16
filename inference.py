from transformers import pipeline, AutoModelWithLMHead

if __name__ == '__main__':
    model_2 = AutoModelWithLMHead.from_pretrained('dbmdz/german-gpt2')

    pipe_2 = pipeline(task='text-generation',
                    model=model_2,
                    tokenizer='dbmdz/german-gpt2',
                    config={'max_length': 800})


    result_2 = pipe_1('Heute ist schönes Wetter')[0]['generated_text']
    print(result_2)