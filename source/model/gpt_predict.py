import time

from openai import OpenAI

def predict(args, prompt):
    my_key = args.api_key
    max_length = 256
    temperature = 0.0
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0
    client = OpenAI(api_key = my_key)
    prompt = "
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct", # text-davinci-003 is deprecated
        prompt=prompt,
        max_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        #   api_key=my_key,
    )
    if args.engine == 'llama2-13b':
        raise NotImplementedError('Engine false when running gpt3.5: {}'.format(args.engine))
    return response.choices[0].text