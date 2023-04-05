# Copyright (c) 2023 by the LEL-A team
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

# This is the same as `Translate-Cleaned-Alpaca-Dataset.ipynb` but uses
# an 4-model ensemble:
#
# ```
# en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
#                        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
#                        tokenizer='moses', bpe='fastbpe')
# ```

import torch
import requests
import more_itertools
from tqdm import tqdm
import json
from tqdm import tqdm


MAX_TOKEN_COUNT = 1000


# Download cleaned Alpaca Dataset from: https://github.com/gururise/AlpacaDataCleaned
# Use specific commit (current latest main) for reproducability)
r = requests.get("https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/2ee9f5ca1d4dc2df3777a765bab88ad061e83378/alpaca_data_cleaned.json")

assert r

data = r.json()

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
_ = en2de.eval()  # disable dropout
_ = en2de.cuda()  # use GPU

source_instructions = [example["instruction"].replace(" \n", "<br>").replace("\n", "<br>") for example in data]
source_inputs = [example["input"].replace(" \n", "<br>").replace("\n", "<br>") for example in data]
source_outputs = [example["output"].replace(" \n", "<br>").replace("\n", "<br>") for example in data]


# some source_outputs are to long for translation - we remove and mark them
source_outputs_clipped = []
for i, s in tqdm(enumerate(source_outputs)):
    e = en2de.encode(s)
    if len(e) > MAX_TOKEN_COUNT:
        source_outputs_clipped.append(True)
        source_outputs[i] = ""
        print(f"cliping source_outputs at index {i}")
    else:
        source_outputs_clipped.append(False)


assert len(source_instructions) == len(source_inputs) == len(source_outputs) == len(source_outputs_clipped)

# FIXME: remove this later - it is just for debug
#source_instructions = source_instructions[:20]
#source_inputs = source_inputs[:20]
#source_outputs = source_outputs[:20]

def translate_texts(texts):
    en_de_texts = []
    chunks = list(more_itertools.chunked(texts, 10))
    for chunk in tqdm(chunks):
        en_de_texts.extend(en2de.translate(chunk))
    return en_de_texts

translated_outputs = translate_texts(source_outputs)
translated_instructions = translate_texts(source_instructions)
translated_inputs = translate_texts(source_inputs)

translated_instructions = list(map(lambda t: t.replace(" < br > ", "\n").replace("< br > ", "\n"), translated_instructions))
translated_inputs = list(map(lambda t: t.replace(" < br > ", "\n").replace("< br > ", "\n"), translated_inputs))
translated_outputs = list(map(lambda t: t.replace(" < br > ", "\n").replace("< br > ", "\n"), translated_outputs))

translated_data = []

for source_input, translated_input, translated_instruction, translated_output, source_outputs_clip in zip(
    source_inputs,
    translated_inputs,
    translated_instructions,
    translated_outputs,
    source_outputs_clipped,
):

    current_example = {
        "instruction": translated_instruction,
        "input": translated_input if source_input else "",
        "output": translated_output,
        "output_cliped": source_outputs_clip,
    }
    translated_data.append(current_example)

with open("translated_german_alpaca_02.json", "wt") as f_p:
    json.dump(translated_data, f_p)
