import pandas as pd

import torch
import tensorflow as tf 
from transformers import AutoTokenizer
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re
from quickspacer import Spacer
import numpy as np


device_kind = ""
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

domain = [
    "Happy",
    "Fear",
    "Embarrassed",
    "Sad",
    "Rage",
    "Hurt",
]

if torch.cuda.is_available():    
    device = torch.device("cuda")
    device_kind="cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    device_kind="cpu"
    print('No GPU available, using the CPU instead.')


model = torch.load('./KcElectra.pt', map_location=torch.device(device_kind))
def predict(sentence):
    print(sentence)
    model.eval()

    tokenized_sentence = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128,
    )
    tokenized_sentence.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sentence["input_ids"],
            attention_mask=tokenized_sentence["attention_mask"],
            token_type_ids=tokenized_sentence["token_type_ids"]
            )

    logits = outputs[0]
    # logits = logits.detach().cpu()

    logits = logits.tolist()[0]
    print(logits)
    return [logits, domain[np.argmax(logits)], logits[np.argmax(logits)] ]
    # return logits


def remove_unnecessary_word(text):
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', text)
    
    spacer = Spacer(level=3)
    # text = text.rstrip().lstrip()
    # text.replace(" " , "")
    text = spacer.space([text])
    return text[0]


def analyze_word(row):
    print("------------")
    try:
        result = predict(remove_unnecessary_word(row))
    except Exception as e:
        print("Get some err" + str(e))
        return
    return result


# Happy '기쁨' = 0                    
# Fear '불안' = 1                   
# Embarrassed '당황' = 2                    
# Sad '슬픔' = 3                    
# Rage '분노' = 4                    
# Hurt '상처' = 5  