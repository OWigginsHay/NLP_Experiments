from AudioTest import AudioBoi
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
from string import punctuation
from collections import Counter

from ihavenomouth import text

sentences = text.split('.')

sentences = [sentence.replace('\n', ' ').replace('—', ' ') for sentence in sentences]
punc = punctuation + " "
print(punc)
# Tokenize sentence
lang_model = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
sentences = [[tok.text for tok in lang_model.tokenizer(sentence.lower()) if tok.text not in punc] for sentence in sentences]

print(sentences[13])

# Create a dictionary which maps tokens to indices (train contains all the training sentences)
# freq_list = Counter()
# for sentence in train:
#     freq_list.update(sentence)

# Convert tokens to indices
# indices = [freq_list[word] for word in sentence if word in freq_list]
