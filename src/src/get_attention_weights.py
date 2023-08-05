# This script evaluates a HAN model on a set of speeches and saves the attention weights.
# These can be viewed in the notebook named visualize_attention.ipynb in this repository.

import sys
import data_handling
import speech_classifiers

import torch
from transformers.pipelines import AutoModel,AutoTokenizer

import numpy as np

from tqdm import tqdm

import json

start_year = 2014
end_year = 2022

# See main_HAN.py for how to load data the first time.
print("Loading data from disk...",file=sys.stderr)
speeches, parties, ids, year_idx = data_handling.load_data(filename="speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")

embedding_size = 768

kb_tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
kb_model = AutoModel.from_pretrained('KB/bert-base-swedish-cased').to("cuda")
kb_model.requires_grad_(False)

print("Getting datasets...",file=sys.stderr)

train_stop = year_idx[-1]
test_stop = len(speeches)

train_set, val_test_set = data_handling.get_word_datasets(speeches,parties,ids,train_stop,test_stop,kb_tokenizer)

val_set, test_set = torch.utils.data.random_split(val_test_set,
                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

batch_size = 1

bert_dict = None

print("Preparing dataloaders...",file=sys.stderr)
_, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[train_set.collate_tokenize_test,val_test_set.collate_tokenize_visualize,val_test_set.collate_tokenize_visualize],train_size=1, device="cpu")

# Define hyper parameters
hidden_size = 128
attention_size = 256
dropout_p = 0.3
num_layers = 2

model_file = "name_of_the_model.pth"

print("Loading the model...",file=sys.stderr)
classifier = speech_classifiers.HierarchialAttentionNetwork(embedding_size,hidden_size,attention_size,kb_model,None,dropout_p=dropout_p,n_layers=num_layers).cuda()
classifier.load_state_dict(torch.load("results/HAN/8_years_20_epochs_filtered/HAN_hidden_128_attention_256_n_layers_2_dropout_0.3_years_2014_2022_maxlr_0.000333_filtered_20_epochs.pth"))

speeches_to_print = 100

classifier.eval()
y_preds = np.array([])
doc_vecs = None
IDs = np.array([])

weighted_texts = {
    "sentences" : [],
    "sent_weights" : [],
    "tok_sentences" : [],
    "word_weights" : [],
    "pred" : [],
    "true" : []
}

print("Getting document vectors and predictions for the whole test set...",file=sys.stderr)
for i, (sentences, X_test, lengths, y_test, ID) in tqdm(enumerate(test_loader)):
    # With batch size 1, this will be just one document
    y_probs, _, sentence_weights, all_word_weights = classifier(X_test.to("cuda"), lengths)
    y_pred = y_probs.softmax(dim=1).argmax(dim=1)
    y_preds = np.concatenate((y_preds,y_pred.cpu().numpy()))
    amount_sentences = lengths[0]
    sent_weights = sentence_weights[:amount_sentences].tolist()[0]
    weighted_texts["sentences"].append(sentences[0].tolist())
    weighted_texts["sent_weights"].append([np.round(item,3) for item in sent_weights])
    weighted_texts["tok_sentences"].append([])
    weighted_texts["word_weights"].append([])
    weighted_texts["pred"].append(train_set.party_order[int(y_pred.cpu().numpy()[0])])
    weighted_texts["true"].append(train_set.party_order[int(y_test[0])])
    for s in range(amount_sentences):
        amount_words = torch.sum(X_test["attention_mask"][s])
        sentence = kb_tokenizer.convert_ids_to_tokens(X_test["input_ids"][s])[:amount_words]
        word_weights = all_word_weights[s][:amount_words].tolist()
        weighted_texts["tok_sentences"][-1].append(sentence)
        weighted_texts["word_weights"][-1].append(word_weights)
    if i >= speeches_to_print:
        break

with open('weighted_documents_filtered.json', 'w') as outfile:
    json.dump(weighted_texts, outfile, ensure_ascii=False)