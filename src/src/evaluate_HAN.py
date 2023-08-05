# This script evaluates an already trained HAN model

import sys
import data_handling
from test import evaluate_model, show_metrics, misclassified
import speech_classifiers

import torch
from transformers.pipelines import AutoModel,AutoTokenizer

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

batch_size = 32

print("Preparing dataloaders...",file=sys.stderr)
_, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[train_set.collate_tokenize,val_test_set.collate_tokenize_test,val_test_set.collate_tokenize_test],train_size=1, device="cpu")

hidden_size = 128
attention_size = 256
dropout_p = 0.3
num_layers = 2

model_file = "name_of_the_model.pth"

print("Loading the model...",file=sys.stderr)
classifier = speech_classifiers.HierarchialAttentionNetwork(embedding_size,hidden_size,attention_size,kb_model,None,dropout_p=dropout_p,n_layers=2).cuda()
classifier.load_state_dict(torch.load(model_file))

print("Evaluating...",file=sys.stderr)

y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,val_test_set.party_order)

cr, cm, cm_norm = show_metrics(y_tests, y_preds, label_names=val_test_set.party_order)