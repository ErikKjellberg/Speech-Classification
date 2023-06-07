import sys
import data_handling
import train_HAN_new
from test_HAN_new import evaluate_model, show_metrics, misclassified
import speech_classifiers_new

import torch
from sentence_transformers import SentenceTransformer, models
from transformers.pipelines import AutoModel,AutoTokenizer

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-f","--firsttime",type=bool)

args = parser.parse_args()

start_year = 1993
end_year = 2022

speeches_per_year = float("inf")

if args.firsttime:
    speeches, parties, ids, year_idx = data_handling.load_data_from_disk(start_year,end_year,sentence_division=True,speeches_per_year=speeches_per_year)
    print("Saving speeches and so on... ",file=sys.stderr)
    data_handling.save_data(speeches, parties, ids, year_idx,"speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")

else:
    print("Loading data from disk...",file=sys.stderr)
    speeches, parties, ids, year_idx = data_handling.load_data(filename="speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")

embedding_size = 768

kb_tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
kb_model = AutoModel.from_pretrained('KB/bert-base-swedish-cased').to("cuda")
kb_model.requires_grad_(False)
#kb_model = torch.compile(kb_model)

print("Getting datasets...",file=sys.stderr)

train_stop = 10000
test_stop = 12000

train_set, val_test_set = data_handling.get_word_datasets(speeches,parties,ids,year_idx[-1],len(speeches),kb_tokenizer)

val_set, test_set = torch.utils.data.random_split(val_test_set,
                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

batch_size = 32



print("Preparing dataloaders...",file=sys.stderr)
train_loader, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[train_set.collate_sentences_words_test,val_test_set.collate_sentences_words_test,val_test_set.collate_sentences_words_test],train_size=1, device="cpu")

print("Initializing BERT dictionary...", file=sys.stderr)
bert_dict = data_handling.BertDictionary([train_set, val_test_set],kb_model)

print("Generating embeddings... ", file=sys.stderr)
bert_dict.generate_embeddings()

#print("Oversampling train set...",file=sys.stderr)
#train_set.oversample()

hidden_size = 128
attention_size = 512
dropout_p = 0.3
num_layers = 2

name = "HAN_hidden_"+str(hidden_size)+"_attention_"+str(attention_size)+"_n_layers_"+str(num_layers)+"_dropout_"+str(dropout_p)

classifier = speech_classifiers_new.HierarchialAttentionNetwork(embedding_size,hidden_size,attention_size,bert_dict,name,dropout_p=dropout_p,n_layers=2).cuda()
#classifier.load_state_dict(torch.load("trained_HANs/HAN_35_epochs.pth"))

trainer = train_HAN_new.Trainer()
print("Beginning training...",file=sys.stderr)

epochs = 20
max_lr = 1e-4
train_loss, val_loss = trainer.train_model(classifier,epochs,train_loader,val_loader,max_lr=max_lr,plot=False)

print("Evaluating...",file=sys.stderr)

y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,val_test_set.party_order)