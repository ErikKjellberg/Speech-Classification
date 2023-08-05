# This script is used to train and evaluate the HAN model

import sys
import data_handling
import train
from test import evaluate_model, show_metrics, misclassified
import speech_classifiers

import torch
from transformers.pipelines import AutoModel,AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f","--firsttime",type=bool)

args = parser.parse_args()

# Download data from https://data.riksdagen.se/data/anforanden/ as json files
# and store each year's data in one directory speech_dir/year_{year+1}
speech_dir = "speech_data"
# The years of data to load, from parliament year start_year/start_year+1 to end_year-1/end_year.
start_year = 2014
end_year = 2022

if args.firsttime:
    speeches, parties, ids, year_idx = data_handling.load_data_from_disk(speech_dir,start_year,end_year,sentence_division=True)
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

train_stop = year_idx[-1]
test_stop = len(speeches)

#train_stop = 1000
#test_stop = 2000

train_set, val_test_set = data_handling.get_word_datasets(speeches,parties,ids,train_stop,test_stop,kb_tokenizer)

val_set, test_set = torch.utils.data.random_split(val_test_set,
                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

batch_size = 32
epochs = 20
max_lr = 3.33e-4

print("Preparing dataloaders...",file=sys.stderr)
train_loader, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[train_set.collate_tokenize,val_test_set.collate_tokenize,val_test_set.collate_tokenize_test],train_size=1, device="cpu")

# Define hyper parameters
hidden_size = 128
attention_size = 256
dropout_p = 0.3
num_layers = 2

name = "HAN_hidden_"+str(hidden_size)+"_attention_"+str(attention_size)+"_n_layers_"+str(num_layers)+"_dropout_"+str(dropout_p)+"_years_"+str(start_year)+"_"+str(end_year)+"_maxlr_"+str(max_lr)+"_filtered"
#name = "sillymodel"

classifier = speech_classifiers.HierarchialAttentionNetwork(embedding_size,hidden_size,attention_size,kb_model,name,dropout_p=dropout_p,n_layers=num_layers).cuda()
#classifier.load_state_dict(torch.load("trained_HANs/HAN_hidden_128_attention_256_n_layers_2_dropout_0.3_years_2014_2022_maxlr_0.001_8_epochs.pth"))

trainer = train.Trainer()
print("Beginning training...",file=sys.stderr)


train_loss, val_loss = trainer.train_model(classifier,epochs,train_loader,val_loader,max_lr=max_lr,plot=False)

print("Evaluating...",file=sys.stderr)

y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,val_test_set.party_order)
