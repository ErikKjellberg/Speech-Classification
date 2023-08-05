# This script evaluates an already trained model

import sys
import data_handling
from test import evaluate_model, show_metrics
import speech_classifiers

import torch

start_year = 2014
end_year = 2022

# See main.py for how to load data the first time.
print("Loading datasets...",file=sys.stderr)
train_set,val_set,test_set = torch.load("datasets_"+str(start_year)+"_"+str(end_year)+"_multilingual.pt")

embedding_size = 768
batch_size = 32

print("Preparing dataloaders...",file=sys.stderr)
_, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[data_handling.collate_pack,data_handling.collate_pack,data_handling.collate_pack_test],train_size=1, device="cpu")

hidden_size = 128
attention_size = 256
n_layers = 2
dropout_p = 0.3

model_file = "name_of_the_trained_model.pth"

print("Loading the model...",file=sys.stderr)
#classifier = speech_classifiers.EmbeddingsAverage(embedding_size, name=name).to("cuda")
#classifier = speech_classifiers.OneLayerLSTM(embedding_size,batch_size,hidden_size,dropout_p=dropout_p,name=name).to("cuda")
#classifier = speech_classifiers.TwoLayerLSTM(embedding_size,batch_size,hidden_size,dropout_p=dropout_p,name=name).to("cuda")
classifier = speech_classifiers.AttentionNetwork(embedding_size,hidden_size,attention_size,None,n_layers=n_layers,dropout_p=dropout_p,device="cuda").cuda()
classifier.load_state_dict(torch.load(model_file))

print("Evaluating...",file=sys.stderr)
y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,train_set.party_order)

cr, cm, cm_norm = show_metrics(y_tests, y_preds, label_names=train_set.party_order)