# This script is used to run and evaluate all models except the HAN model.

import sys
import data_handling
import train
from test import evaluate_model
import speech_classifiers

import torch
from sentence_transformers import SentenceTransformer, models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s","--skip",type=int)
parser.add_argument("-g","--generate",type=bool)
args = parser.parse_args()

# Check for cuda devices. This script is made to run models on the gpu, and changes need to be made in order to run on the CPU instead.
if torch.cuda.is_available():
    print("Found "+str(torch.cuda.device_count())+" cuda device(s).",file=sys.stderr)
else:
    print("No device found.",file=sys.stderr)

# Download data from https://data.riksdagen.se/data/anforanden/ as json files
# and store each year's data in one directory speech_dir/year_{year+1}
speech_dir = "speech_data"
# The years of data to load, from parliament year start_year/start_year+1 to end_year-1/end_year.
start_year = 2014
end_year = 2022

# Change skip argument depending on what data is already stored. For a first run, choose skip=0.
if args.skip == 0:
    
    print("Loading data from disk...",file=sys.stderr)
    speeches, parties, ids, year_idx = data_handling.load_data_from_disk(speech_dir,start_year,end_year,sentence_division=True)

    print("Saving speeches and so on... ",file=sys.stderr)
    data_handling.save_data(speeches, parties, ids, year_idx,"speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")
    
else:
    speeches, parties, ids, year_idx = data_handling.load_data("speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")

if args.skip <= 1:
    train_embeddings_dir = "embeddings/train_sentence_embeddings_"+str(start_year)[:2]+"_"+str(end_year)[:2]+"_multilingual.pt"
    val_test_embeddings_dir = "embeddings/val_test_sentence_embeddings_"+str(start_year)[:2]+"_"+str(end_year)[:2]+"_multilingual.pt"
    if args.generate:
        print("Initializing BERT model...",file=sys.stderr)
        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to("cuda")
        generate = [True, True]
    else:
        embedding_model = None
        generate = [False, False]

    print("Getting datasets...",file=sys.stderr)
    train_set, val_test_set = data_handling.get_sentence_datasets(speeches,parties,ids,year_idx[-1],len(speeches),
                                                       train_embeddings_dir,val_test_embeddings_dir,generate=generate,embedding_model=embedding_model)

    val_set, test_set = torch.utils.data.random_split(val_test_set,
                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

    torch.save([train_set,val_set,test_set],f="datasets_"+str(start_year)+"_"+str(end_year)+"_multilingual.pt")


else:
    print("Loading datasets...",file=sys.stderr)
    train_set,val_set,test_set = torch.load("datasets_"+str(start_year)+"_"+str(end_year)+"_multilingual.pt")

embedding_size = 768
batch_size = 32

# Uncomment to oversample the training set.
#train_set.oversample()

print("Preparing dataloaders...",file=sys.stderr)
train_loader, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[data_handling.collate_pack,\
                                                                                                                        data_handling.collate_pack,\
                                                                                                                        data_handling.collate_pack_test],train_size=1)
# An example of a model being initialized and trained.

hidden_size = 128
attention_size = 256
n_layers = 2
dropout_p = 0.3

name = "attention_hidden_"+str(hidden_size)+"_attention_"+str(attention_size)+"_n_layers_"+str(n_layers)+"_dropout_"+str(dropout_p)

#classifier = speech_classifiers.EmbeddingsAverage(embedding_size, name=name).to("cuda")
#classifier = speech_classifiers.OneLayerLSTM(embedding_size,batch_size,hidden_size,dropout_p=dropout_p,name=name).to("cuda")
#classifier = speech_classifiers.TwoLayerLSTM(embedding_size,batch_size,hidden_size,dropout_p=dropout_p,name=name).to("cuda")
classifier = speech_classifiers.AttentionNetwork(embedding_size,hidden_size,attention_size,name,n_layers=n_layers,dropout_p=dropout_p,device="cuda").cuda()

# Train the model.
trainer = train.Trainer()

epochs = 20
max_lr = 3.33e-4

print("Training...",file=sys.stderr)
train_loss, val_loss = trainer.train_model(classifier,epochs,train_loader,val_loader,max_lr=max_lr,plot=False)

# Evaluate the model.
y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,train_set.party_order)

