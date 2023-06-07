import sys
import data_handling
import train
from test import evaluate_model, show_metrics, misclassified
import speech_classifiers

import torch
from sentence_transformers import SentenceTransformer, models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s","--skip",type=int)
parser.add_argument("-g","--generate",type=bool)
args = parser.parse_args()

if torch.cuda.is_available():
    print("Found "+str(torch.cuda.device_count())+" cuda device(s).",file=sys.stderr)
    #torch.set_default_device("cuda")
else:
    print("No device found.",file=sys.stderr)

start_year = 2015
end_year = 2017


if args.skip == 0:

    print("Loading data from disk...",file=sys.stderr)
    speeches, parties, ids, year_idx = data_handling.load_data_from_disk(start_year,end_year,sentence_division=True)

    print("Saving speeches and so on... ",file=sys.stderr)
    data_handling.save_data(speeches, parties, ids, year_idx,"speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")
    
else:
    speeches, parties, ids, year_idx = data_handling.load_data("speeches_parties_ids_year_ids_"+str(start_year)+"_"+str(end_year)+".pt")

if args.skip <= 1:
    train_embeddings_dir = "embeddings/train_sentence_embeddings_"+str(start_year)[:2]+"_"+str(end_year)[:2]+"_multilingual.pt"
    val_test_embeddings_dir = "embeddings/val_test_sentence_embeddings_"+str(start_year)[:2]+"_"+str(end_year)[:2]+"_multilingual.pt"
    #embedding_size = torch.load(train_embeddings_dir).shape[1]
    if args.generate:
        print("Initializing BERT model...",file=sys.stderr)
        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        generate = [True, True]
    else:
        embedding_model = None
        generate = [False, False]
    embedding_size = 768

    print("Getting datasets...",file=sys.stderr)
    train_set, val_test_set = data_handling.get_sentence_datasets(speeches,parties,ids,year_idx[-1],len(speeches),
                                                       train_embeddings_dir,val_test_embeddings_dir,generate=generate,embedding_model=embedding_model)

    val_set, test_set = torch.utils.data.random_split(val_test_set,
                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

    torch.save([train_set,val_set,test_set],f="datasets_"+str(start_year)[:2]+"_"+str(end_year)[:2]+"_multilingual.pt")

else:
    train_set,val_set,test_set = torch.load("datasets_"+str(start_year)[:2]+"_"+str(end_year)[:2]+"_multilingual.pt")

embedding_size = 768
batch_size = 32

# Oversampling.
#train_set.oversample()

print("Preparing dataloaders...",file=sys.stderr)
train_loader, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[data_handling.collate_pack,\
                                                                                                                        data_handling.collate_pack,\
                                                                                                                        data_handling.collate_pack_test],train_size=1)
classifier = speech_classifiers.AttentionNetwork(embedding_size,128,256,n_layers=2,dropout_p=0.3,device="cuda").cuda()
#classifier.load_state_dict(torch.load("trained_models/OneLayerLSTM_50_epochs_balanced.pth"))

trainer = train.Trainer()

epochs = 50
max_lr = 3.33e-4

print("Training...",file=sys.stderr)
train_loss, val_loss = trainer.train_model(classifier,epochs,train_loader,val_loader,max_lr=max_lr,plot=False)


y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,train_set.party_order)

