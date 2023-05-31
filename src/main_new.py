#%%
import sys
import data_handling
import train
from test import evaluate_model, show_metrics, misclassified
import speech_classifiers

import torch
from sentence_transformers import SentenceTransformer, models
from transformers.pipelines import AutoModel,AutoTokenizer

device = "cpu"
#torch.set_default_device(device)

#%%
print(torch.cuda.current_device(),file=sys.stderr)

#%%

#data_dir = "speech_data/"
#years = ["18_19","19_20","20_21","21_22"]
#directories = [data_dir+year+"/" for year in years]

print("Loading data from disk...",file=sys.stderr)
speeches, parties, ids, year_idx = data_handling.load_data(filename="../speeches_parties_ids_year_ids.pt")

#train_embeddings_dir = "embeddings/train_sentence_embeddings_18_22_multilingual.pt"
#val_test_embeddings_dir = "embeddings/val_test_sentence_embeddings_18_22_multilingual.pt"
#embedding_size = torch.load(train_embeddings_dir).shape[1]

embedding_size = 768

kb_tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
kb_model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

print("Getting datasets...",file=sys.stderr)
train_set, val_test_set = data_handling.get_word_datasets(speeches,parties,ids,10,12,kb_tokenizer)

#train_set = train_set.to(device)
#val_test_set = val_test_set.to(device)

#val_set, test_set = torch.utils.data.random_split(val_test_set,
#                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

batch_size = 256

print("Preparing dataloaders...",file=sys.stderr)
train_loader, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_test_set,val_test_set,batch_size,collate_fns=[train_set.collate_sentences_words]*3,train_size=1, device=device)


classifier = speech_classifiers.HierarchialAttentionNetwork(embedding_size,64,64,kb_model)
#classifier.load_state_dict(torch.load("trained_models/OneLayerLSTM_50_epochs_balanced.pth"))

trainer = train.Trainer()

#%%
epochs = 1
train_loss, val_loss = trainer.train_model(classifier,epochs,train_loader,val_loader,max_lr=1e-3,plot=False)
#%%

y_tests, y_preds, IDs = evaluate_model(classifier,test_loader,val_test_set.party_order)
# %%
