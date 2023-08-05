# This script takes document vectors from HAN, lowers the dimension using UMAP, and plots them.

import sys
import data_handling
import speech_classifiers

import torch
from transformers.pipelines import AutoModel,AutoTokenizer

import umap
import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

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

train_stop = year_idx[-2]
test_stop = year_idx[-1]

train_set, val_test_set = data_handling.get_word_datasets(speeches,parties,ids,train_stop,test_stop,kb_tokenizer)


val_set, test_set = torch.utils.data.random_split(val_test_set,
                                                  [val_test_set.length//2,val_test_set.length-val_test_set.length//2])

batch_size = 32

bert_dict = None

print("Preparing dataloaders...",file=sys.stderr)
_, val_loader, test_loader = data_handling.prepare_data_loaders(train_set,val_set,test_set,batch_size,collate_fns=[train_set.collate_tokenize_test,val_test_set.collate_tokenize_test,val_test_set.collate_tokenize_test],train_size=1, device="cpu")

# Define hyper parameters
hidden_size = 128
attention_size = 256
dropout_p = 0.3
num_layers = 2

model_file = "name_of_the_model.pth"

classifier = speech_classifiers.HierarchialAttentionNetwork(embedding_size,hidden_size,attention_size,kb_model,None,dropout_p=dropout_p,n_layers=num_layers).cuda()
classifier.load_state_dict(torch.load(model_file))

classifier.eval()
y_preds = np.array([])
y_tests = np.array([])
doc_vecs = None
IDs = np.array([])
print("Getting document vectors and predictions for the whole test set...",file=sys.stderr)
for i, (X_test, lengths, y_test, ID) in tqdm(enumerate(test_loader)):
    #y_probs, doc_vec, _, _ = classifier(X_test.to("cuda"), lengths, ID)
    y_probs, doc_vec, _, _ = classifier(X_test.to("cuda"), lengths)
    y_pred = y_probs.softmax(dim=1).argmax(dim=1)
    y_preds = np.concatenate((y_preds,y_pred.cpu().numpy()))
    y_tests = np.concatenate((y_tests,y_test.numpy()))
    if doc_vecs is not None:
        doc_vecs = np.concatenate((doc_vecs, doc_vec.cpu().detach().numpy()))
    else:
        doc_vecs = doc_vec.cpu().detach().numpy()

# Prepare data
print("Reducing to two dimensions using UMAP...",file=sys.stderr)
umap_data = umap.UMAP(n_neighbors=600, n_components=2, min_dist=0.0, metric='cosine').fit_transform(doc_vecs)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = y_tests

torch.save(result,"umap_data.pt")

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=5)

party_order = ["S","M","SD","V","MP","C","KD","L"]

# The colors of the parties
cmap = (mpl.colors.ListedColormap([[0.88,0.18,0.24], [0.49,0.75,0.88], [1,0.77,0.28], [0.57,0.08,0.08], [0.51,0.78,0.51], [0.19,0.65,0.20], [0.20,0.11,0.48], [0.12,0.41,0.67]]))

plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=5, cmap=cmap)
cbar = plt.colorbar(boundaries=np.arange(9)-0.5)
cbar.set_ticks(np.arange(8))
cbar.set_ticklabels(party_order,fontsize=20)
plt.show()
plt.savefig("clusters.png")

print("Done with plotting",file=sys.stderr)