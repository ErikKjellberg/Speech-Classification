import os
import json
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
import spacy
import sys
import re

# A dataset which only contains speeches divided into sentences
class SentenceEmbeddingDataset(Dataset):
    # this takes instead as input a list of speeches, each containing a list of sentences
    def __init__(self,speeches,parties,ids,start_index,stop_index,generate,embedding_model=None,file_name=""):
        assert len(speeches) == len(parties)
        self.length = stop_index-start_index
        self.party_order = ["S","M","SD","V","MP","C","KD","L"]
        self.party_indices = {}
        self.embeddings_file_name = file_name
        for i, p in enumerate(self.party_order):
            self.party_indices[p] = i
        print(self.party_indices,file=sys.stderr)
        self.party_letters = parties[start_index:stop_index]
        for i, p in enumerate(self.party_letters):
            if p == "KDS":
                self.party_letters[i] = "KD"
            elif p == "FP":
                self.party_letters[i] = "L"
        self.party_numbers = torch.Tensor([self.party_indices.get(p) for p in self.party_letters]).long()
        self.ids = ids[start_index:stop_index]
        no_sentences = 0
        sentence_list = []
        count = 0
        # Make a list of all sentences to use for creating embeddings.
        for sp in speeches[start_index:stop_index]:
            count += 1
            for se in sp:
                sentence_list.append(se)
                no_sentences += 1
        if generate and embedding_model!=None:
            print("Encoding sentences...",file=sys.stderr)
            sentence_embeddings = embedding_model.encode(sentence_list,show_progress_bar=True)
            # Save the data to load later
            torch.save(sentence_embeddings,f=self.embeddings_file_name,pickle_protocol=4)
        else:
            print("Loading embeddings from file...",file=sys.stderr)
            sentence_embeddings = torch.load(self.embeddings_file_name)
        self.speeches = []
        self.speech_texts = []
        index = 0
        for i_s, sp in enumerate(speeches[start_index:stop_index]):
            sentence_list = []
            self.speech_texts.append(sp)
            for i_se, se in enumerate(sp):
                sentence_list.append(sentence_embeddings[index])
                index += 1
            self.speeches.append(sentence_list)
        assert index == no_sentences
        print("Done. Amount of speeches: "+str(len(self.speeches))+", amount of sentences: "+str(len(sentence_embeddings)),file=sys.stderr)

    def oversample(self):
        ros = RandomOverSampler(random_state=0)
        X = np.array(list(zip(self.speech_texts, self.speeches, self.ids, self.party_letters)),dtype=object)
        y = self.party_numbers
        X_reshaped, y_reshaped = ros.fit_resample(X,y)
        self.speech_texts, self.speeches, self.ids, self.party_letters = list(zip(*X_reshaped))
        self.party_numbers = y_reshaped
        self.length = len(self.speeches)

    def get_item_verbose(self, index):
        return self.speech_texts[index],self.speeches[index],self.party_letters[index],self.party_numbers[index],self.ids[index]

    def __len__(self):
        return self.length
    
    # Get speech at a certain index
    def __getitem__(self, index):
        speech = np.array(self.speeches[index])
        party = self.party_numbers[index]
        return torch.Tensor(speech), party, self.ids[index]   


# A dataset which uses a BERT tokenizer to return encoded sentences in the collate function
class WordEmbeddingDataset(Dataset):
    # Speeches is here a list containing speeches, which are lists of sentences.
    def __init__(self,speeches,parties,ids,start_index,stop_index,tok):
        assert len(speeches) == len(parties)
        self.length = stop_index-start_index
        self.party_order = ["S","M","SD","V","MP","C","KD","L"]
        self.party_indices = {}
        for i, p in enumerate(self.party_order):
            self.party_indices[p] = i
        
        print(self.party_indices,file=sys.stderr)
        self.party_letters = parties[start_index:stop_index]
        for i, p in enumerate(self.party_letters):
            if p == "KDS":
                self.party_letters[i] = "KD"
            elif p == "FP":
                self.party_letters[i] = "L"
        self.party_numbers = [self.party_indices.get(p) for p in self.party_letters]
        self.ids = ids[start_index:stop_index]
        self.speech_texts = speeches[start_index:stop_index]
        self.tok = tok
        print("Done. Amount of speeches: "+str(len(self.speech_texts)),file=sys.stderr)

    def get_item_verbose(self, index):
        return self.speech_texts[index],self.party_letters[index],self.party_numbers[index],self.ids[index]

    def __len__(self):
        return self.length
    
    # Get speech at a certain index
    def __getitem__(self, index):
        speech = np.array(self.speech_texts[index])
        party = self.party_numbers[index]
        id = self.ids[index]
        return speech, party, id
    
    # Input data should be a list of speech texts, each a list of sentences. 
    # The output is the speeches encoded, a list of the amount of sentences in each speech as well as the corresponding party.
    def collate_tokenize(self,data):
        speeches, parties, _ = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
        encoded_speeches = []
        all_sentences = []
        speech_lengths = [len(s) for s in speeches]
        for speech in speeches:
            for sentence in speech:
                all_sentences.append(sentence)
        # Each document will be padded according to the maximum amount of sentences in the whole batch
        encoded_speeches = self.tok.batch_encode_plus(all_sentences,padding=True,return_tensors="pt")
        return encoded_speeches, speech_lengths, torch.tensor(parties)
    
    # This works like the previous function except it also keeps the IDs of the speeches
    def collate_tokenize_test(self,data):
        speeches, parties, ids = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
        encoded_speeches = []
        all_sentences = []
        speech_lengths = [len(s) for s in speeches]
        for speech in speeches:
            for sentence in speech:
                all_sentences.append(sentence)
        # Each document will be padded according to the maximum amount of sentences in the whole batch
        encoded_speeches = self.tok.batch_encode_plus(all_sentences,padding=True,return_tensors="pt")
        return encoded_speeches, speech_lengths, torch.tensor(parties),ids

    # Specific collate function for visualizing
    def collate_tokenize_visualize(self,data):
        speeches, parties, ids = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
        encoded_speeches = []
        all_sentences = []
        speech_lengths = [len(s) for s in speeches]
        for speech in speeches:
            for sentence in speech:
                all_sentences.append(sentence)
        # Each document will be padded according to the maximum amount of sentences in the whole batch
        encoded_speeches = self.tok.batch_encode_plus(all_sentences,padding=True,return_tensors="pt")
        return speeches, encoded_speeches, speech_lengths, torch.tensor(parties),ids

    # For use when manually tokenizing after collating
    def collate_for_shap(self,data):
        speeches, parties, ids = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
        #encoded_speeches = []
        all_sentences = []
        for speech in speeches:
            for sentence in speech:
                all_sentences.append(sentence)
        return speeches, torch.tensor(parties), ids


    def oversample(self):
        ros = RandomOverSampler(random_state=0)
        X = np.array(list(zip(self.speech_texts, self.ids, self.party_letters)),dtype=object)
        y = self.party_numbers
        X_reshaped, y_reshaped = ros.fit_resample(X,y)
        self.speech_texts, self.ids, self.party_letters = list(zip(*X_reshaped))
        self.party_numbers = y_reshaped
        self.length = len(self.speech_texts)    

def collate_pack(data):
    speeches, parties, _ = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
    lengths = torch.tensor([len(s) for s in speeches])
    input = pack_sequence(speeches)
    output = torch.tensor(parties)
    return input, lengths, output

def collate_pack_test(data):
    speeches, parties, ids = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
    lengths = torch.tensor([len(s) for s in speeches])
    input = pack_sequence(speeches)
    output = torch.tensor(parties)
    return input, lengths, output, ids

# Used for storing BERT embeddings for later use.
# The HAN model in speech_classifiers.py can be modified to use this class instead of 
# calculating embeddings for every prediction
class BertDictionary:
    def __init__(self, bert_model, datasets=None):
        self.bert_model = bert_model
        self.embeddings_dict = {}
        if datasets is not None:
            self.generate_embeddings(datasets)

    def generate_embeddings(self, datasets):
        data_loaders = []
        for dataset in datasets:
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_tokenize_test,generator=torch.Generator(device="cpu"))
            data_loaders.append(data_loader)
        for dataloader in data_loaders:
            for i, (doc, _, _, id) in enumerate(tqdm(dataloader)):
                doc = doc.to("cuda")
                id = id[0]
                with torch.no_grad():
                    embedded_doc_padded = self.bert_model(return_dict=True, **doc)["last_hidden_state"]
                    self.embeddings_dict[id] = embedded_doc_padded.to("cpu")

    def set_embeddings(self,embeddings_dict):
        for k in embeddings_dict.keys():
            self.embeddings_dict[k] = embeddings_dict.get(k)

    def save_dict(self,filename):
        torch.save(self.embeddings_dict,filename)

    def load_dict(self,filename):
        embeddings_dict = torch.load(filename)
        for k in embeddings_dict.keys():
            print(k)
            self.embeddings_dict[k] = embeddings_dict[k]

    def get_embedding(self,id):
        return self.embeddings_dict.get(id)



# Load speeches from json files in data_dirs
def load_data_from_disk(speech_dir, start_year, end_year, possible_parties=["S","M","SD","V","MP","C","KD","L","KDS","FP"], sentence_division=False, speeches_per_year=float("inf"), filter_party_names=False):
    speeches = []
    parties = []
    ids = []
    possible_parties = possible_parties

    if sentence_division:
        nlp = spacy.load('sv_core_news_lg')

    # The directories where data is located, one for each year
    data_dirs = [speech_dir+"/"+str(i)[2:]+"_"+str(i+1)[2:]+"/" for i in range(start_year,end_year)]
    year_indices = []
    data = []
    i = 0
    for year, data_dir in enumerate(data_dirs):
        year_indices.append(len(speeches))
        for k, file in tqdm(enumerate(os.listdir(data_dir))):
            if file.endswith(".json"):
                with open(data_dir+file,"r",encoding="utf-8-sig") as f:
                    data = json.load(f)
                    speech = data["anforande"]["anforandetext"]
                    party = data["anforande"]["parti"]
                    id = data["anforande"]["dok_id"]+"-"+data["anforande"]["anforande_nummer"]
                    # If it is a party who has spoken,
                    if party is not None and party.upper() in possible_parties:
                        paragraphs = []
                        if sentence_division:
                            if 2014 <= year+start_year <= 2022:
                                soup = BeautifulSoup(speech, 'html.parser')
                                speech_text = " ".join([p.get_text() for p in soup.find_all("p")])
                                speech_text = re.sub("STYLEREF Kantrubrik \\\\\* MERGEFORMAT","",speech_text)
                            elif 2003 <= year+start_year <= 2013:
                                speech_text = re.sub("\\r\\n"," ",speech)
                            elif 1993 <= year+start_year <= 2002:
                                speech_text = re.sub("\-\\n"," ",speech)
                                speech_text = re.sub("\\n"," ",speech_text)
                            else:
                                speech_text = ""

                            # If one wants party names to be removed in speeches, set filter_party_names=True
                            if filter_party_names:
                                parties_to_filter_det = ["Socialdemokraterna","Vänsterpartiet","Miljöpartiet","Centerpartiet","Folkpartiet","Liberalerna","Kristdemokraterna","Moderaterna","Sverigedemokraterna"]
                                parties_to_filter_indet_pl = ["socialdemokrater","vänsterpartister","miljöpartister","centerpartister","folkpartister","liberaler","kristdemokrater","moderater","sverigedemokrater"]
                                parties_to_filter_indet_sing = ["socialdemokrat","Vänsterpartist","miljöpartist","centerpartist","folkpartist","kristdemokrat","moderat","sverigedemokrat"]
                                det_regexp = "|".join(parties_to_filter_det)
                                det_replace = "Partiet"

                                indet_pl_regexp = "|".join(parties_to_filter_indet_pl)
                                indet_pl_replace = "partister"

                                indet_sing_regexp = "|".join(parties_to_filter_indet_sing)
                                indet_sing_replace = "partist"

                                speech_text = re.sub(det_regexp,det_replace,speech_text)
                                speech_text = re.sub(indet_pl_regexp,indet_pl_replace,speech_text)
                                speech_text = re.sub(indet_sing_regexp,indet_sing_replace,speech_text)
                            speech_text = re.sub("\(Applåder\)","",speech_text)
                            
                            if speech_text != "":
                                sentences = [str(s) for s in nlp(speech_text).sents]
                                speeches.append(sentences)
                                parties.append(party.upper())
                                ids.append(id)
                        else:
                            for paragraph in soup.find_all("p"):
                                text = paragraph.get_text()
                                if text != "":
                                    paragraphs.append([s.string.strip() for s in nlp(text).sents])
                            if paragraphs != []:
                                parties.append(party)
                                speeches.append(paragraphs)
                                ids.append(id)
            if len(speeches) % speeches_per_year == 0:
                break
        print("Year "+str(year+start_year)+"completed.",file=sys.stderr)
    assert len(parties) == len(speeches)
    return speeches, parties, ids, year_indices

def save_data(speeches, parties, ids, year_idx, filename):
    torch.save([speeches,parties,ids,year_idx],f=filename)

def load_data(filename):
    return torch.load(filename)

def get_sentence_datasets(speeches,parties,ids,train_stop,test_stop,train_embeddings,val_test_embeddings,generate=[True,True],embedding_model=None,split_val_test=True,device="cpu"):
    train_set = SentenceEmbeddingDataset(speeches,parties,ids,0,train_stop,generate[0],embedding_model,file_name=train_embeddings)
    val_test_set = SentenceEmbeddingDataset(speeches,parties,ids,train_stop,test_stop,generate[1],embedding_model,file_name=val_test_embeddings)
    return train_set, val_test_set

def get_word_datasets(speeches,parties,ids,train_stop,test_stop,tok,split_val_test=True):
    train_set = WordEmbeddingDataset(speeches,parties,ids,0,train_stop,tok)
    val_test_set = WordEmbeddingDataset(speeches,parties,ids,train_stop,test_stop,tok)
    return train_set, val_test_set
    

def prepare_data_loaders(train,val,test,batch_size,collate_fns=[collate_pack,collate_pack,collate_pack_test],train_size=1,device="cpu"):
    # Train on only a proportion of the set
    if train_size < 1:
        train_set, _ = torch.utils.data.random_split(train,[train_size,1-train_size])
    else:
        train_set = train
    val_set = val
    test_set = test

    if isinstance(batch_size,int):
        batch_size = [batch_size, batch_size, batch_size]

    train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=True, collate_fn=collate_fns[0],generator=torch.Generator(device=device))
    val_loader = DataLoader(val_set, batch_size=batch_size[1], shuffle=True, collate_fn=collate_fns[1],generator=torch.Generator(device=device))
    test_loader = DataLoader(test_set, batch_size=batch_size[2], shuffle=True, collate_fn=collate_fns[2],generator=torch.Generator(device=device))
    return train_loader, val_loader, test_loader

"""def assert_correct_labeling(dataset, random_samples):
    # If ID starts with 6, its from 18_19, with 7 19_20 and so on.
    ID_to_dir = {"H6":"speech_data/18_19/","H7":"speech_data/19_20/","H8":"speech_data/20_21/","H9":"speech_data/21_22/"}
    for i in range(random_samples):
        index = int(np.random.rand()*len(dataset))
        speech_text, speech_embedding, party_text, party_number, ID = dataset.get_item_verbose(index)
        ID_start = ID[:2]
        dir = ID_to_dir[ID_start]
        with open(dir+ID+".json","r") as f:
            data = json.load(f)
            speech = data["anforande"]["anforandetext"]
            party = data["anforande"]["parti"]
            id = data["anforande"]["dok_id"]
            # Make sure that it is the right party
            try:
                assert party == party_text
            except AssertionError:
                print("Expected ", party," got ",party_text)
            # Make sure that it is the right speech
            print(speech_text)
            print(speech)"""
