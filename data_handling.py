import os
import json
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
import spacy
import sys

# This Dataset is supposed to return a speech divided into paragraphs as well as the party.

class ParagraphDataset(Dataset):
    def __init__(self,speeches,parties,ids,start_index,stop_index,generate,embedding_model=None,file_name=""):
        assert len(speeches) == len(parties)
        self.length = stop_index-start_index
        self.party_order = ["S","M","SD","V","MP","C","KD","L"]
        self.party_indices = {}
        self.embeddings_file_name = file_name
        for i, p in enumerate(self.party_order):
            self.party_indices[p] = i
        self.party_letters = parties[start_index:stop_index]
        self.party_numbers = torch.Tensor([self.party_indices.get(p) for p in self.party_letters]).long()
        self.ids = ids[start_index:stop_index]
        no_paragraphs = 0
        paragraph_list = []
        count = 0
        for sp in speeches[start_index:stop_index]:
            count += 1
            for pa in sp:
                paragraph_list.append(pa)
                no_paragraphs += 1
        if generate and embedding_model!=None:
            print("Encoding sentences...",file=sys.stderr)
            paragraph_embeddings = embedding_model.encode(paragraph_list,show_progress_bar=True)
            # Save the data to load later
            torch.save(paragraph_embeddings,f=self.embeddings_file_name)
        else:
            print("Loading embeddings from file...",file=sys.stderr)
            paragraph_embeddings = torch.load(self.embeddings_file_name)
        self.speeches = []
        self.speech_texts = []
        index = 0
        for i_s, sp in enumerate(speeches[start_index:stop_index]):
            self.speeches.append([])
            self.speech_texts.append(sp)
            for i_p, pa in enumerate(sp):
                self.speeches[-1].append(paragraph_embeddings[index])
                index += 1
        assert index == no_paragraphs
        print("Done. Amount of speeches: "+str(len(self.speeches))+", amount of paragraphs: "+str(len(paragraph_embeddings)),file=sys.stderr)

    def oversample(self):
        ros = RandomOverSampler(random_state=0)
        X = list(zip(self.speech_texts, self.speeches, self.ids, self.party_letters))
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
        # The index list decides which paragraphs to use
        speech = np.array(self.speeches[index])
        #print(speech[0].shape,speech[1].shape)
        party = self.party_numbers[index]
        #return torch.Tensor(speech), torch.Tensor(party)
        return torch.Tensor(speech), party, self.ids[index]
    

class SentenceParagraphDataset(Dataset):
    # this takes instead as input a list of speeches, each containing a list of paragraphs which each contain a list of sentences
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
        self.party_numbers = torch.Tensor([self.party_indices.get(p) for p in self.party_letters]).long()
        self.ids = ids[start_index:stop_index]
        no_sentences = 0
        sentence_list = []
        count = 0
        # Make a list of all sentences to use for creating embeddings.
        for sp in speeches[start_index:stop_index]:
            count += 1
            for pa in sp:
                for se in pa:
                    sentence_list.append(se)
                    no_sentences += 1
        if generate and embedding_model!=None:
            print("Encoding sentences...",file=sys.stderr)
            sentence_embeddings = embedding_model.encode(sentence_list,show_progress_bar=True)
            # Save the data to load later
            torch.save(sentence_embeddings,f=self.embeddings_file_name)
        else:
            print("Loading embeddings from file...",file=sys.stderr)
            sentence_embeddings = torch.load(self.embeddings_file_name)
        self.speeches = []
        self.speech_texts = []
        index = 0
        for i_s, sp in enumerate(speeches[start_index:stop_index]):
            self.speeches.append([])
            self.speech_texts.append(sp)
            for i_p, pa in enumerate(sp):
                self.speeches[-1].append([])
                for i_se, se in enumerate(pa):
                    self.speeches[-1][-1].append(sentence_embeddings[index])
                    index += 1
        assert index == no_sentences
        print("Done. Amount of speeches: "+str(len(self.speeches))+", amount of sentences: "+str(len(sentence_embeddings)),file=sys.stderr)

    def oversample(self):
        ros = RandomOverSampler(random_state=0)
        X = list(zip(self.speech_texts, self.speeches, self.ids, self.party_letters))
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
        # The index list decides which paragraphs to use
        speech = np.array(self.speeches[index])
        #print(speech[0].shape,speech[1].shape)
        party = self.party_numbers[index]
        #return torch.Tensor(speech), torch.Tensor(party)
        return torch.Tensor(speech), party, self.ids[index]
    

# A dataset which only contains speeches divided into sentences
class SentenceDataset(Dataset):
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
            torch.save(sentence_embeddings,f=self.embeddings_file_name)
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
        X = list(zip(self.speech_texts, self.speeches, self.ids, self.party_letters))
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
        # The index list decides which paragraphs to use
        speech = np.array(self.speeches[index])
        #print(speech[0].shape,speech[1].shape)
        party = self.party_numbers[index]
        #return torch.Tensor(speech), torch.Tensor(party)
        return torch.Tensor(speech), party, self.ids[index]
    


# A dataset which only contains speeches divided into sentences
class WordDataset(Dataset):
    # this takes instead as input a list of speeches, each containing a list of sentences
    def __init__(self,speeches,parties,ids,start_index,stop_index,tok):
        assert len(speeches) == len(parties)
        self.length = stop_index-start_index
        self.party_order = ["S","M","SD","V","MP","C","KD","L"]
        self.party_indices = {}
        for i, p in enumerate(self.party_order):
            self.party_indices[p] = i
        print(self.party_indices,file=sys.stderr)
        self.party_letters = parties[start_index:stop_index]
        self.party_numbers = [self.party_indices.get(p) for p in self.party_letters]
        self.ids = ids[start_index:stop_index]
        self.speech_texts = speeches[start_index:stop_index]
        self.tok = tok
        print("Done. Amount of speeches: "+str(len(self.speech_texts)))

    def get_item_verbose(self, index):
        return self.speech_texts[index],self.speeches[index],self.party_letters[index],self.party_numbers[index],self.ids[index]

    def __len__(self):
        return self.length
    
    # Get speech at a certain index
    def __getitem__(self, index):
        # The index list decides which paragraphs to use
        speech = np.array(self.speech_texts[index])
        #print(speech[0].shape,speech[1].shape)
        party = self.party_numbers[index]
        #return torch.Tensor(speech), torch.Tensor(party)
        return speech, party, self.ids[index]
    
    # Input data is a list of speech texts, each a list of sentences. 
# Want to return a list of encoded speeches?
# For use with WordDataset
    def collate_sentences_words(self,data):
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
    



def collate_fn(data):
    #print("Before padding (first in embedding):")
    #print([d[0][:,0] for d in data])
    input = torch.nn.utils.rnn.pad_sequence([torch.flip(d[0],[0]) for d in data], batch_first=True)
    input = torch.flip(input,[1])
    #print("After padding: ")
    #print(input[:,:,0])
    output = torch.tensor(np.array([d[1] for d in data]))
    return input, output


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

# Each speech should be represented as a 3d tensor of dim (no_paragraphs, max_seq_len, embedding_size)
def collate_pack_sentences(data):
    speeches, parties, _ = list(zip(*sorted(data, key = lambda d : -d[0].shape[0])))
    # Each speech is here a list of paragraphs, which in turn is a list of sentence embeddings.
    max_seq_len = 0
    for speech in speeches:
        max_seq_len = max(max_seq_len, len(max(speech,key=len)))
    


# Load speeches from json files in data_dirs
def load_data_from_disk(data_dirs, possible_parties=["S","M","SD","V","MP","C","KD","L"], sentence_division=False, speeches_per_year=float("inf")):
    speeches = []
    parties = []
    ids = []
    all_sentences = []
    all_paragraphs = []
    possible_parties = possible_parties

    if sentence_division:
        nlp = spacy.load('sv_core_news_lg')

    # The directory where data is located
    year_indices = []
    data = []
    i = 0
    for year, data_dir in enumerate(data_dirs):
        year_indices.append(len(speeches))
        for k, file in tqdm(enumerate(os.listdir(data_dir))):
            if file.endswith(".json"):
                with open(data_dir+file,"r") as f:
                    data = json.load(f)
                    speech = data["anforande"]["anforandetext"]
                    party = data["anforande"]["parti"]
                    id = data["anforande"]["dok_id"]+"-"+data["anforande"]["anforande_nummer"]
                    # If it is a party who has spoken,
                    if party in possible_parties:
                        soup = BeautifulSoup(speech, 'html.parser')
                        paragraphs = []
                        if sentence_division:
                            text = " ".join([p.get_text() for p in soup.find_all("p")])
                            sentences = [str(s) for s in nlp(text).sents]
                            speeches.append(sentences)
                            parties.append(party)
                            ids.append(id)
                        else:
                            for paragraph in soup.find_all("p"):
                                text = paragraph.get_text()
                                if text != "":
                                    paragraphs.append([s.string.strip() for s in nlp(text).sents])
                                    """sentences = []
                                        for s in re.split("\. |\! |\? ",text):
                                            if s != "":
                                                sentences.append(s)
                                                all_sentences.append(s)
                                        if sentences != []:
                                            all_paragraphs.append(text)"""
                                    
                                    """else:
                                        all_paragraphs.append(text)
                                        paragraphs.append(text)"""
                                    #print("This paragraph consisted of "+str(len(paragraphs[-1]))+"sentences.")
                            if paragraphs != []:
                                parties.append(party)
                                speeches.append(paragraphs)
                                ids.append(id)
            if len(speeches) % speeches_per_year == 0:
                break
        print("Year "+str(year)+"completed.",file=sys.stderr)
    assert len(parties) == len(speeches)
    return speeches, parties, ids, year_indices





def save_data(speeches, parties, ids, year_idx, filename):
    torch.save([speeches,parties,ids,year_idx],f=filename)

def load_data(filename):
    return torch.load(filename)


def get_sentence_datasets(speeches,parties,ids,train_stop,test_stop,train_embeddings,val_test_embeddings,generate=[True,True],embedding_model=None,split_val_test=True,device="cpu"):
    train_set = SentenceDataset(speeches,parties,ids,0,train_stop,generate[0],embedding_model,file_name=train_embeddings)
    val_test_set = SentenceDataset(speeches,parties,ids,train_stop,test_stop,generate[1],embedding_model,file_name=val_test_embeddings)
    return train_set, val_test_set


def get_word_datasets(speeches,parties,ids,train_stop,test_stop,tok,split_val_test=True):
    train_set = WordDataset(speeches,parties,ids,0,train_stop,tok)
    val_test_set = WordDataset(speeches,parties,ids,train_stop,test_stop,tok)
    return train_set, val_test_set
    """if split_val_test:
        val_test_speeches = speeches[train_stop:]
        val_test_parties = parties[train_stop:]
        val_test_ids = ids[train_stop:]
        assert len(val_test_parties) == len(val_test_speeches)

        # Split into val and test set randomly
        permutation = np.random.permutation(len(val_test_speeches))
        val_indices = permutation[:permutation.shape[0]//2]
        test_indices = permutation[permutation.shape[0]//2:]
        assert val_indices.shape[0]+test_indices.shape[0] == len(val_test_speeches)

        val_speeches = list(np.asarray(val_test_speeches)[val_indices])
        val_parties = list(np.asarray(val_test_parties)[val_indices])
        val_ids = list(np.asarray(val_test_ids)[val_indices])

        test_speeches = list(np.asarray(val_test_speeches)[test_indices])
        test_parties = list(np.asarray(val_test_parties)[test_indices])
        test_ids = list(np.asarray(val_test_ids)[test_indices])

        test_set = ParagraphDataset(val_speeches,val_parties,val_ids,0,len(val_speeches),generate[1],embedding_model,file_name=val_embeddings)
        test_set = ParagraphDataset(test_speeches,test_parties,test_ids,0,len(test_speeches),generate[2],embedding_model,file_name=test_embeddings)

        
        return train_set, val_set, test_set
    else:
        val_test_set = ParagraphDataset(speeches,parties,ids,train_stop,test_stop,generate[2],embedding_model,file_name=test_embeddings)
        return train_set, val_test_set"""
    

def prepare_data_loaders(train,val,test,batch_size,collate_fns=[collate_fn,collate_fn,collate_fn],train_size=1,device="cpu"):
    # Train on only a proportion of the set
    if train_size < 1:
        train_set, _ = torch.utils.data.random_split(train,[train_size,1-train_size])
    else:
        train_set = train
    val_set = val
    test_set = test

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fns[0],generator=torch.Generator(device=device))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fns[1],generator=torch.Generator(device=device))
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=True, collate_fn=collate_fns[2],generator=torch.Generator(device=device))
    return train_loader, val_loader, test_loader

def assert_correct_labeling(dataset, random_samples):
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
            print(speech)