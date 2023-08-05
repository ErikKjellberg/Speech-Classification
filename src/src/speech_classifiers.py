# This script contains neural models for document classification.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

from tqdm import tqdm

class EmbeddingsAverage(nn.Module):
    def __init__(self, input_size, output_size=8, name=None):
        super(EmbeddingsAverage, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        if name is not None:
            self.name = name

    def forward(self, x, lengths):
        x_list = unpack_sequence(x)
        avg_vector = torch.zeros((len(x_list),x_list[0].shape[1]))
        for i,x in enumerate(x_list):
            avg_vector[i,:] = torch.mean(x, dim=0)
        output = self.fc(avg_vector.to("cuda"))
        return output


class OneLayerLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, name=None, dropout_p=0.5, output_size=8):
        super(OneLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.batch_size = batch_size
        if name is not None:
            self.name = name

    def forward(self,x,lengths):
        f, (h,c) = self.lstm(x)
        batch_size = h.shape[1]
        h = torch.permute(h,(1,0,2))
        h = torch.reshape(h,(batch_size,self.hidden_size*2))
        h = self.dropout(h)
        output = self.fc(h)
        return output.squeeze()
        

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, name=None, dropout_p=0.5, output_size=8):
        super(TwoLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(4*hidden_size, output_size)
        self.batch_size = batch_size
        if name is not None:
            self.name = name

    def forward(self,x,lengths):
        f, (h,c) = self.lstm(x)
        batch_size = h.shape[1]
        h = torch.permute(h,(1,0,2))
        h = torch.reshape(h,(batch_size,self.hidden_size*4))
        h = self.dropout(h)
        output = self.fc(h)
        return output.squeeze()


# This class and the HierarchicalAttentionNetwork contains modified code from
# https://github.com/JoungheeKim/Pytorch-Hierarchical-Attention-Network/blob/master/model.py
class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_size, attention_size, name,
                 num_class=8, n_layers=1, dropout_p=0.05, device="cpu"):
        super(AttentionNetwork, self).__init__()
        self.sentence_attention_model = SentenceAttention(input_size=embedding_size,
                                                          hidden_size=hidden_size,
                                                          attention_size=attention_size,
                                                          n_layers=n_layers,
                                                          dropout_p=dropout_p,
                                                          device=device
                                                          )

        self.device = device
        self.output = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.name = name

    def forward(self, packed_sentence_vecs, sentence_per_document):

        # Get document vectors
        doc_vecs, sentence_weights = self.sentence_attention_model(packed_sentence_vecs,
                                                                   sentence_per_document)
        # |doc_vecs| = (batch_size, hidden_size)
        # |sentence_weights| = (batch_size)

        y = self.output(doc_vecs)
        return y


class HierarchialAttentionNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_size, attention_size, bert, name,
                 num_class=8, n_layers=1, dropout_p=0.05, device="cuda", return_more=True):
        super(HierarchialAttentionNetwork, self).__init__()
        
        self.name = name
        self.word_attention_model = WordAttention(embedding_size=embedding_size,
                                                  hidden_size=hidden_size,
                                                  attention_size=attention_size,
                                                  n_layers=n_layers,
                                                  dropout_p=dropout_p,
                                                  device=device
                                                  )

        self.sentence_attention_model = SentenceAttention(input_size=hidden_size,
                                                          hidden_size=hidden_size,
                                                          attention_size=attention_size,
                                                          n_layers=n_layers,
                                                          dropout_p=dropout_p,
                                                          device=device
                                                          )

        self.device = device
        self.output = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.bert = bert
        self.return_more = return_more

    def forward(self, document, sentence_per_document):
        attention_mask = document["attention_mask"]
        word_per_sentence = torch.sum(attention_mask,dim=1)
        # Attention mask is of shape (total sentences, max words per sentence)
        with torch.no_grad():
            embedded_doc_padded = self.bert(return_dict=True, **document)["last_hidden_state"]
            
        # This is of shape (total no sentences, max words per sentence, embedding size)
        # Turn it into a packed sequence
        embedded_doc_list = []
        word_per_sentence_list = []
        index = 0
        for i in range(len(sentence_per_document)):
            embedded_doc_list.append(embedded_doc_padded[index:index+sentence_per_document[i]])
            word_per_sentence_list.append(word_per_sentence[index:index+sentence_per_document[i]])
            index += sentence_per_document[i]
        packed_sentences = pack_sequence(embedded_doc_list)
        packed_words_per_sentence = pack_sequence(word_per_sentence_list)
        # Now we have a packed sequence of all documents, each being a padded tensor of dim (sentences, max_words, embedding_size)
        
        # |document| = (batch_size, max_sentence_length, max_word_length)
        # |sentence_per_document| = (batch_size)
        # |word_per_sentence| = (batch_size, max_sentence_length)

        # Remove sentence-padding in word_per_sentence "pack_padded_sequence.data"
        #packed_words_per_sentence = pack_sequence(word_per_sentence,
        #                                 lengths=sentence_per_document,
        #                                 batch_first=True,
        #                                 enforce_sorted=False)
        #packed_words_per_sentence = pack_sequence()
        # |packed_words_per_sentence.data| = (sum(sentence_length))

        # Get sentence vectors
        sentence_vecs, word_weights = self.word_attention_model(packed_sentences.data,
                                                                packed_words_per_sentence.data)
        # |sentence_vecs| = (sum(sentence_length), hidden_size)
        # |word_weights| = (sum(sentence_length, max(word_per_sentence))

        # "packed_sentences" have same information to recover PackedSequence for sentence
        packed_sentence_vecs = PackedSequence(data=sentence_vecs,
                                              batch_sizes=packed_sentences.batch_sizes,
                                              sorted_indices=packed_sentences.sorted_indices,
                                              unsorted_indices=packed_sentences.unsorted_indices)

        # Get document vectors
        doc_vecs, sentence_weights = self.sentence_attention_model(packed_sentence_vecs,
                                                                   sentence_per_document)
        # |doc_vecs| = (batch_size, hidden_size)
        # |sentence_weights| = (batch_size)

        y = self.output(doc_vecs)
        if self.return_more:
            return y, doc_vecs, sentence_weights, word_weights
        else:
            return y
    
    
class WordAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super(WordAttention, self).__init__()

        self.device=device
        #self.emb = nn.Embedding(dictionary_size, embedding_size).to(device)
        self.rnn = nn.LSTM(input_size=embedding_size,
                          hidden_size=int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=True,
                          batch_first=True
                          )

        self.attn = Attention(hidden_size=hidden_size,
                              attention_size=attention_size)


    def forward(self, sentence, word_per_sentence):
        # |sentence| = (sentence_length, max_word_length)
        # |word_per_sentence| = (sentence_length)

        #sentence = self.emb(sentence)
        # |sentence| = (sentence_length, max_word_length, embedding_size)

        # Pack sentence before insert rnn model.
        packed_sentences = pack(sentence,
                                lengths=word_per_sentence.tolist(),
                                batch_first=True,
                                enforce_sorted=False)

        # Apply RNN and get hiddens layers of each words
        last_hiddens, _ = self.rnn(packed_sentences)

        # Unpack ouput of rnn model
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = (sentence_length, max(word_per_sentence), hidden_size)

        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = self.generate_mask(word_per_sentence)
        # |mask| = (sentence_length, max(word_per_sentence))

        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (sentence_length, hidden_size)
        # |context_weights| = (sentence_length, max(word_per_sentence))

        return context_vectors, context_weights

    def generate_mask(self, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat(
                    [torch.zeros((1, l), dtype=torch.uint8), torch.ones((1, (max_length - l)), dtype=torch.uint8)],
                    dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [torch.zeros((1, l), dtype=torch.uint8)]

        mask = torch.cat(mask, dim=0).byte()

        return mask.to(self.device)    


class SentenceAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super(SentenceAttention, self).__init__()

        self.device=device
        self.rnn = nn.LSTM(input_size=input_size,
                          hidden_size=int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=True,
                          batch_first=True
                          )

        self.attn = Attention(hidden_size=hidden_size,
                              attention_size=attention_size)


    def forward(self, packed_sentences, sentence_per_document):
        # |packed_sentences| = PackedSequence()

        # Apply RNN and get hiddens layers of each sentences
        last_hiddens, _ = self.rnn(packed_sentences)

        # Unpack ouput of rnn model
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = (sentence_length, max(word_per_sentence), hidden_size)

        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = self.generate_mask(sentence_per_document)
        # |mask| = (sentence_length, max(word_per_sentence))

        # Get attention weights and context vectors
        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (sentence_length, hidden_size)
        # |context_weights| = (sentence_length, max(word_per_sentence))

        return context_vectors, context_weights

    def generate_mask(self, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat(
                    [torch.zeros((1, l), dtype=torch.uint8), torch.ones((1, (max_length - l)), dtype=torch.uint8)],
                    dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [torch.zeros((1, l), dtype=torch.uint8)]

        mask = torch.cat(mask, dim=0).byte()

        return mask.to(self.device)
    

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, attention_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        ## Context vector
        self.context_weight = nn.Parameter(torch.Tensor(attention_size, 1))
        self.context_weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, h_src, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |mask| = (batch_size, length)
        batch_size, length, hidden_size = h_src.size()

        # Resize hidden_vectors to generate weight
        #weights = h_src.view(-1, hidden_size)
        weights = h_src.reshape(-1, hidden_size)
        weights = self.linear(weights)
        weights = self.tanh(weights)

        weights = torch.mm(weights, self.context_weight).view(batch_size, -1)
        # |weights| = (batch_size, length)

        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weights.masked_fill_(mask, -float('inf'))

        # Modified every values to (0~1) by using softmax function
        weights = self.softmax(weights)
        # |weights| = (batch_size, length)

        context_vectors = torch.bmm(weights.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        context_vectors = context_vectors.squeeze(1)
        # |context_vector| = (batch_size, hidden_size)

        return context_vectors, weights
    
# This is a model which works exactly like HierarchicalAttentionNetwork, except it is compatible with
# the BERTDict in datahandling.py. This makes it possible to store all embeddings beforehand.
class HANWithBERTDict(nn.Module):

    def __init__(self, embedding_size, hidden_size, attention_size, bert_dict, name,
                 num_class=8, n_layers=1, dropout_p=0.05, device="cuda"):

        super(HierarchialAttentionNetwork, self).__init__()
        
        self.name = name
        self.word_attention_model = WordAttention(embedding_size=embedding_size,
                                                  hidden_size=hidden_size,
                                                  attention_size=attention_size,
                                                  n_layers=n_layers,
                                                  dropout_p=dropout_p,
                                                  device=device
                                                  )

        self.sentence_attention_model = SentenceAttention(input_size=hidden_size,
                                                          hidden_size=hidden_size,
                                                          attention_size=attention_size,
                                                          n_layers=n_layers,
                                                          dropout_p=dropout_p,
                                                          device=device
                                                          )

        self.device = device
        self.output = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.bert_dict = bert_dict

    def forward(self, document, sentence_per_document, indices):
        attention_mask = document["attention_mask"]
        word_per_sentence = torch.sum(attention_mask,dim=1)
        # Attention mask is of shape (total sentences, max words per sentence)
        embedded_docs = []
        for index in indices:
            embedded_docs.append(self.bert_dict.get_embedding(index).to("cuda"))
        max_seq_length = max(embedded_docs,key=lambda d:d.shape[1]).shape[1]
        for i, d in enumerate(embedded_docs):
            embedded_docs[i] = nn.functional.pad(d,(0,0,0,max_seq_length-d.shape[1],0,0))
        # Now we have a list of embedded docs which we can turn into a packed sequence.

        word_per_sentence_list = []
        index = 0
        for i in range(len(sentence_per_document)):
            word_per_sentence_list.append(word_per_sentence[index:index+sentence_per_document[i]])
            index += sentence_per_document[i]
        packed_sentences = pack_sequence(embedded_docs)
        packed_words_per_sentence = pack_sequence(word_per_sentence_list)
        # Now we have a packed sequence of all documents, each being a padded tensor of dim (sentences, max_words, embedding_size)
        
        # |document| = (batch_size, max_sentence_length, max_word_length)
        # |sentence_per_document| = (batch_size)
        # |word_per_sentence| = (batch_size, max_sentence_length)

        # Remove sentence-padding in word_per_sentence "pack_padded_sequence.data"
        #packed_words_per_sentence = pack_sequence(word_per_sentence,
        #                                 lengths=sentence_per_document,
        #                                 batch_first=True,
        #                                 enforce_sorted=False)
        #packed_words_per_sentence = pack_sequence()
        # |packed_words_per_sentence.data| = (sum(sentence_length))

        # Get sentence vectors
        sentence_vecs, word_weights = self.word_attention_model(packed_sentences.data,
                                                                packed_words_per_sentence.data)
        # |sentence_vecs| = (sum(sentence_length), hidden_size)
        # |word_weights| = (sum(sentence_length, max(word_per_sentence))

        # "packed_sentences" have same information to recover PackedSequence for sentence
        packed_sentence_vecs = PackedSequence(data=sentence_vecs,
                                              batch_sizes=packed_sentences.batch_sizes,
                                              sorted_indices=packed_sentences.sorted_indices,
                                              unsorted_indices=packed_sentences.unsorted_indices)

        # Get document vectors
        doc_vecs, sentence_weights = self.sentence_attention_model(packed_sentence_vecs,
                                                                   sentence_per_document)
        # |doc_vecs| = (batch_size, hidden_size)
        # |sentence_weights| = (batch_size)

        #y = self.softmax(self.output(doc_vecs))
        y = self.output(doc_vecs)

        return y, sentence_weights, word_weights
