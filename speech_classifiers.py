import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence


class EmbeddingsAverage(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbeddingsAverage, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    # x is a packed sequence
    def forward(self, x):
        x_list = unpack_sequence(x)
        avg_vector = torch.zeros((len(x_list),x_list[0].shape[1]))
        for i,x in enumerate(x_list):
            avg_vector[i,:] = torch.mean(x, dim=0)
        output = self.fc(avg_vector)
        return output

"""
    def forward(self, x, lengths):
        # Input is a batch of paragraph embeddings.
        # They have different lengths and therefore the average needs to be taken over different amounts of sentences for different examples.
        avg_vector = torch.zeros((x.shape[0],x.shape[2]))
        for i in range(x.shape[0]):
            avg_vector[i,:] = torch.mean(x[i,:lengths[i],:], dim=0)
        #print("The average is "+str(avg_vector[0,:10]))
        output = self.fc(avg_vector)
        return output"""

#%% 
i = 3
print(i)

#%%


class OneLayerLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=64, output_size=8):
        super(OneLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self,x,lengths):
        # To normalize:
        #x = torch.div(x,torch.linalg.vector_norm(x,ord=2,dim=1).unsqueeze(1))
        f, (h,c) = self.lstm(x)
        batch_size = h.shape[1]
        h = torch.permute(h,(1,0,2))
        h = torch.reshape(h,(batch_size,self.hidden_size*2))
        h = self.dropout(h)
        output = self.fc(h)
        return output.squeeze()


class OneLayerLSTMDoubleFC(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=64, output_size=8):
        super(OneLayerLSTMDoubleFC, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(2*hidden_size, 256)
        self.fc_2 = nn.Linear(256, output_size)
        self.batch_size = batch_size

    def forward(self,x):
        # To normalize:
        #x = torch.div(x,torch.linalg.vector_norm(x,ord=2,dim=1).unsqueeze(1))
        f, (h,c) = self.lstm(x)
        batch_size = h.shape[1]
        h = torch.permute(h,(1,0,2))
        h = torch.reshape(h,(batch_size,self.hidden_size*2))
        h = self.dropout(h)
        output = self.fc_2(self.fc_1(h))
        return output.squeeze()
    

class OneLayerLSTMExtraDropout(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=64, output_size=8):
        super(OneLayerLSTMExtraDropout, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self,x):
        # To normalize:
        #x = torch.div(x,torch.linalg.vector_norm(x,ord=2,dim=1).unsqueeze(1)
        # 
        # This does not work yet
        x = self.dropout_1(x)
        f, (h,c) = self.lstm(x)
        batch_size = h.shape[1]
        h = torch.permute(h,(1,0,2))
        h = torch.reshape(h,(batch_size,self.hidden_size*2))
        h = self.dropout_2(h)
        output = self.fc(h)
        return output.squeeze()
    

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=64, output_size=8):
        super(TwoLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        # Maybe fix so this doesn't use all hidden layers, i.e. change to hidden_size * 2
        self.fc = nn.Linear(4*hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self,x):
        # To normalize:
        #x = torch.div(x,torch.linalg.vector_norm(x,ord=2,dim=1).unsqueeze(1))
        f, (h,c) = self.lstm(x)
        batch_size = h.shape[1]
        h = torch.permute(h,(1,0,2))
        h = torch.reshape(h,(batch_size,self.hidden_size*4))
        h = self.dropout(h)
        output = self.fc(h)
        return output.squeeze()
    

"""class SentenceLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=64, output_size=8):
        super(SentenceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.batch_size = batch_size
        
    # Idea 1: x is a list of packed sequences of paragraphs, each consisting of a variable amount of sentences
    def forward(self,x):
        sequence_outputs = torch.zeros(len(x),self.batch_size)
        for i in range(len(x)):
            # x[i] is a packed sequence, containing a variable amount of sentence lists
            f, (h,c) = self.lstm(x[i])
            sequence_outputs[i] = f

    # Idea 2: x is a packed sequence of tensors of dim (no_paragraphs,max_no_sentences,embedding_size)
    def forward(self,x):
        # LSTM on whole batch:
        f, (h,c) = self.lstm(x)
        # Take average over f:
        avg = torch.zeros(self.batch_size,max_seq_length,self.embedding_size)
        for i in range(self.batch_size):
            avg[i] = torch.mean(f,dim=1)"""

# Code from https://github.com/JoungheeKim/Pytorch-Hierarchical-Attention-Network/blob/master/model.py
class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_size, attention_size,
                 num_class, n_layers=1, dropout_p=0.05, device="cpu"):
        super(AttentionNetwork, self).__init__()
        #self.sentence_attention_model = SentenceAttention(input_size=hidden_size,
        #                                                  hidden_size=hidden_size,
        #                                                  attention_size=attention_size,
        #                                                  n_layers=n_layers,
        #                                                  dropout_p=dropout_p,
        #                                                  device=device
        #                                                  )
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

    #def forward(self, document, sentence_per_document):
    def forward(self, packed_sentence_vecs, sentence_per_document):
        #batch_size, max_sentence_length, max_word_length = document.size()
        #batch_size, max_sentence_length = document.shape
        # |document| = (batch_size, max_sentence_length, max_word_length)
        # |sentence_per_document| = (batch_size)
        # |word_per_sentence| = (batch_size, max_sentence_length)

        # Remove sentence-padding in document by using "pack_padded_sequence.data"
        #packed_sentences = pack(document,
        #                        lengths=sentence_per_document.tolist(),
        #                        batch_first=True,
        #                        enforce_sorted=False)
        # |packed_sentences.data| = (sum(sentence_length), max_word_length)

        # Remove sentence-padding in word_per_sentence "pack_padded_sequence.data"
        #packed_words_per_sentence = pack(word_per_sentence,
        #                                 lengths=sentence_per_document.tolist(),
        #                                 batch_first=True,
        #                                 enforce_sorted=False)
        # |packed_words_per_sentence.data| = (sum(sentence_length))

#Eriks kommentar: i mitt fall är sum(sentence_length) det totala antalet meningar. Det innebär att
# |sentence_vecs| = (total_amount)
# Jag borde bara kunna ta in en packed sequence av sentence_vecs direkt!
        # Get sentence vectors
        #sentence_vecs, word_weights = self.word_attention_model(packed_sentences.data,
        #                                                        packed_words_per_sentence.data)
        # |sentence_vecs| = (sum(sentence_length), hidden_size)
        # |word_weights| = (sum(sentence_length, max(word_per_sentence))

        # "packed_sentences" have same information to recover PackedSequence for sentence
        #packed_sentence_vecs = PackedSequence(data=sentence_vecs,
        #                                      batch_sizes=packed_sentences.batch_sizes,
        #                                      sorted_indices=packed_sentences.sorted_indices,
        #                                      unsorted_indices=packed_sentences.unsorted_indices)

        # Get document vectors
        doc_vecs, sentence_weights = self.sentence_attention_model(packed_sentence_vecs,
                                                                   sentence_per_document)
        # |doc_vecs| = (batch_size, hidden_size)
        # |sentence_weights| = (batch_size)

        #y = self.softmax(self.output(doc_vecs))

        #return y, sentence_weights, word_weights
        y = self.output(doc_vecs)
        return y

    def set_embedding(self, embedding, requires_grad = True):
        self.word_attention_model.emb.weight.data.copy_(embedding)
        return True


class HierarchialAttentionNetwork(nn.Module):

    def __init__(self, embedding_size, hidden_size, attention_size, bert,
                 num_class=8, n_layers=1, dropout_p=0.05, device="cpu"):

        super(HierarchialAttentionNetwork, self).__init__()

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

    def forward(self, document, sentence_per_document):
        attention_mask = document["attention_mask"]
        word_per_sentence = torch.sum(attention_mask,dim=1)
        # Attention mask is of shape (total sentences, max words per sentence)
        embedded_doc_padded = self.bert(return_dict=True, **document)["last_hidden_state"]
        # This should be of shape (total no sentences, max words per sentence, embedding size)
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
        
        #batch_size, max_sentence_length, max_word_length = document.size()
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

    def set_embedding(self, embedding, requires_grad = True):
        self.word_attention_model.emb.weight.data.copy_(embedding)
        return True


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