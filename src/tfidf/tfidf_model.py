import spacy
from spacy.lang.sv import Swedish
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os 
import json
from bs4 import BeautifulSoup
import numpy as np
import string
import torch
import re
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-p","--svdpercentage",type=float)
parser.add_argument("-s","--startyear",type=int)
parser.add_argument("-e","--endyear",type=int)

args = parser.parse_args()

stop_words = spacy.lang.sv.stop_words.STOP_WORDS
punctuations = string.punctuation

nlp = spacy.load('sv_core_news_lg')

def swedish_tokenizer(text):
    doc = nlp(text)
    return [token.lower_ for token in doc if token.text not in stop_words and token.text not in punctuations]

print("Initializing models...",file=sys.stderr)

tfidf_vectorizer = TfidfVectorizer(tokenizer=swedish_tokenizer,min_df=10)

svd = TruncatedSVD(n_components=512)

log_reg = LogisticRegression(multi_class="multinomial", max_iter=1000)

firsttime = True
loadjson = True

possible_parties = ["S","M","SD","V","MP","C","KD","L","KDS","FP"]
start_year = args.startyear
end_year = args.endyear

if firsttime:
    
    if loadjson:
        print("Constructing speech list from json files...",file=sys.stderr)

        speeches = []
        parties = []

        
        # The directory where data is located
        data_dirs = ["../speech_data/"+str(i)[2:]+"_"+str(i+1)[2:]+"/" for i in range(start_year,end_year)]
        year_indices = [0] * len(data_dirs)
        speeches_per_year = float("inf")
        data = []
        i = 0
        for year, data_dir in enumerate(data_dirs):
            year_indices[year] = len(speeches)
            print("Year "+str(year+start_year),file=sys.stderr)
            speech_count = 0
            for file in os.listdir(data_dir):
                if file.endswith(".json"):
                    if speech_count >= speeches_per_year:
                        break
                    with open(data_dir+file,"r",encoding="utf-8-sig") as f:
                        data = json.load(f)
                        speech = data["anforande"]["anforandetext"]
                        party = data["anforande"]["parti"]
                        # If it is a party who has spoken, catch the speech
                        if party is not None and party.upper() in possible_parties:
                            if 2014 <= year+start_year <= 2022:
                                soup = BeautifulSoup(speech, 'html.parser')
                                speech_text = " ".join([p.get_text() for p in soup.find_all("p")])
                                speech_text = re.sub("STYLEREF Kantrubrik \\\* MERGEFORMAT","",speech_text)
                            elif 2003 <= year+start_year <= 2013:
                                speech_text = re.sub("\\r\\n"," ",speech)
                            elif 1993 <= year+start_year <= 2002:
                                speech_text = re.sub("\-\\n"," ",speech)
                                speech_text = re.sub("\\n"," ",speech_text)
                            else:
                                speech_text = ""
                            if speech_text != "":
                                parties.append(party.upper())
                                speeches.append(speech_text)
                                speech_count += 1
        assert len(parties) == len(speeches)
        torch.save([speeches, parties, year_indices],"speeches_parties_year_indices.pt")

    else:
        speeches, parties, year_indices = torch.load("data_1993_2023.pt")


    party_to_number = {}
    number_to_party = {}
    for i, p in enumerate(possible_parties[:8]):
        party_to_number[p] = i
        number_to_party[i] = p
    party_to_number["FP"]=party_to_number["L"]
    party_to_number["KDS"]=party_to_number["KD"]
    parties = [party_to_number.get(p) for p in parties]

    train_stop = year_indices[-1]
    test_stop = len(speeches)

    X_train = speeches[:train_stop]
    y_train = parties[:train_stop]

    X_test = speeches[train_stop:test_stop]
    y_test = parties[train_stop:test_stop]

    start = time.time()
    print("Fitting tfidf vectorizer on training set...",file=sys.stderr)

    X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)

    stop = time.time()
    print("Fitting tfidf vectorizer took "+str(stop-start)+" seconds.",file=sys.stderr)

    print("Saving model and tfidf matrix...", file=sys.stderr)
    torch.save([X_train_vectorized, y_train, X_test, y_test],"train_test.pt")
    torch.save(tfidf_vectorizer, "tfidf_vectorizer.pt")

else:
    X_train_vectorized, y_train, X_test, y_test = torch.load("train_test.pt")
    tfidf_vectorizer = torch.load("tfidf_vectorizer.pt")

if args.svdpercentage is not None:
    svd_percentage = args.svdpercentage
else:
    svd_percentage = 0.1
svd_size = int(X_train_vectorized.shape[0]*svd_percentage)

print("Total size of tfidf matrix: "+str(X_train_vectorized.shape[0]),file=sys.stderr)
print("Size to use SVD on: "+str(svd_size),file=sys.stderr)

start = time.time()
print("Applying SVD to train set...",file=sys.stderr)
X_dim_reduced = svd.fit_transform(X_train_vectorized[:svd_size])
stop = time.time()
print("SVD took "+str(stop-start)+" seconds.",file=sys.stderr)

print("Saving SVD model...",file=sys.stderr)
torch.save(svd, "svd.pt")

start = time.time()
print("Transforming test set", file=sys.stderr)
X_test_vectorized = tfidf_vectorizer.transform(X_test)
stop1 = time.time()
print("Tfidf transform took "+str(stop1-start)+" seconds.",file=sys.stderr)
X_test_dim_reduced = svd.transform(X_test_vectorized)
stop2 = time.time()
print("Applying SVD to test set took "+str(stop2-stop1)+" seconds.",file=sys.stderr)
print("Total transforming of test set took "+str(stop2-start)+" seconds.",file=sys.stderr)

print("Fitting logistic regression model.",file=sys.stderr)

log_reg.fit(X_dim_reduced,y_train[:svd_size])

print("Evaluating on test set...",file=sys.stderr)

y_pred = log_reg.predict(X_test_dim_reduced)
print(classification_report(y_test,y_pred,target_names=possible_parties[:8]),file=sys.stdout)
