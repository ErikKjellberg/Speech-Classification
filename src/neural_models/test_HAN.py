from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm
import json
import sys

def evaluate_model(model,test_loader,labels=None):
    model.eval()
    y_tests = np.array([])
    y_preds = np.array([])
    IDs = np.array([])
    print("Evaluating model...")
    for i, (X_test, lengths, y_test, ID) in tqdm(enumerate(test_loader)):
        y_pred = model(X_test.to("cuda"), lengths)[0].softmax(dim=1).argmax(dim=1)
        y_tests = np.concatenate((y_tests,y_test.numpy()))
        y_preds = np.concatenate((y_preds,y_pred.cpu().numpy()))
        IDs = np.concatenate((IDs, ID))
    show_metrics(y_tests,y_preds,labels)
    return y_tests, y_preds, IDs

# Input: the correct classes, the predicted classes and lists of classes 
def misclassified(y_tests,y_preds,correct,wrong):
    idx = []
    for i, (y_test, y_pred) in enumerate(zip(y_tests,y_preds)):
        if y_test in correct and y_pred in wrong:
            idx.append(i)
    return idx

def get_speech_and_party_from_ID(ID):
    ID_to_dir = {"H6":"speech_data/18_19/","H7":"speech_data/19_20/","H8":"speech_data/20_21/","H9":"speech_data/21_22/"}
    ID_start = ID[:2]
    dir = ID_to_dir[ID_start]
    with open(dir+ID+".json","r") as f:
        data = json.load(f)
        speech = data["anforande"]["anforandetext"]
        party = data["anforande"]["parti"]
        return speech, party

def show_metrics(y_tests,y_preds,label_names):
    cr = classification_report(y_tests,y_preds,target_names=label_names)
    cm = confusion_matrix(y_tests, y_preds)
    print(cr)
    print(cm)
    return cm
    
