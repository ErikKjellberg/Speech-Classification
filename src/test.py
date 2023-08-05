from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# Evaluate a model on a test set using a dataloader
def evaluate_model(model,test_loader,labels=None):
    model.eval()
    y_tests = np.array([])
    y_preds = np.array([])
    IDs = np.array([])
    print("Evaluating model...")
    for i, (X_test, lengths, y_test, ID) in tqdm(enumerate(test_loader)):
        out = model(X_test.to("cuda"), lengths)
        if type(out) is tuple:
            y_pred = out[0].softmax(dim=1).argmax(dim=1)
        else:
            y_pred = out.softmax(dim=1).argmax(dim=1)
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

# Display classification report and confusion matrices
def show_metrics(y_tests,y_preds,label_names,plot=True,save=False):
    cr = classification_report(y_tests,y_preds,target_names=label_names)
    cm = confusion_matrix(y_tests, y_preds)
    cm_normalized = confusion_matrix(y_tests, y_preds, normalize="pred")
    print(cr)
    if plot:
        cm_disp = ConfusionMatrixDisplay(cm,display_labels=label_names)
        cm_disp_normalized = ConfusionMatrixDisplay(cm_normalized,display_labels=label_names)
        cm_disp.plot()
        plt.title("Confusion matrix")
        if save:
            plt.savefig("confusion_matrix.png")
        cm_disp_normalized.plot()
        plt.title("Normalized confusion matrix")
        if save:
            plt.savefig("confusion_matrix_normalized.png")
    else:
        print(cm)
        print(cm_normalized)
    return cr, cm, cm_normalized
    
