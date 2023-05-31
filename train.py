from tqdm import tqdm
import torch
import torch.nn as nn
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import sys

class Trainer:
    def __init__(self,opt=torch.optim.Adam,loss_fn=nn.CrossEntropyLoss()) -> None:
        self.optimizer = opt
        self.loss_function = loss_fn
        self.train_losses = []
        self.val_losses = []
        self.train_length = 0
        self.val_length = 0

    def train_model(self, model, epochs, train_loader, val_loader,  one_cycle=True,max_lr=1e-3, save=True, plot=True):
        optimizer = self.optimizer(model.parameters())
        if one_cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
        
        train_length = len(train_loader)
        val_length = len(val_loader)

        train_losses = []
        val_losses = []
        learning_rates = []
        

        for epoch in tqdm(range(epochs)):
            # Train
            model.train()
            for i, (X_train, lengths, y_train) in enumerate(train_loader):
                y_pred = model(X_train, lengths)[0]
                loss = self.loss_function(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                learning_rates.append(scheduler.get_last_lr()[0])
                train_losses.append(loss.item())

            # Evaluate
            model.eval()
            for i, (X_val, lengths, y_val) in enumerate(val_loader):
                y_pred = model(X_val, lengths)[0]
                # Maybe I should evaluate accuracy as well
                loss =self.loss_function(y_pred, y_val)
                val_losses.append(loss.item())
            clear_output()
            print("Train loss: "+str(sum(train_losses[-train_length:])/(train_length)),file=sys.stderr)
            print("Validation loss: "+str(sum(val_losses[-(val_length):])/((val_length))),file=sys.stderr)
            print("Learning rate: "+str(learning_rates[-1]),file=sys.stderr)
        
        if save:
            # Save and plot losses
            torch.save(train_losses,f="../losses/train_losses_"+str(model.__class__.__name__)+"_"+str(epochs)+"_epochs.pt")
            torch.save(val_losses,f="../losses/val_losses_"+str(model.__class__.__name__)+"_"+str(epochs)+"_epochs.pt")
        if plot:
            self.plot_losses(train_losses,train_length,val_losses,val_length)
        return train_losses, val_losses

    def plot_losses(self,train_loss,train_length,val_loss,val_length):
        t1 = np.arange(len(train_loss))
        t2 = train_length/val_length*np.arange(len(val_loss))
        plt.scatter(t1,train_loss,s=0.2)
        plt.scatter(t2,val_loss,s=0.2)
        plt.show()
