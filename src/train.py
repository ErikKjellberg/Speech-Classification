from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys

class Trainer:
    def __init__(self,opt=torch.optim.Adam,loss_fn=nn.CrossEntropyLoss()):
        self.optimizer = opt
        self.loss_function = loss_fn

    def train_model(self, model, epochs, train_loader, val_loader,  one_cycle=True,max_lr=1e-3, save=True, plot=True):
        optimizer = self.optimizer(model.parameters())
        if one_cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
        
        train_length = len(train_loader)
        val_length = len(val_loader)

        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Loop through the training set each epoch
        for epoch in range(epochs):
            print("Epoch "+str(epoch+1)+"/"+str(epochs),file=sys.stderr)

            # Train the model
            model.train()
            with tqdm(train_loader) as pbar:
                for i, (X_train, lengths, y_train) in enumerate(pbar):
                    out = model(X_train.to("cuda"), lengths)
                    if type(out) is tuple:
                        y_pred = out[0]
                    else:
                        y_pred = out
                    loss = self.loss_function(y_pred, y_train.to("cuda"))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    learning_rates.append(scheduler.get_last_lr()[0])
                    train_losses.append(loss.item())
                    pbar.set_postfix({"loss":"%.4f"%loss.item()},refresh=False)

            # Evaluate after each epoch
            model.eval()
            for i, (X_val, lengths, y_val) in enumerate(tqdm(val_loader)):
                out = model(X_val.to("cuda"), lengths)
                if type(out) is tuple:
                    y_pred = out[0]
                else:
                    y_pred = out
                loss =self.loss_function(y_pred, y_val.to("cuda"))
                val_losses.append(loss.item())
            
            print("Train loss: "+str(sum(train_losses[-train_length:])/(train_length)),file=sys.stderr)
            print("Validation loss: "+str(sum(val_losses[-(val_length):])/((val_length))),file=sys.stderr)
            print("Learning rate: "+str(learning_rates[-1]),file=sys.stderr)
            torch.cuda.empty_cache()
            print("Total memory occupied: "+str(torch.cuda.mem_get_info()[1]),file=sys.stderr)

            # Save the model parameters after each epoch
            if hasattr(model, "name"):
                torch.save(model.state_dict(),f="trained_models/"+model.name+"_"+str(epoch+1)+"_epochs.pth")
        
        # Save and plot losses
        if save:
            torch.save(train_losses,f="losses/train_losses_"+str(model.__class__.__name__)+"_"+str(epochs)+"_epochs.pt")
            torch.save(val_losses,f="losses/val_losses_"+str(model.__class__.__name__)+"_"+str(epochs)+"_epochs.pt")
        if plot:
            self.plot_losses(train_losses,train_length,val_losses,val_length)
        return train_losses, val_losses

    def plot_losses(self,train_loss,train_length,val_loss,val_length):
        t1 = np.arange(len(train_loss))
        t2 = train_length/val_length*np.arange(len(val_loss))
        plt.scatter(t1,train_loss,s=0.2)
        plt.scatter(t2,val_loss,s=0.2)
        plt.show()
