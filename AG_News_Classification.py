import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from preprocessing import Preprocessor

class DatasetMapper(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class AG_News_Classifier:
    '''Classification Pipeline'''
    
    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 learning_rate,
                 num_epochs,
                 batch_size):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
        self.model = model.to(self.device)
        self.criterion = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
       
    def loadData(self, vocab_size, embedding_size):
        
        preprocessor = Preprocessor()
        preprocessor.load_data(data_path='./data/', val_size = 0.5)
        preprocessor.tokenize_data(max_words = vocab_size, max_length = embedding_size)
        
        train_dataset = DatasetMapper(preprocessor.X_train, preprocessor.y_train)
        val_dataset = DatasetMapper(preprocessor.X_val, preprocessor.y_val)
        test_dataset = DatasetMapper(preprocessor.X_test, preprocessor.y_test)
                
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=4,
                                       pin_memory=True,
                                       shuffle=True)
        
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.batch_size*2,
                                     num_workers=4,
                                     pin_memory=True,
                                     shuffle=False)
        
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.batch_size*2,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=False)
        
        print(f'\nTrain set size: {len(self.train_loader.dataset)}')
        print(f'Val set size: {len(self.val_loader.dataset)}')
        print(f'Test set size: {len(self.test_loader.dataset)}')
        
        self.data_loaders = {'train': self.train_loader,
                             'val': self.val_loader,
                             'test': self.test_loader}
        
        self.classes = ['World', 'Sports', 'Business', 'Sci & Tech']       
    
    def train(self, grad_clip=0):
        
        self.history = {}
        self.history['Training Loss'] = []
        self.history['Training Accuracy'] = []
        self.history['Validation Loss'] = []
        self.history['Validation Accuracy'] = []
        
        print(f'\nLoaded data and pre-processed\n')
        
        print(f'batch size: {self.batch_size}, Num batches(# steps per epoch): {len(self.train_loader)}\n')
        
        '''Learing Rate Scheduler'''
        # scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
        #                                     max_lr=self.learning_rate,
        #                                     epochs=self.num_epochs, 
        #                                     steps_per_epoch=len(self.train_loader))
        
        # scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
        #                                 step_size=1,
        #                                 gamma=0.1)
        
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                   mode = 'max',
                                                   factor=0.1,
                                                   patience=1,
                                                   verbose=True)
        
        print(f'Training...\n')
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):

            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-'*10)

            self.model.train()
            train_losses = []
            lrs = []

            running_loss = 0.0
            running_corrects = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)
                _, preds = torch.max(outputs, dim=1)
                loss = self.criterion(outputs, y_batch)

                train_losses.append(loss)

                if grad_clip:
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for param_group in self.optimizer.param_groups:
                    lrs.append(param_group['lr'])

                running_loss += loss.item() * x_batch.size(0)
                running_corrects += torch.sum(preds == y_batch.data)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc_train = running_corrects.double() / len(self.train_loader.dataset)
            self.history['Training Loss'].append(epoch_loss)
            self.history['Training Accuracy'].append(epoch_acc_train.item())
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.4f}')

            '''Validation'''
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0

            for x_batch, y_batch in self.val_loader:

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with torch.no_grad():
                    outputs = self.model(x_batch)
                    _, preds = torch.max(outputs, dim=1)
                    loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * x_batch.size(0)
                running_corrects += torch.sum(preds == y_batch.data)

            epoch_loss = running_loss / len(self.val_loader.dataset)
            epoch_acc_val = running_corrects.double() / len(self.val_loader.dataset)

            self.history['Validation Loss'].append(epoch_loss)
            self.history['Validation Accuracy'].append(epoch_acc_val.item())
            
            print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc_val:.4f}')

           
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                best_epoch = epoch
                
                # saving weights
                best_model_wts = copy.deepcopy(self.model.state_dict())

            self.scheduler.step(epoch_acc_val)
            print()

        time_elapsed = time.time() - since

        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f} at epoch:{best_epoch+1}\n')

        self.model.load_state_dict(best_model_wts)
        
    def test(self):
        
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(4)]
            n_class_samples = [0 for i in range(4)]

            for b, (x_batch, y_batch) in enumerate(self.test_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(x_batch)

                _, predicted = torch.max(outputs, 1)
                n_samples += y_batch.size(0)
                n_correct += (predicted == y_batch).sum().item()

                for i in range(len(y_batch)):
                    label = y_batch[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            for i in range(4):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {self.classes[i]}: {acc} %')
            print()        