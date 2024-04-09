from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cnn import CNN


batchs = [32,64,128]
lrs = [0.001, 0.0001, 0.00001]
pic_sizes = [(50, data_50), (128, data_128)]
epochs = [25,50,75,100]


k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)


hyper_tuning_res = []
for epoch_num in epochs:
    for batch_size in batchs:
        for lr in lrs:
            for pic_size in pic_sizes:
                # Load data in X_train and X_test etc
                # Load both before hand and call them different names (smarter)

                results = {
                    'batch_size' : batch_size,
                    'lr' : lr,
                    'pic_size' : pic_size,
                    'epoch_num' : epoch_num,
                    'fold_accs' : [],
                    'mean_fold_acc' : None 
                }

                for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
                    print(f'Fold:{fold}')
                    X_train_fold, y_train_fold = X_train[train_ids], y_train[train_ids]
                    X_val_fold, y_val_fold = X_train[val_ids], y_train[val_ids]

                    model = CNN(16, pic_size)
                    model.train_model(X_train_fold, y_train_fold, epochs=epoch_num, lr=lr, batch_size=batch_size)

                    print(f'Training of fold complete')
                    # You can save the model here to not recreate it later (even though youll need to)
                    print(f'Start Evaluation')

                    y_pred = model.predict(X_val_fold, batch_size=batch_size) 
                    fold_acc = ((torch.argmax(y_pred,1)==y_val_fold).sum().item())/len(y_val_fold)

                    results['fold_accs'].append(fold_acc)
                    print(f'The fold accuracy is:{fold_acc}')
                results['mean_fold_acc'] = sum(results['fold_accs'])/len(results['fold_accs'])
                hyper_tuning_res.append(results)
                #Dump hyper_tuning_res info into a file NOW