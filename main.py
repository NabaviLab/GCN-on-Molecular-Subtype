#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:00:37 2020

@author: bingjun
@author: tianyu
"""
   
import sys, os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import pdb #pdb.set_trace()
import collections
import argparse
import time
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import sklearn.metrics
import pandas as pd
import sys
sys.path.insert(0, 'lib/')

import warnings
warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from coarsening import coarsen, laplacian
from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L
from layermodel import *
import utilsdata
from utilsdata import *
import warnings
warnings.filterwarnings("ignore")

## Set up the arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default='personal', help="personal or hpc")
parser.add_argument('--lr', type=float, default = 0.01, help='learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--num_omic', type=int, default = 1, help='# of omics')
parser.add_argument('--epochs', type=int, default = 30, help='# of epoch')
parser.add_argument('--batchsize', type=int, default = 64, help='# of genes')
parser.add_argument('--database', type=str, default='biogrid', choices=['biogrid', 'string', 'coexpression'],help="netWork")
parser.add_argument('--singleton', type=bool, default=True, help="include Singleton")
parser.add_argument('--savemodel', type=int, default = 0, help='if save the model')
parser.add_argument('--loaddata', type=bool, default=True, help="if load the org data")

args = parser.parse_args()

# Start the timer
t_start = time.process_time()


# Load data
generateTrainTest = 1
    
print('load data...')

expression_data_path = 'data/common_expression_data.tsv'
cnv_data_path = 'data/common_cnv_data.tsv'
expression_variance_file = 'data/expression_variance.tsv'
shuffle_index_path = 'data/common_shuffle_index.tsv'
if args.database == 'biogrid':
    adjacency_matrix_file = 'data/adj_matrix_biogrid.npz'
    non_null_index_path = 'data/biogrid_non_null.csv'
elif args.database == 'string':
    adjacency_matrix_file = 'data/adj_matrix_string_filtered.npz'
    non_null_index_path = 'data/spring_non_null.csv'
elif args.database == 'coexpression':
    adjacency_matrix_file = 'data/adj_matrix_coexpression_filtered.npz'
    non_null_index_path = 'data/coexpression_non_null.csv'

if args.loaddata:
    if args.num_omic == 1:
        expr_all_data = load_singleomic_data(expression_data_path)

        adj, train_data_all, labels, shuffle_index = utilsdata.downSampling_singleomics_data(expression_variance_path=expression_variance_file,
                                                                            expression_data=expr_all_data,
                                                                            non_null_index_path=non_null_index_path,
                                                                            shuffle_index_path=shuffle_index_path,
                                                                            adjacency_matrix_path=adjacency_matrix_file,
                                                                            number_gene=args.num_gene,
                                                                            singleton=args.singleton)
    elif args.num_omic == 2:
        expr_all_data, cnv_all_data = load_multiomics_data(expression_data_path, cnv_data_path)

        adj, train_data_all, labels, shuffle_index = utilsdata.downSampling_multiomics_data(expression_variance_path=expression_variance_file,
                                                                      expression_data=expr_all_data,
                                                                      cnv_data=cnv_all_data,
                                                                      non_null_index_path=non_null_index_path,
                                                                      shuffle_index_path=shuffle_index_path,
                                                                      adjacency_matrix_path=adjacency_matrix_file,
                                                                      number_gene=args.num_gene,
                                                                      singleton=False)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)  
if not args.singleton:      
    adj, train_data_all = removeZeroAdj(adj, train_data_all)

if args.singleton:
    adj.setdiag(0)
    adj = adj + sp.eye(adj.shape[0])
else:
    adj, train_data_all = removeZeroAdj(adj, train_data_all) 


print('load done.')
adj_for_loss = adj.todense()
adj = adj/np.max(adj)
adj = adj.astype('float32')

print('******************************',adj.shape, train_data_all.shape)


if generateTrainTest:
    shuffle_index = shuffle_index.astype(np.int32).reshape(-1)

    train_size, val_size = int(len(shuffle_index)* 0.8), int(len(shuffle_index)* 0.9)
    train_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[0:train_size]]
    val_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[train_size:val_size]]
    test_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[val_size:]]
    train_labels = labels[np.array(shuffle_index[0:train_size])]
    val_labels = labels[shuffle_index[train_size:val_size]]
    test_labels = labels[shuffle_index[val_size:]]
        
    ll, cnt = np.unique(train_labels,return_counts=True)
    
    nclass = len(np.unique(labels))
    
        
L = [laplacian(adj, normalized=True)]

train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)
train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)

dset_train = Data.TensorDataset(train_data, train_labels)
train_loader = Data.DataLoader(dset_train, batch_size = args.batchsize, shuffle = True)
dset_test = Data.TensorDataset(test_data, test_labels)
test_loader = Data.DataLoader(dset_test, shuffle = False)



##Delete existing network if exists
try:
    del model
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')



# network parameters
F_0 = 2
D_g = train_data.shape[1] # features(genes)
CL1_F = 5
CL1_K = 5
FC1_F = 32
FC2_F = 0
NN_FC1 = 256
NN_FC2 = 32
out_dim = nclass

net_parameters = [F_0,D_g, CL1_F, CL1_K, FC1_F,FC2_F,NN_FC1, NN_FC2, out_dim]
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.0)
            
# instantiate the object net of the class
net = Graph_GCN(net_parameters)
net.apply(weight_init)

if torch.cuda.is_available():
    net.cuda()
print(net)


# Weights
L_net = list(net.parameters())


# learning parameters
dropout_value = 0.2
l2_regularization = 5e-4
batch_size = args.batchsize
num_epochs = args.epochs
train_size = train_data.shape[0]
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=',num_epochs,', train_size=',train_size,', nb_iter=',nb_iter)


# Optimizer
global_lr = args.lr
global_step = 0
decay = 0.95
decay_steps = train_size


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * pow( decay , float(global_step// decay_steps) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
# optimizer = optim.Adam(net.parameters(),lr= args.lr, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr= args.lr)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

## Train   
net.train()
losses_train = []
acc_train = []


t_total_train = time.time()

for epoch in range(num_epochs):  # loop over the dataset multiple times

    # update learning rate
    cur_lr = adjust_learning_rate(optimizer,epoch)
    
    # reset time
    t_start = time.time()

    # extract batches
    epoch_loss = 0.0
    epoch_acc = 0.0
    count = 0
    
    # confusion_matrix = torch.zeros(nclass, nclass)
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
        optimizer.zero_grad()
        out_gae, out_hidden, output, out_adj = net(batch_x, dropout_value, L)
        
        loss_batch = net.loss(out_gae, batch_x, output, batch_y, l2_regularization)
        acc_batch = utilsdata.accuracy(output, batch_y).item()
        
        loss_batch.backward()
        optimizer.step()
        
        count += 1
        epoch_loss += loss_batch.item()
        epoch_acc += acc_batch
        global_step += args.batchsize
        
        # print
        if count % 1000 == 0: # print every x mini-batches
            print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch + 1, count, loss_batch.item(), acc_batch))


    epoch_loss /= count
    epoch_acc /= count
    losses_train.append(epoch_loss) # Calculating the loss
    acc_train.append(epoch_acc) # Calculating the acc
    # print
    t_stop = time.time() - t_start
    print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
          (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
    print('training_time:',t_stop)

t_total_train = time.time() - t_total_train

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.decomposition import PCA
# Test set


def test_model(loader,num_classes):
    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([num_classes,num_classes])
    predictions = pd.DataFrame()
    y_true = []
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        out_gae, out_hidden, pred, out_adj = net(batch_x, dropout_value, L)
        
        test_acc += utilsdata.accuracy(pred, batch_y).item()
        count += 1
        y_true.append(batch_y.item())
        #y_pred.append(pred.max(1)[1].item())
        confusionGCN[batch_y.item(), pred.max(1)[1].item()] += 1
        px = pd.DataFrame(pred.detach().cpu().numpy())            
        predictions = pd.concat((predictions, px),0)
        
    preds_labels = np.argmax(np.asarray(predictions), 1)
    test_acc = test_acc/float(count)
    predictions.insert(0, 'trueLabels', y_true)

    return test_acc, confusionGCN, predictions, preds_labels


t_start_test = time.time()
test_acc,confusionGCN, predictions, preds_labels = test_model(test_loader, nclass)
t_stop_test = time.time() - t_start_test    
print('  accuracy(test) = %.3f %%, time= %.3f' % (test_acc, t_stop_test))

## compute classification metrics
classification_report = sklearn.metrics.classification_report(test_labels, preds_labels, labels=range(nclass))
print(classification_report)

testPreds4save = pd.DataFrame(preds_labels,columns=['predLabels'])
testPreds4save.insert(0, 'trueLabels', list(predictions.iloc[:,0]))
aa = np.exp(np.asarray(predictions.iloc[:,1:]))
confusionGCN = pd.DataFrame(confusionGCN)


if args.savemodel:
    OutputDir = 'results'
    testPreds4save.to_csv(OutputDir+'/gcn_test_preds_'+ args.database+ str(args.num_gene)+str(args.singleton)+'_'+str(CL1_F)+str(CL1_K)+ '.csv')
    predictions.to_csv(OutputDir+'/gcn_testProbs_preds_'+ args.database+ str(args.num_gene)+str(args.singleton)+'_'+str(CL1_F)+str(CL1_K)+ '.csv')
    confusionGCN.to_csv(OutputDir+'/gcn_confuMat_'+ args.database+ str(args.num_gene)+str(args.singleton)+'_'+str(CL1_F)+str(CL1_K)+ '.csv')    
    np.savetxt(OutputDir+'/gcn_train_time_'+args.database + str(args.num_gene) +str(args.singleton)+'_'+str(CL1_F)+str(CL1_K)+'.txt', [t_total_train])   
    np.savetxt(OutputDir+'/gcn_test_time_'+args.database + str(args.num_gene)+str(args.singleton) +'_'+str(CL1_F)+str(CL1_K)+'.txt', [t_stop_test]) 

    torch.save(net.state_dict(), 'model/net' + str(args.num_gene)+str(args.singleton) + '.pt')


