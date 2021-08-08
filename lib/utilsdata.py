#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:00:37 2019

@author: bingjun
@author: tianyu
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import Normalizer
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics.pairwise import euclidean_distances
import os
from sklearn import preprocessing
from sklearn import linear_model

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def high_variance_expression_gene(expression_variance_path, non_null_path, num_gene, singleton=False):
    gene_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)
    if singleton:
        non_null_row = pd.read_csv(non_null_path, sep=',', header=0)
        gene_variance['id'] = range(gene_variance.shape[0])
#         print(non_null_row['gene'])
        gene_variance_non_null = gene_variance.loc[gene_variance.index.isin(non_null_row['gene']),:]
        gene_list = gene_variance_non_null.nlargest(num_gene, 'variance').index
#         print(gene_list)
        gene_variance_non_null.index = gene_variance_non_null['id']
        gene_list_index = gene_variance_non_null.nlargest(num_gene, 'variance').index
    else:
        ## load expression data
        # print(gene_variance['variance'])
        gene_list = gene_variance.nlargest(num_gene, 'variance').index
        gene_variance.index = range(gene_variance.shape[0])
        gene_list_index = gene_variance.nlargest(num_gene, 'variance').index
    return gene_list, gene_list_index

def load_singleomic_data(expression_data_path):
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    return expression_data

def load_multiomics_data(expression_data_path, cnv_data_path):
    ## load multi-omics data
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    cnv_data = pd.read_csv(cnv_data_path, sep='\t', index_col=0, header=0)
    return expression_data, cnv_data

def downSampling_singleomics_data(expression_variance_path, expression_data, non_null_index_path, shuffle_index_path, adjacency_matrix_path, number_gene, singleton=False):
    ## obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    ## get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1
    
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    num_samples = expression_data.shape[0]
    ## concatenate expr and cnv
    data= np.asarray(expression_data).reshape(num_samples, -1 ,1)
    # data = np.concatenate([data, np.asarray(expression_data).reshape(num_samples, -1, 1) ], axis = 2)

    ## load adjacency matrix
    adj = sp.load_npz(adjacency_matrix_path)
    adj_mat = adj.todense()
    adj_mat_selected = adj_mat[high_variance_gene_index,:]
    adj_mat_selected = adj_mat_selected[:,high_variance_gene_index]
    print(adj_mat_selected.shape)

    ## convert the dense matrix back to sparse matrix
    adj_selected = sp.csr_matrix(adj_mat_selected)

    if singleton:
        print('including singleton')
        adj_selected = adj_selected + sp.eye(adj_selected.shape[0])

    # del features['iCluster']
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    # print(shuffle_index.shape)
    
    return adj_selected, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()

def downSampling_multiomics_data(expression_variance_path, expression_data, cnv_data, non_null_index_path, shuffle_index_path, adjacency_matrix_path, number_gene, singleton=False):
    ## obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    ## get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1
    
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    cnv_data = cnv_data.loc[:,high_variance_gene_list]
    num_samples = cnv_data.shape[0]
    ## concatenate expr and cnv
    data= np.asarray(cnv_data).reshape(num_samples, -1 ,1)
    data = np.concatenate([data, np.asarray(expression_data).reshape(num_samples, -1, 1) ], axis = 2)

    ## load adjacency matrix
    adj = sp.load_npz(adjacency_matrix_path)
    adj_mat = adj.todense()
    adj_mat_selected = adj_mat[high_variance_gene_index,:]
    adj_mat_selected = adj_mat_selected[:,high_variance_gene_index]
    print(adj_mat_selected.shape)

    ## convert the dense matrix back to sparse matrix
    adj_selected = sp.csr_matrix(adj_mat_selected)

    if singleton:
        print('including singleton')
        adj_selected = adj_selected + sp.eye(adj_selected.shape[0])

    # del features['iCluster']
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    # print(shuffle_index.shape)
    
    return adj_selected, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()


       
        
