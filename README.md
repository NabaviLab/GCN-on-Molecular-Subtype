# GCN-on-Molecular-Subtype
This is the code for our paper, Cancer molecular subtype classification by graph convolutional networks on multi-omics data. [Paper](https://dl.acm.org/doi/abs/10.1145/3459930.3469542)

## Set up
Major dependencies are listed in the requirements.txt file
## Data
All the data needed can be downloaded from the Google drive. [Data Folder](https://drive.google.com/drive/folders/1sp6tv9iSvo_m9hy6nZl_ZbptmAkqueg2?usp=sharing)
## Run the code
```
## Run a single omic (expression) model with 1000 genes (with singletons) and BioGrid network
python3 main.py --num_gene 1000
```