# Graph Embedding Task

## Task Description:
Graphs are widely used as descriptions for samples in Non-Euclidean spaces, such as social networks and molecule structure. But for deep learning models we need mathematical formats in the Euclidean spaces as the sample representations, so usually we will apply models to embed the graphs into vectors or matrices, and the graphs neural networks (GNN) are often applied with good performance on this purpose. The goal of the this task is to check the power of the graph neural networks on graph embedding, including node-wise embedding and graph-wise embedding. We will provide the protein samples describe as heterogeneous graphs while the nodes are secondary structure elements (can be regarded as the units of the proteins) and the edges are their sequential or spatial relationships. The samples in our dataset are clustered in an hierarchical structure described in SCOPe, so each sample is assigned to one class (which would be the label) in each hierarchy. In this task we considered the fold level hierarchy in SCOPe, so we will provide 1080 classes respectively for the discriminative task. We also provide a protein sequence which can be utilized in the generative challenges.

## Environment
Go to the folder [Enivironment](https://github.tamu.edu/shen-group/data-buffet/tree/shaowen/Environment) and install the environment with the *.yml* file.

## Challenge 1: Discriminative Task on Protein Folds 

### Challenge Description
In this challenge a feature matrix and an adjacency tensor would be provided for each sample and the goal is to predict the label (on fold or family level). The user need to construct their own GNN (either on homogeneous graph or heterogeneous graph, depending on the setting)to get the node-wise or graph-wise embeddings. For graph-wise embedding if will be directly sent to an MLP for classification; for the node-wise embedding the user can select one of the pooling layers we provided for graph-wise embeddings.

### Dataset:
For each sample 1 feature matrix (Numpy array, max_node_amount(60) x feat_dim(11)) and 1 adjacency tensor (Numpy array, channel_num(5) x max_node_amount(60) x max_node_amount(60); for homogeneous graph channel_num = 1). The datasets are stored as lists of Numpy arrays and the label vectors are stored as Numpy arrays. During the training process the feature matrix and the adjacency matrix will be transformed into pytorch Tensors (float).

### Evaluation Criteria: 
Accuracy, precision, recall, F1-score.

### Requirement
The GNN can only and must take the feature matrix and the adjacency tensor as the input.

## Challenge 2: Generative Task on Protein Folds

### Challenge Description
In this challenge the user need to provide 

### Dataset:
For each sample 1 feature matrix (Numpy array), 1 adjacency matrix (Numpy array) and . The datasets are stored as lists of Numpy arrays and the label vectors are stored as Numpy arrays. During the training process the feature matrix and the adjacency matrix will be transformed into pytorch Tensors (float).

### Evaluation Criteria: 
Perplexity, cross-entropy, seqence identity.

### Requirement:
The GNN can only and must take the feature matrix, the adjacency tensor and the sequence tensor as the input.

## Tutorial
Follow the instructions in the [Illustration.ipynb](https://github.tamu.edu/shen-group/data-buffet/blob/shaowen/Challenges/Illustration.ipynb) for more details and ruunning the pipeline.

