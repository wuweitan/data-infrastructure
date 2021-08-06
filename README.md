# data-buffet
This repository is to speed up data processing process and provide packed data for quick run

We start with some good example repos and learn the best from them
* [sidechainnet](https://github.com/jonathanking/sidechainnet)
* [atom3d](https://github.com/drorlab/atom3d)
* [deepchem](https://github.com/deepchem/deepchem)
* [TDC](https://github.com/mims-harvard/TDC)

# Protein Structure Data (this branch)
## Discription 
This branch would provide a pipeline to deal with the protein structure databases, including general protein structure databases (e.g. PDB), hierarchical database (focus on the geometric shape of the structures; including SCOPe and CATH) and protein complex database (e.g. SAbDab, AbDb, IMGT and CoV3D). With the development and the outstanding performance of Alphafold2, we may provide a enlarged database with predicted structure (on UniRef90) in the future once it is available from DeepMind (maybe several months later).


## Correlation with other branches
* For sequence-structure relationship (with Yuanfei): though the sequeces are provided once the structures (pdb files) are available, more homologous sequences would be provided from the sequence databases and they may share similar structures. 
* 


## Blueprint
The goal of this branch is to help to learn the structure embedding methods (both for discriminative and generative tasks, single-domain proteins and complexes) and descover the sequence-structure relationship problems (coorperate with the sequence data branch). The generative tasks are focused on learning the distribution of the protein sequences given the structures (conditional distribution, can also be transferred into joint distributions) which would also serve the mutation effect prediction tasks.

Users may directly downloaded the processed data and then pursue the steps like data splitting, data loading, model training (require their own model) and evaluations with our pipeline. They may also construct their own dataset with a list of PDB IDs and then do the data process with the pipeline which consists of downloading the data (raw data or processed data) and data preprocess (extract the information and change the format).

### Dataset Process

### DataLoader

### Evaluation
