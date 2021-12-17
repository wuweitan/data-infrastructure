# data-buffet
This repository is to speed up data processing process and provide packed data for quick run

We start with some good example repos and learn the best from them
* [sidechainnet](https://github.com/jonathanking/sidechainnet)
* [atom3d](https://github.com/drorlab/atom3d)
* [deepchem](https://github.com/deepchem/deepchem)
* [TDC](https://github.com/mims-harvard/TDC)

# Protein Structure Data (this branch)
## Discription 
This branch would provide a pipeline to deal with the protein structure databases, including general protein structure databases (e.g. PDB, PDB70), hierarchical database (e.g. SCOPe and CATH) and protein complex database (e.g. SAbDab, AbDb, IMGT and CoV3D). The goal of this branch is to help to learn the structure embedding methods, sequence-struture relationship (sequence distribution modelling or structure prediction) and structure-structure relationship (protein docking based on protein complexes). With the development and the outstanding performance of Alphafold2, we may provide a enlarged database with predicted structure (on UniRef90) in the future once it is available from DeepMind (maybe several months later).

## Correlation with other branches
* For sequence-structure relationship (with Yuanfei): though the sequeces are provided once the structures (pdb files) are available, more homologous sequences would be provided from the sequence databases and they may share similar structures. 
* For protein complexes (with Yuning and Rujie): while my projects focus on modeling the conditional sequences distribution, the protein-protein interation is also important to determine the condition, so there is an overlapping part with Rujie and Yuning's work. While they would consider a more expansive space, we may share our pipelines to each other.
* Mutation effect (with Wuwei): the modeled sequence distribution can be applied to predict the mutation effect, which can be compared with Wuwei's work on genes.

## Blueprint
A hierarchical tree will be provided (like SCOPe, to show the clusters of structures) with the statistics of each dataset. Users may refer to the statistics to select their datasets. They may also search for a benchmark dataset with the name. Once they have determined a list they can directly downloaded the processed data and then pursue the steps like data splitting, data loading, model training (require their own model) and evaluations (some classical metrics will be provided). For some unseen data the may also apply our process pipeline to deal with as long as a required format (i.e. pdb) is provided.

### Data Process
#### For samples
* Information Read: extract aa sequences, get ss and sa, calculate the rmsd, irmsd (for complexes) and fnat (for complexes), etc.
* Structure represenation: gcWGAN representations (fold-level), cVAE grammar (fold-level), SSE graphs (fold-level), residue-residue graphs (structure level) and atomic graphs (structure level).
#### For datasets
* Query the datasets with the name of the benchmark DBs or a threshold (on sequence or structure similarity).
* Data split process

### DataLoader
* Loading the processed datasets and generate batches.

### Evaluation
#### Sequence
* Sequence identity 
* Sequence coverage 
* Perplexity

#### Structure
* TMscore 
* RMSD 

#### Compexes
* iRMSD
* fnat

# Challenges
For the prepared tasks go the folder *Challenges*.
