# data-buffet

### Plan:
Implementation
* Quiry on data (DeepAffinity, DeepRelations, Platinum for user):
  * Raw data: protein sequence with uniprot id, compound smile, interaction label, (more interaction label by plip).
  * Processed data in npy format. [Done]
* Functions to process data (for developer):
  * Convert protein sequence to tokenized sequence. [Done]
  * Conver compound smile to graphs. [Done]
  * Dataset split based on similarity.
  * RCSB GraphQL query demo [Done]
