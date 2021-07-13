# data-buffet

## Plan:

### Blueprint (Bio-knowledge graph)
* Available onjects: protein, compound, function ontology, desease; in total 5 * 5 - 5 = 20 relationships
* Question I can answer: modeling relationship between protein and compound
* Under-explored questions: 19 relationships

### Implementation
* Quiry on data (DeepAffinity, DeepRelations, Platinum):
  * Raw data: protein sequence with uniprot id, compound smile, interaction label (more interaction label by plip).
  * Processed data in npy format. [Done]
* Functions to process data:
  * Convert protein sequence to tokenized sequence. [Done]
  * Conver compound smile to graphs. [Done]
  * Dataset split based on similarity.
  * RCSB GraphQL query demo [Done]
