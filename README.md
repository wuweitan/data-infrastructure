# data-buffet

## Plan:


### Blueprint (Bio-knowledge graph)
* Available onjects: protein, compound, function ontology, desease, cellline; in total (6 * 6 - 6) / 2 + 6 = 21 relationships
* Relations to be explored
  * protein vs compound (DeepAffinity, DeepRelations, Platinum)
  * protein vs cell line (wuwei)
  * protein vs protein (rujie)
  * protein vs function (yue)
  * desease vs function (mostafa)
* Final goal: Modeling heterogeneous knowledge graph rather than isolated relations


### Implementation (protein vs compound)
* Quiry on data (DeepAffinity, DeepRelations, Platinum):
  * Raw data: protein sequence with uniprot id, compound smile, interaction label (more interaction label by plip).
  * Processed data in npy format. [Done]
* Functions to process data:
  * Convert protein sequence to tokenized sequence. [Done]
  * Conver compound smile to graphs. [Done]
  * Dataset split based on similarity.
  * RCSB GraphQL query demo [Done]
