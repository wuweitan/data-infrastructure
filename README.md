# data-buffet

## Plan:


### Blueprint (Bio-knowledge graph)
* Available objects: (i) protein, (ii) compound, (iii) function ontology, (iv) desease, (v) cellline; in total (5 * 5 - 5) / 2 + 5 = 15 relationships
* Relations to be explored
  * protein vs compound (DeepAffinity, DeepRelations, Platinum)
  * protein vs cellline (wuwei)
  * cellline vs cellline (wuwei)
  * protein vs protein (rujie)
  * protein vs function (yue)
  * desease vs function (mostafa)
  * function vs function (GO term)
* Final goal: Modeling heterogeneous knowledge graph rather than isolated relations
* Priority:
  * PPI mutation effect
  * CPI celline specific
  * Protein graphs, hierarchy

### Implementation (protein vs compound)
* Quiry on data (DeepAffinity, DeepRelations, Platinum):
  * Raw data: protein sequence with uniprot id, compound smile, interaction label (more interaction label by plip).
  * Processed data in npy format. [Done]
* Functions to process data:
  * Convert protein sequence to tokenized sequence. [Done]
  * Conver compound smile to graphs. [Done]
  * Dataset split based on similarity.
  * RCSB GraphQL query demo [Done]
