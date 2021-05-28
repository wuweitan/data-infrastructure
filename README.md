hic_processing
Download .hic; extract interaction frequency from .hic. Prepare the interaction frequency matrix(chromatin structure) and genome sequence matching for training, validation and test.

epigenomic_label_preprocessing
Provide a list of the N ENCODE experiments(right now uses the ENCODE experiment ID, need to be replaced by the combination of cell line and epigenomic events later), the code will download the .narrowPeak data and convert it into binary label for the whole genome wide. Then merge all epigenomic event labels as an M*N matrix. M is the number of genome regions. N is the number of epigenomic events used.

The chromatin structure and epigenomic label are matched.
