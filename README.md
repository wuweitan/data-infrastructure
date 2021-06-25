The main goal is to generate the genome sequence and epigenomic data pairs that can be used to train a noncoding mutation effect prediction model for interested cell lines or epigenomic events. After the epigenomic event changes are predicted, the related gene expression, protein sequences changes may also be provided later.

This pipeline consists two main parts, generating the genome sequence and epignomic events pairing data and generating the corresponding chromatin structure to improve the cell line specific epigenomic events analysis.

All genomic and epigenomic labels can be used.
1. By using the reference gene files, e.g. hg19.knownGene.gft(based on hg19 assembly, can be updated to hg38 later), each selected regions(on DNA) will be linked to the cloest gene. Later, the Gene Ontology terms will be added, to have a better understanding of the functions of the linked genes. The same GO terms can be added to the RNA level gene expression data to help the understanding of the expression changes.
2. Current epigenomic labels can be processed are ChIP-seq(for Transcription factors and histone marks), DNase-seq and ATAC-seq(for chromosome accessibility) for the DNA. RNA-seq can be processed to provide the gene expression level, unfortunately, no expression data available to assess the effects of the noncoding mutations. For the general chromatin structures, HiC and ChIA-PET are available that can provide cell line specific or cell line and transcription factor specific chromatin structure.

To do: 
1. Add the pipeline that can select the ENCODE experiment given the cell line and epigenomic events.

2. Add the indicator for each selected genome regions and the related gene(which gene, GO terms, etc.), which can help the interpretation of each noncoding SNVs.

epigenomic_label_preprocessing(requires numpy)
Provide a list of the N ENCODE experiments(right now uses the ENCODE experiment ID, will be replaced by the combination of cell line and epigenomic events later. The epigenomic events list, especially for TFs, may be replaced by some PDB IDs, if applicable, later.), the code will download the .narrowPeak data and convert it into binary label for the whole genome wide. Then merge all epigenomic event labels as an M*N matrix. M is the number of genome regions. N is the number of epigenomic events used.

Detailed functions:
prepare_whole_genome_bed_file will prepareing a bed file contains all regions(200bp or other length) for the whole genome.

convert_peak_to_binary will convert the peak callling output files(narrowPeak) into binary labels for each region.

merge_binary_label will merge the peak labels among different epigenomic events to find all regions we are interested in(with positive labels)

write_selected_region_to_bed will prepare the bed files contains all regions we are interested in.

select_regions_on_all_peak_files filter all epigenomic label files, only keep the regions we are interested in.

prepare_final_binary_label_matrix will merge all processed epigenomic label files, we will have a M*N file.

hic_processing(requires numpy and encode tools juicer)
Download .hic; extract interaction frequency from .hic. Prepare the interaction frequency matrix(chromatin structure) and genome sequence matching for training, validation and test.

Detailed functions:
download_hic will download the hic files from the ENCODE.

extract_if_from_hic will preprocess the hic files, using the tool juicer, to extract the chromatin pairwise interaction frequency among genome regions.

if_txt_to_npy will convert the output of the juicer into the numpy for each chromatin pairs.

merge_if_npy will merge all chromatin pairs interaction frequency into one genome wide interaction frequency matrix.

The chromatin structure and epigenomic label are matched based on the interaction frequency matrix and selected genome region bed file.
