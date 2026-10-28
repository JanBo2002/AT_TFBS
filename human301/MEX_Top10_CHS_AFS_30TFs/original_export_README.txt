ZENODO MEX-Archipelago data, v2024.1
====================================

This repository contains preprocessed Codebook-MEX data suitable for training 
and testing advanced models of transcription factor binding sites, see README for details.
These data are derived from the ChIP-Seq and genomic HT-SELEX datasets 
of the primary MEX data release (ZENODO doi:10.5281/zenodo.8327372). 

Here we provide:

(1) TFs_CHS+AFS.txt: list of Codebook transcription factors (TFs, including positive controls) 
with at least one ChIP-Seq plus at least one GHT-SELEX dataset, 
only approved-by-curation MEX-eligible data are taken into account.

datasets_approved.txt: list of approved Codebook datasets for the respective TFs.

(2) pwms.zip: sets of top20 log-odds PWMs for each CHS (ChIP-Seq) and GHTS (genomic HT-SELEX) 
replicate obtained from all Codebook data types.
To allow proper testing in the transfer setting (CHS>GHTS and GHTS>CHS),
the PWM set for training CHS models does not include any GHTS-derived PWMs and vice versa.

pcmppms.zip: the original ppm (normalized position probability matrices) or pcm (position count matrices) files.

(3) datasets.zip: pre-processed training and test data for each TF and each data type (CHS or GHTS):

(a) positives.bed: (MEX peaks, pooled across replicates), 301bp regions centered at peak summits, hg38 genome release

(b) 3 types of negative controls.
aliens.bed: regions sampled from peaks of other TFs, 100:1 class balance, 
            or all eligible peaks if there were fewer candidate negative peaks than necessary
random.bed: random genomic regions, 100:1 class balance
shades.bed: regions neighboring the positive regions, 2:1 class balance
            as both upstream and downstream regions were considered.

Please refer to the 'Methods' in MEX publication for more details.

Note 1: 'aliens' and 'random' are GC-matched to positives.
Note 2: the train and test data use separate subsets of chromosomes.
