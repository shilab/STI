# STI
Split Transformer Impute source code

## Model architecture

<p align="center">
  <img width="100%" height="auto" src="https://github.com/shilab/STI/blob/main/architecture.png">
</p>

## Overal workflow of STI in pseudocode
```
PROGRAM STI:
  Read the data;
  Perform one-hot encoding on the data;
  Partition the data into training, validation, and test sets;
  IF (data is diploid)
      THEN break them into haploids but keep the order intact;
      ELSE do nothing;
  ENDIF;
  FOR each iteration
  	Shuffle haploids in training and validation sets separately;
  	FOR each training batch
		Randomly select 50% of the values in the training batch and replace them with missing values;
  		Train the model using the training batch;
	ENDFOR;
	FOR each validation batch
		Randomly select 50% of the values in the validation batch and replace them with missing values;
  		Evaluate the model using the validation batch;
	ENDFOR;
  ENDFOR;
  Perform prediction using the model on the test set;
  IF (data is diploid)
      THEN replace each two consecutive test samples with the respective diploid
      ELSE do nothing;
  ENDIF;
  Save the resulting predictions into a file;
  
END.
```
## Data

The datasets associated with the paper can be downloaded and processed from the online sources mentioned in the paper. However, we include a copy of the data for more accessibility in **data** directory.

genotype_full.txt > Genotypes for the yeast dataset

HLA.recode.vcf > Genotypes for the HLA dataset

DELL.chr22.genotypes.full.vcf > Genotypes for deletions in chromosome 22

ALL.chr22.mergedSV.v8.20130502.svs.genotypes.vcf > All genotypes in chromosome 22

beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz > The training dataset for missing variant experiment using SNPs on chromosome 22 
test_data_beadchip_hwe_filtered.vcf.gz > The test dataset for missing variant experiment using SNPs on chromosome 22. This file contains Omni2.5 microarray genotypes
test_true_data_beadchip_hwe_filtered.vcf.gz > The ground truth for the missing variants in Omni2.5 microarray dataset.

### Instruction on how to obtain the dataset for the Missing variants experiment (Imputing microarray data using WGS data)
1. Follow the instructions in the following link to obtain the dataset (VCF + Omni BeadChip manifest + Hg19 fast file): https://github.com/kanamekojima/rnnimp
2. [optional] Filter the data using bcftools and/or plink. Sample commands are in `command_used_for_beadchip_ref_filtering.txt` file.
3. Split the data to train and test. Sample test ids we used for our experiment can be found in `test_samples.txt` inside `STI_benchmark_datasets.zip`
Please note that variant IDs should be unique for the code to work correctly.

## General usage in HPC servers:
In order to use STI on a server, you can use `STI V1.1_beta.py` script to train the model(s) and impute the data for sporadic and missing variants cases. Generally, we recommend to use a minimum masking rate of 0.5 and increasing it up to 0.8 accordingly if the missing rate of the target dataset is higher. The list of command line arguments used for `STI V1.1_beta.py` is as follows:

```
--mode {impute,train}
                        Operation mode: impute | train (default=train)
  --restart-training {false,true,0,1}
                        Whether to clean previously saved models in target directory and restart the training
  --ref REF             Reference file path.
  --target TARGET       Target file path. Must be provided in "impute" mode.
  --tihp {false,true,0,1}
                        Whether the target is going to be haps or phased.
  --ref-comment REF_COMMENT
                        The character(s) used to indicate comment lines in the reference file (default="\t").
  --target-comment TARGET_COMMENT
                        The character(s) used to indicate comment lines in the target file (default="\t").
  --ref-sep REF_SEP     The separator used in the reference input file (If -ref-file-format is infer, this
                        argument will be inferred as well).
  --target-sep TARGET_SEP
                        The separator used in the target input file (If -target-file-format is infer, this
                        argument will be inferred as well).
  --ref-vac {false,true,0,1}
                        [Used for non-vcf formats] Whether variants appear as columns in the reference file
                        (default: false).
  --target-vac {false,true,0,1}
                        [Used for non-vcf formats] Whether variants appear as columns in the target file
                        (default: false).
  --ref-fcai {false,true,0,1}
                        [Used for non-vcf formats] Whether the first column in the reference file is (samples |
                        variants) index (default: false).
  --target-fcai {false,true,0,1}
                        [Used for non-vcf formats] Whether the first column in the target file is (samples |
                        variants) index (default: False).
  --ref-file-format {infer,csv,vcf,tsv}
                        Reference file format: infer | vcf | csv | tsv. Default is infer.
  --target-file-format {infer,csv,vcf,tsv}
                        Target file format: infer | vcf | csv | tsv. Default is infer.
  --save-dir SAVE_DIR   the path to save the results and the model. This path is also used to load a trained
                        model for imputation.
  --compress-results {false,true,0,1}
                        Default: true
  --co CO               Chunk overlap in terms of SNPs/SVs(default 100)
  --cs CS               Chunk size in terms of SNPs/SVs(default 2000)
  --sites-per-model SITES_PER_MODEL
                        Number of SNPs/SVs used per model(default 16000)
  --mr MR               Masking rate(default 0.8)
  --val-frac VAL_FRAC   Fraction of reference samples to be used for validation (default=0.1).
  --random-seed RANDOM_SEED
                        Random seed used for splitting the data into training and validation sets (default
                        2022).
  --epochs EPOCHS       Maximum number of epochs (default 1000)
  --na-heads NA_HEADS   Number of attention heads (default 16)
  --embed-dim EMBED_DIM
                        Embedding dimension size (default 128)
  --lr LR               Learning Rate (default 0.001)
  --batch-size-per-gpu BATCH_SIZE_PER_GPU
                        Batch size per gpu(default 2)
```

- To train a model, you need a minimum of arguments in the following command:
  `python3 STI\ V1.1_beta.py --mode train --ref [reference/training file] --tihp [0 | 1] --save-dir [saving directory]`
- To impute the data, you need a minimum of arguments in the following command:
  `python3 STI\ V1.1_beta.py --mode impute --target [target file] --ref [reference/training file] --save-dir [saving directory used for training]`

## Getting Started:

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites
```
Python >= 3.9.11 
Miniconda >= 3.3
```

### Setup

The code is adjusted to use all available GPUs. If you do not want that to happen, use the following command before running the code to prohibit using all GPUs:
`export CUDA_VISIBLE_DEVICES=[comma separated GPU IDs, e,g,. 0,1]`

Install required dependencies:
```
conda create --name <env> --file requirements.txt
```
Then install tensorflow >= 2.13.* using the official tensorflow website: https://www.tensorflow.org/install/pip
```
>git clone https://github.com/shilab/STI.git
>cd STI-hpc
>conda activate <env>
[optional step]>export CUDA_VISIBLE_DEVICES=0,1,2
python3 STIv1.1_beta.py --mode ...
```

## Roadmap
- [x] Add a single script to handle haploids/diploids
- [x] Add GPU(and HPC) support
- [x] Unify the scripts for sporadic missingness imputation and variant imputation

***Please note that the code is under development and you might run into bugs. Feel free to open issues in github to address them.***

## Contact
You can reach out to us regarding your questions , suggestions, and possible collaboration using either of these emails:

M. Erfan Mowlaei: tul67492[at]temple[dot]edu or erfan[dot]molaei[at]gmail[dot]com

Prof. Xinghua Shi: mindyshi[at]temple[dot]edu

## Citation
If you use our model in any project or publication, please cite our paper [Split-Transformer Impute (STI): Genotype Imputation Using a Transformer-Based Model](https://www.biorxiv.org/content/10.1101/2023.03.05.531190v1.abstract)

```
@article {Mowlaei2023.03.05.531190,
	author = {Mowlaei, Mohammad Erfan and Li, Chong and Chen, Junjie and Jamialahmadi, Benyamin and Kumar, Sudhir and Rebbeck, Timothy Richard and Shi, Xinghua},
	title = {Split-Transformer Impute (STI): Genotype Imputation Using a Transformer-Based Model},
	elocation-id = {2023.03.05.531190},
	year = {2023},
	doi = {10.1101/2023.03.05.531190},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {With recent advances in DNA sequencing technologies, researchers are able to acquire increasingly larger volumes of genomic datasets, enabling the training of powerful models for downstream genomic tasks. However, genome scale dataset often contain many missing values, decreasing the accuracy and power in drawing robust conclusions drawn in genomic analysis. Consequently, imputation of missing information by statistical and machine learning methods has become important. We show that the current state-of-the-art can be advanced significantly by applying a novel variation of the Transformer architecture, called Split-Transformer Impute (STI), coupled with improved preprocessing of data input into deep learning models. We performed extensive experiments to benchmark STI against existing methods using resequencing datasets from human 1000 Genomes Project and yeast genomes. Results establish superior performance of our new methods compared to competing genotype imputation methods in terms of accuracy and imputation quality score in the benchmark datasets.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/03/06/2023.03.05.531190},
	eprint = {https://www.biorxiv.org/content/early/2023/03/06/2023.03.05.531190.full.pdf},
	journal = {bioRxiv}
}
```
