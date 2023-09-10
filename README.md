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
4. Use the code provided (scripts/test_data_preparation.py) in https://github.com/kanamekojima/rnnimp to generate the test set (microarray data).
5. Use `hapslegend2vcf.py` to convert the output of the last step to vcf format. This code is provided by Dr Kaname Kojima under MIT license.
Please note that variant IDs should be unique for the code to work correctly.


### Using your data

The model is not bound to a specific file format and as long as your inputs are one-hot encoded, the model should be able to handle them (minor pre-/post-processing might be needed, e.g., diploid > haploids during pre-processing and reverse operations during post-processing)


## Source code for the experiments
**notebooks\_experiment** contains the code for all of our experiments in jupyter notebooks. The only thing to consider before running them is to replace **[path] | [data_path] | [save_path]** in file paths in each notebook. Excluding _AE_ that only runs of GPU, the rest of models should be runnable on Google Colab TPU/GPU without changes.

## Source codes for result analysis [old]
**notebooks_result_analysis** contains the codes we used to calculate evaluation metrics on the saved results of the experiments. Visualization codes are not included.

## Demo
### Sporadic missing imputation
In case you want to have a quick demo, we suggest you to use **[TPU] STI Chr.22 ALL.ipynb** notebook in **notebooks_experiment** folder using **ALL.chr22.mergedSV.v8.20130502.svs.genotypes.vcf** dataset, which is one of the relatively small datasets. The expected outputs are present in the same notebook as well and you can compare them to the outputs you obtain. The training and testing time for each fold in this notebook should take ~15 minutes if you are using TPU on google colab.

### Missing variants imputation
**[TPU][Beadchip] STI HMR WGS+Microarray.ipynb** contains the code we used to train the model and impute the test set for this experiment. The output of the previous notebook is numpy arrays. You need to use **[TPU][Beadchip] Ligate.ipynb** in order to convert them to a vcf file.

## Getting Started:

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites
```
Python >= 3.9.11 
virtualenv >= 16.4.3
Jupyterlab >= 4.0.0a36
```

### Setup

We suggest running the code using Google Colab to utilize supreme power of TPU for faster training time. However, if you want to run the code on a machine, please follow the guide below:
Create virtual environment

Install requirment dependents
```
pip3 install scipy==1.7.3 sklearn==1.0.1 pandas==1.3.4 tensorflow>=2.11 jupyterlab matplotlib3.5.0 seaborn==0.12.1 scikit-allel==1.3.5
```

Then download the project and start jupyter lab to run the codes
```
git clone https://github.com/shilab/STI.git
cd STI-main
mkdir venv
python3 -m venv venv/
source venv/bin/activate
jupyter-lab
```

## Roadmap
- Add a single script to handle haploids/diploids
- Add GPU(and HPC) support
- Unify the scripts for sporadic missingness imputation and variant imputation

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
