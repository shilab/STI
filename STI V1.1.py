import os, sys, math, re, random, shutil, gzip, argparse
from tqdm import tqdm
import numpy as np
from typing import Union
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras.applications import efficientnet as efn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from tensorflow.keras.constraints import Constraint
from scipy.spatial.distance import squareform
print("Tensorflow version " + tf.__version__)

strategy = tf.distribute.MirroredStrategy()
N_REPLICAS = strategy.num_replicas_in_sync
print(f"Num gpus to be used: {N_REPLICAS}")

class DataReader:
    """
    If the reference is unphased, cannot handle phased target data, so the valid (ref, target) combinations are:
    (phased, phased), (phased, unphased), (unphased, unphased)
    If the reference is haps, the target cannot be unphased (can we merge every two haps to form unphased diploids?)
    Important note: for each case, the model should be trained separately
    """

    def __init__(self, ):
        self.target_is_gonna_be_phased = None
        self.target_set = None
        self.target_sample_value_index = 2
        self.ref_sample_value_index = 2
        self.target_file_extension = None
        self.allele_count = 2
        self.genotype_vals = None
        self.ref_is_phased = None
        self.reference_panel = None
        self.VARIANT_COUNT = 0
        self.is_phased = False
        self.MISSING_VALUE = None
        self.ref_is_hap = False
        self.target_is_hap = False
        self.ref_n_header_lines = []
        self.ref_n_data_header = ""
        self.target_n_header_lines = []
        self.target_n_data_header = ""
        self.ref_separator = None
        self.map_values_1_vec = np.vectorize(self.map_hap_2_ind_parent_1)
        self.map_values_2_vec = np.vectorize(self.map_hap_2_ind_parent_2)
        self.map_haps_to_vec = np.vectorize(self.map_haps_2_ind)
        self.delimiter_dictionary = {"vcf": "\t", "csv": ",", "tsv": "\t", "infer": "\t"}
        self.ref_file_extension = "vcf"
        self.test_file_extension = "vcf"
        self.target_is_phased = True
        ## Idea: keep track of possible alleles in each variant, and filter the predictions based on that

    def read_csv(self, file_path, is_vcf=False, is_reference=False, separator="\t", first_column_is_index=True,
                 comments="##") -> pd.DataFrame:
        """
        In this form the data should not have more than a column for ids. The first column can be either sample ids or variant ids. In case of latter, make sure to pass :param variants_as_columns=True. Example of sample input file:
        ## Comment line 0
        ## Comment line 1
        Sample_id 17392_chrI_17400_T_G ....
        HG1023               1
        HG1024               0
        """
        print("Reading the file...")
        data_header = None
        path_sep = "/" if "/" in file_path else os.path.sep
        root, ext = os.path.splitext(file_path)
        with gzip.open(file_path, 'rt') if ext == '.gz' else open(file_path, 'rt') as f_in:
            # skip info
            while True:
                line = f_in.readline()
                if line.startswith(comments):
                    if is_reference:
                        self.ref_n_header_lines.append(line)
                    else:
                        self.target_n_header_lines.append(line)
                else:
                    data_header = line
                    break
        if data_header is None:
            raise IOError("The file only contains comments!")
        df = pd.read_csv(file_path,
                         sep=separator,
                         comment=comments[0],
                         index_col=0 if first_column_is_index else None,
                         dtype='category',
                         names=data_header.strip().split(separator) if is_vcf else None)
        # df = df.astype('category')
        return df

    def find_file_extension(self, file_path, file_format, delimiter):
        # Default assumption
        separator = "\t"
        found_file_format = "vcf"

        if file_format not in {"vcf", "csv", "tsv", "infer"}:
            raise ValueError("File extension must be one of {'vcf', 'csv', 'tsv', 'infer'}.")
        if file_format == 'infer':
            file_name_tokenized = file_path.split(".")
            for possible_extension in file_name_tokenized[::-1]:
                if possible_extension in {"vcf", "csv", "tsv"}:
                    found_file_format = possible_extension
                    separator = self.delimiter_dictionary[possible_extension] if delimiter is None else delimiter
                    break
        else:
            found_file_format = file_format
            separator = self.delimiter_dictionary[file_format] if delimiter is None else delimiter

        return found_file_format, separator

    def assign_training_set(self, file_path: str,
                            target_is_gonna_be_phased_or_haps: bool,
                            variants_as_columns: bool = False,
                            delimiter=None,
                            file_format="infer",
                            first_column_is_index=True,
                            comments="##") -> None:
        """
        :param file_path: reference panel or the training file path. Currently, VCF, CSV, and TSV are supported
        :param target_is_gonna_be_phased: Indicates whether the targets for the imputation will be phased or unphased.
        :param variants_as_columns: Whether the columns are variants and rows are samples or vice versa.
        :param delimiter: the seperator used for the file
        :param file_format: one of {"vcf", "csv", "tsv", "infer"}. If "infer" then the class will try to find the extension using the file name.
        :param first_column_is_index: used for csv and tsv files to indicate if the first column should be used as identifier for samples/variants.
        :param comments: The token to be used to filter out the lines indicating comments.
        :return: None
        """
        self.target_is_gonna_be_phased = target_is_gonna_be_phased_or_haps
        self.ref_file_extension, self.ref_separator = self.find_file_extension(file_path, file_format, delimiter)
        if file_format == "infer":
            print(f"Ref file format is {self.ref_file_extension} and Ref file sep is {self.ref_separator}.")

        self.reference_panel = self.read_csv(file_path, is_reference=True, is_vcf=False, separator=self.ref_separator,
                                             first_column_is_index=first_column_is_index,
                                             comments=comments) if self.ref_file_extension != 'vcf' else self.read_csv(
            file_path, is_reference=True, is_vcf=True, separator='\t', first_column_is_index=False, comments="##")

        if self.ref_file_extension != "vcf":
            if variants_as_columns:
                self.reference_panel = self.reference_panel.transpose()
            self.reference_panel.reset_index(drop=False, inplace=True)
            self.reference_panel.rename(columns={self.reference_panel.columns[0]: "ID"}, inplace=True)
        else:  # VCF
            self.ref_sample_value_index += 8

        self.ref_is_hap = not ("|" in self.reference_panel.iloc[0, self.ref_sample_value_index] or "/" in
                               self.reference_panel.iloc[0, self.ref_sample_value_index])
        self.ref_is_phased = "|" in self.reference_panel.iloc[0, self.ref_sample_value_index]
        ## For now I won't support merging haploids into unphased data
        if self.ref_is_hap and not target_is_gonna_be_phased_or_haps:
            raise ValueError(
                "The reference contains haploids while the target will be unphased diploids. The model cannot predict the target at this rate.")

        if not (self.ref_is_phased or self.ref_is_hap) and target_is_gonna_be_phased_or_haps:
            raise ValueError(
                "The reference contains unphased diploids while the target will be phased or haploid data. The model cannot predict the target at this rate.")

        self.VARIANT_COUNT = self.reference_panel.shape[0]
        print(f"{self.VARIANT_COUNT} {'haplotype' if self.ref_is_hap else 'diplotype'} variants found!")

        self.is_phased = target_is_gonna_be_phased_or_haps and (self.ref_is_phased or self.ref_is_hap)

        original_allele_sep = "|" if self.ref_is_phased or self.ref_is_hap else "/"
        final_allele_sep = "|" if self.is_phased else "/"

        def get_num_allels(g):
            v1, v2 = g.split(final_allele_sep)
            return max(int(v1), int(v2)) + 1

        genotype_vals = np.unique(self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].values)
        if self.ref_is_phased and not target_is_gonna_be_phased_or_haps:  # In this case ref is not haps due to the above checks
            # Convert phased values in the reference to unphased values
            phased_to_unphased_dict = {}
            for i in range(genotype_vals.shape[0]):
                key = genotype_vals[i]
                v1, v2 = [int(s) for s in genotype_vals[i].split(original_allele_sep)]
                genotype_vals[i] = f"{min(v1, v2)}/{max(v1, v2)}"
                phased_to_unphased_dict[key] = genotype_vals[i]
            self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].replace(phased_to_unphased_dict,
                                                                                   inplace=True)

        self.genotype_vals = np.unique(genotype_vals)

        self.allele_count = max(map(get_num_allels, self.genotype_vals)) if not self.ref_is_hap else len(
            self.genotype_vals)
        self.MISSING_VALUE = self.allele_count if self.is_phased else len(self.genotype_vals)

        def key_gen(v1, v2):
            return f"{v1}{final_allele_sep}{v2}"

        if self.is_phased:
            self.hap_map = {str(i): i for i in range(self.allele_count)}
            self.hap_map.update({".": self.allele_count})
            self.r_hap_map = {i: k for k, i in self.hap_map.items()}
            self.map_preds_2_allele = np.vectorize(lambda x: self.r_hap_map[x])
        else:
            unphased_missing_genotype = "./."
            self.replacement_dict = {g: i for i, g in enumerate(self.genotype_vals)}
            self.replacement_dict[unphased_missing_genotype] = len(self.genotype_vals)
            self.reverse_replacement_dict = {v: k for k, v in enumerate(self.replacement_dict)}

        self.SEQ_DEPTH = self.allele_count + 1
        print("Done!")

    def assign_test_set(self, file_path,
                        variants_as_columns=False,
                        delimiter=None,
                        file_format="infer",
                        first_column_is_index=True,
                        comments="##") -> None:
        """
        :param file_path: reference panel or the training file path. Currently, VCF, CSV, and TSV are supported
        :param variants_as_columns: Whether the columns are variants and rows are samples or vice versa.
        :param delimiter: the seperator used for the file
        :param file_format: one of {"vcf", "csv", "tsv", "infer"}. If "infer" then the class will try to find the extension using the file name.
        :param first_column_is_index: used for csv and tsv files to indicate if the first column should be used as identifier for samples/variants.
        :param comments: The token to be used to filter out the lines indicating comments.
        :return: None
        """
        if self.reference_panel is None:
            raise RuntimeError("First you need to use 'DataReader.assign_training_set(...) to assign a training set.' ")

        self.target_file_extension, separator = self.find_file_extension(file_path, file_format, delimiter)

        test_df = self.read_csv(file_path, is_reference=False, is_vcf=False, separator=separator,
                                first_column_is_index=first_column_is_index,
                                comments=comments) if self.ref_file_extension != 'vcf' else self.read_csv(file_path,
                                                                                                          is_reference=False,
                                                                                                          is_vcf=True,
                                                                                                          separator='\t',
                                                                                                          first_column_is_index=False,
                                                                                                          comments="##")

        if self.target_file_extension != "vcf":
            if variants_as_columns:
                test_df = test_df.transpose()
            test_df.reset_index(drop=False, inplace=True)
            test_df.rename(columns={test_df.columns[0]: "ID"}, inplace=True)
        else:  # VCF
            self.target_sample_value_index += 8

        self.target_is_hap = not ("|" in test_df.iloc[0, self.target_sample_value_index] or "/" in test_df.iloc[
            0, self.target_sample_value_index])
        is_phased = "|" in test_df.iloc[0, self.target_sample_value_index]
        test_var_count = test_df.shape[0]
        print(f"{test_var_count} {'haplotype' if self.target_is_hap else 'diplotype'} variants found!")
        if (self.target_is_hap or is_phased) and not (self.ref_is_phased or self.ref_is_hap):
            raise RuntimeError("The training set contains unphased data. The target must be unphased as well.")
        if self.ref_is_hap and not (self.target_is_hap or is_phased):
            raise RuntimeError(
                "The training set contains haploids. The current software version supports phased or haploids as the target set.")

        self.target_set = test_df.merge(right=self.reference_panel["ID"], on='ID', how='right')
        if self.target_file_extension == "vcf" == self.ref_file_extension:
            self.target_set[self.reference_panel.columns[:9]] = self.reference_panel[self.reference_panel.columns[:9]]
        self.target_set = self.target_set.astype('str')
        self.target_set.fillna("." if self.target_is_hap else ".|." if self.is_phased else "./.", inplace=True)
        self.target_set = self.target_set.astype('category')
        print("Done!")

    def map_hap_2_ind_parent_1(self, x) -> int:
        return self.hap_map[x.split('|')[0]]

    def map_hap_2_ind_parent_2(self, x) -> int:
        return self.hap_map[x.split('|')[1]]

    def map_haps_2_ind(self, x) -> int:
        return self.hap_map[x]

    def __diploids_to_hap_vecs(self, data: pd.DataFrame) -> np.ndarray:
        _x = np.empty((data.shape[1] * 2, data.shape[0]), dtype=np.int32)
        _x[0::2] = self.map_values_1_vec(data.values.T)
        _x[1::2] = self.map_values_2_vec(data.values.T)
        return _x

    def __get_forward_data(self, data: pd.DataFrame) -> np.ndarray:
        if self.is_phased:
            is_haps = "|" not in data.iloc[0, 0]
            print(f"__get_forward_data > data.iloc[0, 0]={data.iloc[0, 0]}, is_haps={is_haps}")
            if not is_haps:
                return self.__diploids_to_hap_vecs(data)
            else:
                return self.map_haps_to_vec(data.values.T)
        else:
            return data.replace(self.replacement_dict).values.T.astype(np.int32)

    def get_ref_set(self, starting_var_index=0, ending_var_index=0) -> np.ndarray:
        if 0 <= starting_var_index < ending_var_index:
            return self.__get_forward_data(
                data=self.reference_panel.iloc[starting_var_index:ending_var_index, self.ref_sample_value_index - 1:])
        else:
            print("No variant indices provided or indices not valid, using the whole sequence...")
            return self.__get_forward_data(data=self.reference_panel.iloc[:, self.ref_sample_value_index - 1:])

    def get_target_set(self, starting_var_index=0, ending_var_index=0) -> np.ndarray:
        if 0 <= starting_var_index < ending_var_index:
            return self.__get_forward_data(
                data=self.target_set.iloc[starting_var_index:ending_var_index, self.target_sample_value_index - 1:])
        else:
            print("No variant indices provided or indices not valid, using the whole sequence...")
            return self.__get_forward_data(data=self.target_set.iloc[:, self.target_sample_value_index - 1:])

    def convert_genotypes_to_vcf(self, genotypes, pred_format="GT:DS:GP"):
        n_samples, n_variants = genotypes.shape
        new_vcf = self.target_set.copy()
        new_vcf.iloc[:n_variants, 9:] = genotypes.T
        new_vcf["FORMAT"] = pred_format
        new_vcf["QUAL"] = "."
        new_vcf["FILTER"] = "."
        new_vcf["INFO"] = "IMPUTED"
        return new_vcf

    def convert_hap_probs_to_diploid_genotypes(self, allele_probs) -> np.ndarray:
        n_haploids, n_variants, n_alleles = allele_probs.shape
        allele_probs_normalized = softmax(allele_probs, axis=-1)

        if n_haploids % 2 != 0:
            raise ValueError("Number of haploids should be even.")

        n_samples = n_haploids // 2
        genotypes = np.zeros((n_samples, n_variants), dtype=object)

        for i in tqdm(range(n_samples)):
            # haploid_1 = allele_probs_normalized[2 * i]
            # haploid_2 = allele_probs_normalized[2 * i + 1]

            for j in range(n_variants):
                # phased_probs = np.multiply.outer(haploid_1[j], haploid_2[j]).flatten()
                # unphased_probs = np.array([phased_probs[0], sum(phased_probs[1:3]), phased_probs[-1]])
                # unphased_probs_str = ",".join([f"{v:.6f}" for v in unphased_probs])
                # alt_dosage = np.dot(unphased_probs, [0, 1, 2])
                variant_genotypes = [str(v) for v in np.argmax(allele_probs_normalized[i * 2:(i + 1) * 2, j], axis=-1)]
                genotypes[i, j] = '|'.join(variant_genotypes)  # + f":{alt_dosage:.3f}:{unphased_probs_str}"

        return genotypes

    def convert_hap_probs_to_hap_genotypes(self, allele_probs) -> np.ndarray:
        allele_probs_normalized = softmax(allele_probs, axis=-1)
        return np.argmax(allele_probs_normalized, axis=1).astype(str)

    def convert_unphased_probs_to_genotypes(self, allele_probs) -> np.ndarray:
        n_samples, n_variants, n_alleles = allele_probs.shape
        allele_probs_normalized = softmax(allele_probs, axis=-1)
        genotypes = np.zeros((n_samples, n_variants), dtype=object)

        for i in tqdm(range(n_samples)):
            for j in range(n_variants):
                unphased_probs = allele_probs_normalized[i, j]
                variant_genotypes = np.vectorize(self.reverse_replacement_dict.get)(
                    np.argmax(unphased_probs, axis=-1)).flatten()
                genotypes[i, j] = variant_genotypes

        return genotypes

    def __get_headers_for_output(self, contain_probs):
        headers = ["##fileformat=VCFv4.2",
                   '''##source=STI v1.1.0''',
                   '''##INFO=<ID=IMPUTED,Number=0,Type=Flag,Description="Marker was imputed">''',
                   '''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">''',
                   ]
        probs_headers = [
            '''##FORMAT=<ID=DS,Number=A,Type=Float,Description="Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]">''',
            '''##FORMAT=<ID=GP,Number=G,Type=Float,Description="Estimated Posterior Probabilities for Genotypes 0/0, 0/1 and 1/1">''']
        return headers.extend(probs_headers) if contain_probs else headers

    def preds_to_genotypes(self, predictions: Union[str, np.ndarray]) -> pd.DataFrame:
        """
        :param predictions: The path to numpy array stored on disk or numpy array of (n_samples, n_variants, n_alleles)
        :return: numpy array of the same shape, with genotype calls, e.g., "0/1"
        """
        if isinstance(predictions, str):
            preds = np.load(predictions)
        else:
            preds = predictions

        target_df = self.target_set.copy()
        if not self.is_phased:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.convert_unphased_probs_to_genotypes(
                preds).T
        elif self.target_is_hap:
            target_df[target_df.columns[self.target_sample_value_index - 1:]] = self.convert_hap_probs_to_hap_genotypes(
                preds).T
        else:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.convert_hap_probs_to_diploid_genotypes(
                preds).T
        return target_df

    def write_ligated_results_to_file(self, df: pd.DataFrame, file_name: str) -> None:
        with gzip.open(file_name, 'wt') if file_name.endswith(".gz") else open(file_name, 'wt') as f_out:
            # write info
            if self.ref_file_extension == "vcf":
                f_out.write("\n".join(self.__get_headers_for_output(contain_probs=False)) + "\n")
            else:  # Not the best idea?
                f_out.write("\n".join(self.ref_n_header_lines))
        df.to_csv(file_name, sep=self.ref_separator, mode='a', index=False)

def main(args):
    '''
    target_is_gonna_be_phased_or_haps:bool,
    variants_as_columns:bool=False,
    delimiter=None,
    file_format="infer",
    first_column_is_index=True,
    comments="##"
    '''
    parser = argparse.ArgumentParser(description='ShiLab\'s Imputation model (STI v1.1).')

    ## Function mode
    parser.add_argument('-fm', type=str, help='Operation mode: impute | train (default=train)',
                        choices=['impute', 'train'], default='train')
    ## Input args
    parser.add_argument('-ref', type=str, required=True, help='Reference file path')
    parser.add_argument('-target', type=str, required=False, help='[optional] Target file path')
    parser.add_argument('-tihp', type=bool, required=True, help='Whether the target is going to be haps or phased.')
    parser.add_argument('-ref-sep', type=str, required=False, help='The separator used in the reference input file (If -ref-file-format is infer, this argument will be inferred as well).')
    parser.add_argument('-target-sep', type=str, required=False, help='The separator used in the target input file (If -target-file-format is infer, this argument will be inferred as well).')
    parser.add_argument('-ref-vac', type=bool, required=False, help='[Used for non-vcf formats] Whether variants appear as columns in the reference file (default: False).', default=False)
    parser.add_argument('-target-vac', type=bool, required=False, help='[Used for non-vcf formats] Whether variants appear as columns in the target file (default: False).', default=False)
    parser.add_argument('-ref-fcai', type=bool, required=False, help='[Used for non-vcf formats] Whether the first column in the reference file is (samples | variants) index (default: False).', default=False)
    parser.add_argument('-target-fcai', type=bool, required=False, help='[Used for non-vcf formats] Whether the first column in the target file is (samples | variants) index (default: False).', default=False)
    parser.add_argument('-ref-file-format', type=str, required=False,
                        help='Reference file format: infer | vcf | csv | tsv. Default is infer.',
                        default="infer",
                        choices=['infer', 'vcf', 'csv', 'tsv'])
    parser.add_argument('-target-file-format', type=str, required=False,
                        help='Target file format: infer | vcf | csv | tsv. Default is infer.',
                        default="infer",
                        choices=['infer', 'vcf', 'csv', 'tsv'])

    ## path args
    parser.add_argument('-save_dir', type=str, required=True, help='the path to save the results and the model.\n'
                                                                   'This path is also used to load a trained model for imputation.')


    ## Chunking args
    parser.add_argument('-wo', type=int, required=False, help='Chunk overlap in terms of SNPs/SVs(default 100)', default=100)
    parser.add_argument('-cs', type=int, required=False, help='Chunk size in terms of SNPs/SVs(default 2000)', default=2000)
    parser.add_argument('-sites-per-model', type=int, required=False, help='Number of SNPs/SVs used per model(default 30000)', default=30000)

    ## Model hyper-params
    parser.add_argument('-na-heads', type=int, required=False, help='Number of attention heads(default 16)', default=16)
    parser.add_argument('-embed-dim', type=int, required=False, help='Embedding dimension size(default 128)', default=128)
    parser.add_argument('-lr', type=float, required=False, help='Learning Rate (default 0.001)', default=0.001)
    parser.add_argument('-batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 4)', default=4)


    args = parser.parse_args()

    phenoIndex = args.pi

if __name__ == '__main__':
    main()