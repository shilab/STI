import os, sys, math, re, random, shutil, gzip, argparse, pathlib
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
import json

print("Tensorflow version " + tf.__version__)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
N_REPLICAS = strategy.num_replicas_in_sync
print(f"Num gpus to be used: {N_REPLICAS}")


## Custom Layers
class CrossAttentionLayer(layers.Layer):
    def __init__(self, local_dim, global_dim,
                 start_offset=0, end_offset=0,
                 activation=tf.nn.gelu, dropout_rate=0.1,
                 n_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.num_heads = n_heads
        self.layer_norm00 = layers.LayerNormalization()
        self.layer_norm01 = layers.LayerNormalization()
        self.layer_norm1 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(self.local_dim // 2, activation=self.activation,
                             ),
                layers.Dense(self.local_dim,
                             activation=self.activation,
                             ), ]
        )
        self.add0 = layers.Add()
        self.add1 = layers.Add()
        self.attention = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                   key_dim=self.local_dim)

    def call(self, inputs, training):
        local_repr = self.layer_norm00(inputs[0])
        global_repr = self.layer_norm01(inputs[1])
        query = local_repr[:, self.start_offset:local_repr.shape[1] - self.end_offset, :]
        key = global_repr
        value = global_repr

        # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
        attention_output = self.attention(
            query, key, value
        )
        # Skip connection 1.
        attention_output = self.add0([attention_output, query])

        # Apply layer norm.
        attention_output = self.layer_norm1(attention_output)
        # Apply Feedforward network.
        outputs = self.ffn(attention_output)
        # Skip connection 2.
        outputs = self.add1([outputs, attention_output])
        return outputs


class MaskedTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, attention_range, start_offset=0, end_offset=0,
                 attn_block_repeats=1, activation=tf.nn.gelu, dropout_rate=0.1, use_ffn=True):
        super(MaskedTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.attention_range = attention_range
        self.attn_block_repeats = attn_block_repeats
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_ffn = use_ffn
        self.att0 = [layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=self.embed_dim) for _ in range(attn_block_repeats)]
        if self.use_ffn:
            self.ffn = [tf.keras.Sequential(
                [
                    layers.Dense(self.ff_dim, activation=self.activation,
                                 ),
                    layers.Dense(self.embed_dim,
                                 activation=self.activation,
                                 ), ]
            ) for _ in range(attn_block_repeats)]
        self.layer_norm0 = [layers.LayerNormalization() for _ in range(attn_block_repeats)]
        self.layer_norm1 = [layers.LayerNormalization() for _ in range(attn_block_repeats)]

    def build(self, input_shape):
        assert (self.end_offset >= 0)
        self.feature_size = input_shape[1]
        attention_mask = np.zeros((self.feature_size,
                                   self.feature_size), dtype=bool)
        for i in range(self.start_offset, self.feature_size - self.end_offset):
            attention_indices = np.arange(max(0, i - self.attention_range),
                                          min(self.feature_size, i + self.attention_range))
            attention_mask[i, attention_indices] = True
        self.attention_mask = tf.constant(attention_mask[self.start_offset:self.feature_size - self.end_offset])

    def call(self, inputs, training):

        x = inputs
        for i in range(self.attn_block_repeats - 1):
            x = self.layer_norm0[i](x)
            attn_output = self.att0[i](x, x)
            out1 = x + attn_output
            out1 = self.layer_norm1[i](out1)
            if self.use_ffn:
                ffn_output = self.ffn[i](out1)
                x = out1 + ffn_output
            else:
                x = out1

        x = self.layer_norm0[-1](inputs)
        attn_output = self.att0[-1](x[:, self.start_offset:x.shape[1] - self.end_offset, :], x,
                                    )
        out1 = x[:, self.start_offset:x.shape[1] - self.end_offset, :] + attn_output
        out1 = self.layer_norm1[-1](out1)
        if self.use_ffn:
            ffn_output = self.ffn[-1](out1)
            x = out1 + ffn_output
        else:
            x = out1
        return x


class GenoEmbeddings(layers.Layer):
    def __init__(self, embedding_dim,
                 embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None):
        super(GenoEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)

    def build(self, input_shape):
        # print(input_shape)

        self.num_of_allels = input_shape[-1]
        self.n_snps = input_shape[-2]
        self.position_embedding = layers.Embedding(
            input_dim=self.n_snps, output_dim=self.embedding_dim
        )
        self.embedding = self.add_weight(
            shape=(self.num_of_allels, self.embedding_dim),
            initializer=self.embeddings_initializer,
            trainable=True, name='geno_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False
        )
        self.positions = tf.range(start=0, limit=self.n_snps, delta=1)

    def call(self, inputs):
        self.immediate_result = tf.einsum('ijk,kl->ijl', inputs, self.embedding)
        return self.immediate_result + self.position_embedding(self.positions)


class SelfAttnChunk(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, attention_range,
                 start_offset=0, end_offset=0,
                 attn_block_repeats=1,
                 include_embedding_layer=False):
        super(SelfAttnChunk, self).__init__()
        self.attention_range = attention_range
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn_block_repeats = attn_block_repeats
        self.include_embedding_layer = include_embedding_layer

        self.attention_block = MaskedTransformerBlock(self.embed_dim,
                                                      self.num_heads, self.ff_dim,
                                                      attention_range, start_offset,
                                                      end_offset, attn_block_repeats=1)
        if include_embedding_layer:
            self.embedding = GenoEmbeddings(embed_dim)

    def build(self, input_shape):
        pass

    def call(self, inputs, training):
        if self.include_embedding_layer:
            x = self.embedding(inputs)
        else:
            x = inputs
        x = self.attention_block(x)
        return x


class CrossAttnChunk(layers.Layer):
    def __init__(self, start_offset=0, end_offset=0, n_heads=8):
        super(CrossAttnChunk, self).__init__()
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.n_heads = n_heads

    def build(self, input_shape):
        self.local_dim = input_shape[0][-1]
        self.global_dim = input_shape[1][-1]
        self.attention_block = CrossAttentionLayer(self.local_dim, self.global_dim,
                                                   self.start_offset, self.end_offset,
                                                   n_heads=self.n_heads)
        pass

    def call(self, inputs, training):
        x = inputs
        x = self.attention_block(x)
        return x


class ConvBlock(layers.Layer):
    def __init__(self, embed_dim):
        super(ConvBlock, self).__init__()
        self.embed_dim = embed_dim
        self.const = None
        self.conv000 = layers.Conv1D(embed_dim, 3, padding='same', activation=tf.nn.gelu,
                                     kernel_constraint=self.const,
                                     )
        self.conv010 = layers.Conv1D(embed_dim, 5, padding='same', activation=tf.nn.gelu,
                                     kernel_constraint=self.const,
                                     )
        self.conv011 = layers.Conv1D(embed_dim, 7, padding='same', activation=tf.nn.gelu,
                                     kernel_constraint=self.const,
                                     )

        self.conv020 = layers.Conv1D(embed_dim, 7, padding='same', activation=tf.nn.gelu,
                                     kernel_constraint=self.const,
                                     )
        self.conv021 = layers.Conv1D(embed_dim, 15, padding='same', activation=tf.nn.gelu,
                                     kernel_constraint=self.const,
                                     )
        self.add = layers.Add()

        self.conv100 = layers.Conv1D(embed_dim, 3, padding='same',
                                     activation=tf.nn.gelu,
                                     kernel_constraint=self.const, )
        self.bn0 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.dw_conv = layers.Conv1D(embed_dim, 1, padding='same')
        self.activation = layers.Activation(tf.nn.gelu)

    def call(self, inputs, training):
        # Could add skip connection here?
        xa = self.conv000(inputs)

        xb = self.conv010(xa)
        xb = self.conv011(xb)

        xc = self.conv020(xa)
        xc = self.conv021(xc)

        xa = self.add([xb, xc])
        xa = self.conv100(xa)
        xa = self.bn0(xa)
        xa = self.dw_conv(xa)
        xa = self.bn1(xa)
        xa = self.activation(xa)
        return xa


def chunk_module(input_len, embed_dim, num_heads, attention_range,
                 start_offset=0, end_offset=0):
    projection_dim = embed_dim
    inputs = layers.Input(shape=(input_len, embed_dim))
    xa = inputs
    xa0 = SelfAttnChunk(projection_dim, num_heads, projection_dim // 2, attention_range,
                        start_offset, end_offset, 1, include_embedding_layer=False)(xa)

    xa = ConvBlock(projection_dim)(xa0)
    xa_skip = ConvBlock(projection_dim)(xa)

    xa = layers.Dense(projection_dim, activation=tf.nn.gelu)(xa)
    xa = ConvBlock(projection_dim)(xa)
    xa = CrossAttnChunk(0, 0)([xa, xa0])
    xa = layers.Dropout(0.25)(xa)
    xa = ConvBlock(projection_dim)(xa)

    xa = layers.Concatenate(axis=-1)([xa_skip, xa])

    model = keras.Model(inputs=inputs, outputs=xa)
    return model


## STI Model
class SplitTransformer(keras.Model):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 offset_before=0,
                 offset_after=0,
                 chunk_size=2000,
                 activation=tf.nn.gelu,
                 dropout_rate=0.25,
                 attn_block_repeats=1,
                 attention_range=100,
                 in_channel=2):
        super(SplitTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attn_block_repeats = attn_block_repeats
        self.attention_range = attention_range
        self.offset_before = offset_before
        self.offset_after = offset_after
        self.in_channel = in_channel

    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.chunk_starts = list(range(0, input_shape[1], self.chunk_size))
        self.chunk_ends = []
        for cs in self.chunk_starts:
            self.chunk_ends.append(min(cs + self.chunk_size, input_shape[1]))
        self.mask_starts = [max(0, cs - self.attention_range) for cs in self.chunk_starts]
        self.mask_ends = [min(ce + self.attention_range, input_shape[1]) for ce in self.chunk_ends]
        self.chunkers = [chunk_module(self.mask_ends[i] - self.mask_starts[i],
                                      self.embed_dim, self.num_heads,
                                      self.attention_range,
                                      start_offset=cs - self.mask_starts[i],
                                      end_offset=self.mask_ends[i] - self.chunk_ends[i]
                                      ) for i, cs in enumerate(self.chunk_starts)]

        self.concat_layer = layers.Concatenate(axis=-2)
        self.embedding = GenoEmbeddings(self.embed_dim)
        self.slice_layer = layers.Lambda(lambda x: x[:, self.offset_before:self.seq_len - self.offset_after],
                                         name="output_slicer")
        self.after_concat_layer = layers.Conv1D(self.embed_dim // 2, 5, padding='same', activation=tf.nn.gelu)
        self.last_conv = layers.Conv1D(self.in_channel - 1, 5, padding='same', activation=tf.nn.softmax)
        super(SplitTransformer, self).build(input_shape)

    def call(self, inputs):
        x = self.embedding(inputs)
        chunks = [self.chunkers[i](x[:,
                                   self.mask_starts[i]:self.mask_ends[i]]) for i, chunker \
                  in enumerate(self.chunkers)]
        x = self.concat_layer(chunks)
        x = self.after_concat_layer(x)
        x = self.last_conv(x)
        x = self.slice_layer(x)
        return x


## Loss
class ImputationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        cat_loss = loss_obj(y_true, y_pred)

        loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        kl_loss = loss_obj(y_true, y_pred)

        return cat_loss + kl_loss


## Model creation
def create_model(args):
    model = SplitTransformer(embed_dim=args["embedding_dim"],
                             num_heads=args["num_heads"],
                             chunk_size=args["chunk_size"],
                             activation=tf.nn.gelu,
                             attention_range=args["chunk_overlap"],
                             in_channel=args["in_channel"],
                             offset_before=args["offset_before"],
                             offset_after=args["offset_after"])
    optimizer = tfa.optimizers.LAMB(learning_rate=args["lr"])
    model.compile(optimizer, loss=ImputationLoss(), metrics=tf.keras.metrics.CategoricalAccuracy())
    return model


def create_callbacks(metric="val_loss", save_path="."):
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric,
        mode='auto',
        factor=0.5,
        patience=3,
        verbose=0
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        mode='auto',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path,
        monitor=metric,
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
    )

    callbacks = [
        reducelr,
        earlystop,
        checkpoint
    ]

    return callbacks


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
        self.map_values_1_vec = np.vectorize(self.__map_hap_2_ind_parent_1)
        self.map_values_2_vec = np.vectorize(self.__map_hap_2_ind_parent_2)
        self.map_haps_to_vec = np.vectorize(self.__map_haps_2_ind)
        self.delimiter_dictionary = {"vcf": "\t", "csv": ",", "tsv": "\t", "infer": "\t"}
        self.ref_file_extension = "vcf"
        self.test_file_extension = "vcf"
        self.target_is_phased = True
        ## Idea: keep track of possible alleles in each variant, and filter the predictions based on that

    def __read_csv(self, file_path, is_vcf=False, is_reference=False, separator="\t", first_column_is_index=True,
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
        return df

    def __find_file_extension(self, file_path, file_format, delimiter):
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
        self.ref_file_extension, self.ref_separator = self.__find_file_extension(file_path, file_format, delimiter)
        if file_format == "infer":
            print(f"Ref file format is {self.ref_file_extension} and Ref file sep is {self.ref_separator}.")

        self.reference_panel = self.__read_csv(file_path, is_reference=True, is_vcf=False, separator=self.ref_separator,
                                               first_column_is_index=first_column_is_index,
                                               comments=comments) if self.ref_file_extension != 'vcf' else self.__read_csv(
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
        print(
            f"{self.reference_panel.shape[1] - (self.ref_sample_value_index - 1)} {'haploid' if self.ref_is_hap else 'diploid'} samples with {self.VARIANT_COUNT} variants found!")

        self.is_phased = target_is_gonna_be_phased_or_haps and (self.ref_is_phased or self.ref_is_hap)

        original_allele_sep = "|" if self.ref_is_phased or self.ref_is_hap else "/"
        final_allele_sep = "|" if self.is_phased else "/"

        def get_diploid_allels(genotype_vals):
            allele_set = set()
            for genotype_val in genotype_vals:
                v1, v2 = genotype_val.split(final_allele_sep)
                allele_set.update([v1, v2])
            return np.array(list(allele_set))

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
        self.alleles = get_diploid_allels(self.genotype_vals) if not self.ref_is_hap else self.genotype_vals
        self.allele_count = len(self.alleles)
        self.MISSING_VALUE = self.allele_count if self.is_phased else len(self.genotype_vals)

        if self.is_phased:
            self.hap_map = {str(v): i for i, v in enumerate(self.alleles)}
            self.hap_map.update({".": self.MISSING_VALUE})
            self.r_hap_map = {i: k for k, i in self.hap_map.items()}
            self.map_preds_2_allele = np.vectorize(lambda x: self.r_hap_map[x])
        else:
            unphased_missing_genotype = "./."
            self.replacement_dict = {g: i for i, g in enumerate(self.genotype_vals)}
            self.replacement_dict[unphased_missing_genotype] = self.MISSING_VALUE
            self.reverse_replacement_dict = {v: k for k, v in enumerate(self.replacement_dict)}

        self.SEQ_DEPTH = self.allele_count + 1 if self.is_phased else len(self.genotype_vals)
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

        self.target_file_extension, separator = self.__find_file_extension(file_path, file_format, delimiter)

        test_df = self.__read_csv(file_path, is_reference=False, is_vcf=False, separator=separator,
                                  first_column_is_index=first_column_is_index,
                                  comments=comments) if self.ref_file_extension != 'vcf' else self.__read_csv(file_path,
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

    def __map_hap_2_ind_parent_1(self, x) -> int:
        return self.hap_map[x.split('|')[0]]

    def __map_hap_2_ind_parent_2(self, x) -> int:
        return self.hap_map[x.split('|')[1]]

    def __map_haps_2_ind(self, x) -> int:
        return self.hap_map[x]

    def __diploids_to_hap_vecs(self, data: pd.DataFrame) -> np.ndarray:
        _x = np.empty((data.shape[1] * 2, data.shape[0]), dtype=np.int32)
        _x[0::2] = self.map_values_1_vec(data.values.T)
        _x[1::2] = self.map_values_2_vec(data.values.T)
        return _x

    def __get_forward_data(self, data: pd.DataFrame) -> np.ndarray:
        if self.is_phased:
            is_haps = "|" not in data.iloc[0, 0]
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

    def __convert_genotypes_to_vcf(self, genotypes, pred_format="GT:DS:GP"):
        n_samples, n_variants = genotypes.shape
        new_vcf = self.target_set.copy()
        new_vcf.iloc[:n_variants, 9:] = genotypes.T
        new_vcf["FORMAT"] = pred_format
        new_vcf["QUAL"] = "."
        new_vcf["FILTER"] = "."
        new_vcf["INFO"] = "IMPUTED"
        return new_vcf

    def __convert_hap_probs_to_diploid_genotypes(self, allele_probs) -> np.ndarray:
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

    def __convert_hap_probs_to_hap_genotypes(self, allele_probs) -> np.ndarray:
        allele_probs_normalized = softmax(allele_probs, axis=-1)
        return np.argmax(allele_probs_normalized, axis=1).astype(str)

    def __convert_unphased_probs_to_genotypes(self, allele_probs) -> np.ndarray:
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
        :param predictions: The path to numpy array stored on disk or numpy array of shape (n_samples, n_variants, n_alleles)
        :return: numpy array of the same shape, with genotype calls, e.g., "0/1"
        """
        if isinstance(predictions, str):
            preds = np.load(predictions)
        else:
            preds = predictions

        target_df = self.target_set.copy()
        if not self.is_phased:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_unphased_probs_to_genotypes(
                preds).T
        elif self.target_is_hap:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_hap_probs_to_hap_genotypes(
                preds).T
        else:
            target_df[
                target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_hap_probs_to_diploid_genotypes(
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


@tf.function()
def add_attention_mask(X_sample, y_sample, depth, mr):
    mask_size = tf.cast(X_sample.shape[0] * mr, dtype=tf.int32)
    mask_idx = tf.reshape(tf.random.shuffle(tf.range(X_sample.shape[0]))[:mask_size], (-1, 1))
    updates = tf.math.add(tf.zeros(shape=(mask_idx.shape[0]), dtype=tf.int32), depth - 1)
    X_masked = tf.tensor_scatter_nd_update(X_sample, mask_idx, updates)

    return tf.one_hot(X_masked, depth), tf.one_hot(y_sample, depth - 1)


@tf.function()
def onehot_encode(X_sample, depth):
    return tf.one_hot(X_sample, depth)


def get_training_dataset(x, batch_size, depth,
                         offset_before=0, offset_after=0,
                         training=True, masking_rate=0.8):
    AUTO = tf.data.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices((x, x[:, offset_before:x.shape[1] - offset_after]))
    # # Add Attention Mask

    if training:
        dataset = dataset.shuffle(x.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.repeat()

    # Add Attention Mask
    dataset = dataset.map(lambda xx, yy: add_attention_mask(xx, yy, depth, masking_rate),
                          num_parallel_calls=AUTO, deterministic=False)

    # Prefetech to not map the whole dataset
    dataset = dataset.prefetch(AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=AUTO)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)  # use this as input for your model
    dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset


def get_test_dataset(x, batch_size, depth):
    AUTO = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((x))
    # one-hot encode
    dataset = dataset.map(lambda xx: onehot_encode(xx, depth), num_parallel_calls=AUTO, deterministic=True)

    # Prefetech to not map the whole dataset
    dataset = dataset.prefetch(AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=False, num_parallel_calls=AUTO)

    return dataset


def create_directories(save_dir,
                       models_dir="models",
                       outputs="out") -> None:
    for dir in [save_dir,
                f"{save_dir}/{models_dir}",
                f"{save_dir}/{outputs}"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    pass


def clear_dir(path) -> None:
    # credit: https://stackoverflow.com/a/72982576/4260559
    for entry in os.scandir(path):
        if entry.is_dir():
            clear_dir(entry)
        else:
            os.remove(entry)
    os.rmdir(path)  # if you just want to delete the dir content but not the dir itself, remove this line


def load_chunk_info(save_dir, break_points):
    chunk_info = {ww: False for ww in list(range(len(break_points) - 1))}
    if os.path.isfile(f"{save_dir}/models/chunks_info.json"):
        with open(f"{save_dir}/models/chunks_info.json", 'r') as f:
            loaded_chunks_info = json.load(f)
            if isinstance(loaded_chunks_info, dict) and len(loaded_chunks_info) == len(chunk_info):
                print("Resuming the training...")
                chunk_info = {int(k): v for k, v in loaded_chunks_info.items()}
    return chunk_info


def save_chunk_status(save_dir, chunk_info) -> None:
    with open(f"{save_dir}/models/chunks_info.json", "w") as outfile:
        json.dump(chunk_info, outfile)


def train_the_model(args) -> None:
    if args.val_frac <= 0 or args.val_frac >= 1:
        raise args.ArgumentError(message="Validation fraction should be a positive value in range of (0, 1)")
    if args.restart_training:
        clear_dir(args.save_dir)

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size_per_gpu * N_REPLICAS

    create_directories(args.save_dir)
    with open(f"{args.save_dir}/commandline_args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    dr = DataReader()
    dr.assign_training_set(file_path=args.ref,
                           target_is_gonna_be_phased_or_haps=args.tihp,
                           variants_as_columns=args.ref_vac,
                           delimiter=args.ref_sep,
                           file_format=args.ref_file_format,
                           first_column_is_index=args.ref_fcai,
                           comments=args.ref_comment)

    x_train_indices, x_valid_indices = train_test_split(range(dr.get_ref_set(0, 1).shape[0]),
                                                        test_size=args.val_frac,
                                                        random_state=args.random_seed,
                                                        shuffle=True, )
    steps_per_epoch = len(x_train_indices) // BATCH_SIZE
    validation_steps = len(x_valid_indices) // BATCH_SIZE

    break_points = list(np.arange(0, dr.VARIANT_COUNT, args.sites_per_model)) + [dr.VARIANT_COUNT]
    chunks_done = load_chunk_info(args.save_dir, break_points)
    for w in range(len(break_points) - 1):
        if chunks_done[w]:
            print(f"Skipping part {w + 1} out of {len(break_points) - 1} due to previous training.")
            continue
        print(f"Doing part {w + 1} out of {len(break_points) - 1}")
        final_start_pos = max(0, break_points[w] - 2 * args.co)
        final_end_pos = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * args.co)
        offset_before = break_points[w] - final_start_pos
        offset_after = final_end_pos - break_points[w + 1]
        ref_set = dr.get_ref_set(final_start_pos, final_end_pos).astype(np.int32)
        print(f"Data shape: {ref_set.shape}")
        train_dataset = get_training_dataset(ref_set[x_train_indices], BATCH_SIZE,
                                             depth=dr.SEQ_DEPTH,
                                             offset_before=offset_before,
                                             offset_after=offset_after,
                                             masking_rate=args.mr)
        valid_dataset = get_training_dataset(ref_set[x_valid_indices], BATCH_SIZE,
                                             depth=dr.SEQ_DEPTH,
                                             offset_before=offset_before,
                                             offset_after=offset_after, training=False,
                                             masking_rate=args.mr)
        del ref_set
        K.clear_session()
        callbacks = create_callbacks(save_path=f"{args.save_dir}/models/w_{w}")
        model_args = {
            "embedding_dim": args.embed_dim,
            "num_heads": args.na_heads,
            "chunk_size": args.cs,
            "chunk_overlap": args.co,
            "in_channel": dr.SEQ_DEPTH,
            "offset_before": offset_before,
            "offset_after": offset_after,
            "lr": args.lr
        }
        with strategy.scope():
            model = create_model(model_args)
            history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                                epochs=NUM_EPOCHS,
                                validation_data=valid_dataset,
                                validation_steps=validation_steps,
                                callbacks=callbacks, verbose=1)
            chunks_done[w] = True
            save_chunk_status(args.save_dir, chunks_done)


def impute_the_target(args):
    if args.target is None:
        raise argparse.ArgumentError(
            message="Target file missing for imputation. use -target to specify a target file.")

    dr = DataReader()
    dr.assign_training_set(file_path=args.ref,
                           target_is_gonna_be_phased_or_haps=args.tihp,
                           variants_as_columns=args.ref_vac,
                           delimiter=args.ref_sep,
                           file_format=args.ref_file_format,
                           first_column_is_index=args.ref_fcai,
                           comments=args.ref_comment)
    dr.assign_test_set(file_path=args.target,
                       variants_as_columns=args.target_vac,
                       delimiter=args.target_sep,
                       file_format=args.target_file_format,
                       first_column_is_index=args.target_fcai,
                       comments=args.target_comment)
    pass


def str_to_bool(s):
    # Define accepted string values for True and False
    true_values = ['true', '1']
    false_values = ['false', '0']

    # Convert the input string to lowercase for case-insensitive comparison
    lower_s = s.lower()

    # Check if the input string is in the list of true or false values
    if lower_s in true_values:
        return True
    elif lower_s in false_values:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}. Accepted values are 'true', 'false', '0', '1'.")


def main():
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
    parser.add_argument('--mode', type=str, help='Operation mode: impute | train (default=train)',
                        choices=['impute', 'train'], default='train')
    parser.add_argument('--restart-training', type=str, required=False,
                        help='Whether to clean previously saved models in target directory and restart the training',
                        choices=['false', 'true', '0', '1'], default='0')
    ## Input args
    parser.add_argument('--ref', type=str, required=True, help='Reference file path.')
    parser.add_argument('--target', type=str, required=False,
                        help='Target file path. Must be provided in "impute" mode.')
    parser.add_argument('--tihp', type=str, required=True,
                        help='Whether the target is going to be haps or phased.',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--ref-comment', type=str, required=False,
                        help='The character(s) used to indicate comment lines in the reference file (default="\\t").',
                        default="##")
    parser.add_argument('--target-comment', type=str, required=False,
                        help='The character(s) used to indicate comment lines in the target file (default="\\t").',
                        default="\t")
    parser.add_argument('--ref-sep', type=str, required=False,
                        help='The separator used in the reference input file (If -ref-file-format is infer, '
                             'this argument will be inferred as well).')
    parser.add_argument('--target-sep', type=str, required=False,
                        help='The separator used in the target input file (If -target-file-format is infer, '
                             'this argument will be inferred as well).')
    parser.add_argument('--ref-vac', type=str, required=False,
                        help='[Used for non-vcf formats] Whether variants appear as columns in the reference file ('
                             'default: false).',
                        default='0',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--target-vac', type=str, required=False,
                        help='[Used for non-vcf formats] Whether variants appear as columns in the target file ('
                             'default: false).',
                        default='0',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--ref-fcai', type=str, required=False,
                        help='[Used for non-vcf formats] Whether the first column in the reference file is (samples | '
                             'variants) index (default: false).',
                        default='0',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--target-fcai', type=str, required=False,
                        help='[Used for non-vcf formats] Whether the first column in the target file is (samples | '
                             'variants) index (default: False).',
                        default='0',
                        choices=['false', 'true', '0', '1'])
    parser.add_argument('--ref-file-format', type=str, required=False,
                        help='Reference file format: infer | vcf | csv | tsv. Default is infer.',
                        default="infer",
                        choices=['infer', 'vcf', 'csv', 'tsv'])
    parser.add_argument('--target-file-format', type=str, required=False,
                        help='Target file format: infer | vcf | csv | tsv. Default is infer.',
                        default="infer",
                        choices=['infer', 'vcf', 'csv', 'tsv'])

    ## save args
    parser.add_argument('--save-dir', type=str, required=True, help='the path to save the results and the model.\n'
                                                                    'This path is also used to load a trained model for imputation.')

    ## Chunking args
    parser.add_argument('--co', type=int, required=False, help='Chunk overlap in terms of SNPs/SVs(default 100)',
                        default=100)
    parser.add_argument('--cs', type=int, required=False, help='Chunk size in terms of SNPs/SVs(default 2000)',
                        default=2000)
    parser.add_argument('--sites-per-model', type=int, required=False,
                        help='Number of SNPs/SVs used per model(default 30000)', default=30000)

    ## Model (hyper-)params
    parser.add_argument('--mr', type=float, required=False, help='Masking rate(default 0.8)', default=0.8)
    parser.add_argument('--val-frac', type=float, required=False,
                        help='Fraction of reference samples to be used for validation (default=0.1).', default=0.1)
    parser.add_argument('--random-seed', type=int, required=False,
                        help='Random seed used for splitting the data into training and validation sets (default 2022).',
                        default=2022)
    parser.add_argument('--epochs', type=int, required=False, help='Maximum number of epochs (default 1000)',
                        default=1000)
    parser.add_argument('--na-heads', type=int, required=False, help='Number of attention heads (default 16)',
                        default=16)
    parser.add_argument('--embed-dim', type=int, required=False, help='Embedding dimension size (default 128)',
                        default=128)
    parser.add_argument('--lr', type=float, required=False, help='Learning Rate (default 0.001)', default=0.001)
    parser.add_argument('--batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 4)',
                        default=4)

    args = parser.parse_args()

    # Convert to boolean arguments
    args.restart_training = str_to_bool(args.restart_training)
    args.tihp = str_to_bool(args.tihp)
    args.ref_vac = str_to_bool(args.ref_vac)
    args.target_vac = str_to_bool(args.target_vac)
    args.ref_fcai = str_to_bool(args.ref_fcai)
    args.target_fcai = str_to_bool(args.target_fcai)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    print("Save directory will be:", args.save_dir)

    if args.mode == 'train':
        train_the_model(args)
    else:
        impute_the_target(args)


if __name__ == '__main__':
    main()
