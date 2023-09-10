# Copyright 2023 by Kaname Kojima, Tohoku University
#
# Licensed under the MIT license
# https://opensource.org/licenses/mit-license.php
from argparse import ArgumentParser
from contextlib import contextmanager
import gzip
import os
import sys


@contextmanager
def reading(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'rt') if ext == '.gz' else open(filename, 'rt')
    try:
        yield fp
    finally:
        fp.close()


def convert_to_vcf(
        haps_file,
        legend_file,
        sample_file,
        chromosome_name,
        output_file):
    sample_name_list = []
    if sample_file is not None:
        with open(sample_file, 'rt') as fp:
            fp.readline()
            fp.readline()
            for line in fp:
                items = line.strip().split()
                sample_name_list.append(items[0])
    with reading(haps_file) as fp:
        items = fp.readline().rstrip().split()
        sample_size = len(items) // 2
        if sample_file is None:
            sample_name_list = [
                'sample_{:d}'.format(i + 1) for i in range(sample_size)]
    assert len(sample_name_list) == sample_size
    with open(output_file, 'wt') as fout, \
         reading(haps_file) as haps_fp, \
         reading(legend_file) as legend_fp:
        fout.write('##fileformat=VCFv4.1\n')
        fout.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')
        fout.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT')
        if sample_size > 0:
            fout.write('\t')
            fout.write('\t'.join(sample_name_list))
        fout.write('\n')
        header_items = legend_fp.readline().rstrip().split()
        try:
            snp_col = header_items.index('id')
            position_col = header_items.index('position')
            a0_col = header_items.index('a0')
            a1_col = header_items.index('a1')
        except ValueError:
            print('Some header not found', file=sys.stderr)
            sys.exit(0)
        GT_list = [None] * sample_size
        for line in haps_fp:
            legend_items = legend_fp.readline().rstrip().split()
            snp_name = legend_items[snp_col]
            ref = legend_items[a0_col]
            alt = legend_items[a1_col]
            position = legend_items[position_col]
            items = line.rstrip().split()
            assert len(items) == 2 * sample_size
            fout.write(
                '{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t.\tPASS\t.\tGT'.format(
                    chromosome_name, position, snp_name, ref, alt))
            if sample_size > 0:
                for i in range(sample_size):
                    allele_num1 = items[2 * i]
                    allele_num2 = items[2 * i + 1]
                    if allele_num1 != '0' and allele_num1 != '1':
                        GT_list[i] = './.'
                    elif allele_num2 != '0' and allele_num2 != '1':
                        GT_list[i] = './.'
                    else:
                        GT_list[i] = '{:s}|{:s}'.format(
                            allele_num1, allele_num2)
                fout.write('\t')
                fout.write('\t'.join(GT_list))
            fout.write('\n')


def main():
    description = 'convert hapslegend to vcf'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--haps', type=str, required=True,
                        dest='haps_file', help='haps file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--sample', type=str, default=None,
                        dest='sample_file', help='sample file')
    parser.add_argument('--chromosome', type=str, required=True,
                        dest='chromosome_name', help='chromosome name')
    parser.add_argument('--output-file', type=str, required=True,
                        dest='output_file', help='output file')
    args = parser.parse_args()

    convert_to_vcf(
        args.haps_file, args.legend_file, args.sample_file,
        args.chromosome_name, args.output_file)


if __name__ == '__main__':
    main()
