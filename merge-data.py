#!/usr/bin/env python3
import argparse
import pandas
import os.path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Merge to genotype tables to one.')
    parser.add_argument('-1', '--first-file', help='1st file')
    parser.add_argument('-2', '--second-file', help='2nd file')
    parser.add_argument('-o', '--output', help='Output file')

    return parser.parse_args()


def main():
    args = parse_args()
    df_1 = pandas.read_csv(args.first_file, sep='\t')
    df_2 = pandas.read_csv(args.second_file, sep='\t')
    df = pandas.merge(df_1, df_2)
    df.to_csv(args.output, index=False, sep='\t')


if __name__ == '__main__':
    main()