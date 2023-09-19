import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ShiLab\'s Imputation model (STI v1.1).')

    ## Function mode
    parser.add_argument('--mode', type=str, help='Operation mode: impute | train (default=train)',
                        choices=['impute', 'train'], default='train')
    args = parser.parse_args()
    print(args.mode)