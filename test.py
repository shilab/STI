import argparse
import numpy as np

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ShiLab\'s Imputation model (STI v1.1).')
    parser.add_argument('--restart-training', type=str, required=False,
                        help='Whether to clean previously saved models in target directory and restart the training',
                        choices=['false', 'true', '0', '1'], default='0')
    args = parser.parse_args()
    args.restart_training = str_to_bool(args.restart_training)
    print(args.restart_training)