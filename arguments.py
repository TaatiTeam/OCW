import argparse

"""
arguments for the OC Dataset
"""

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model-name', type=str, default='elmo', help='the model name either elmo or HF model')
    parse.add_argument('--dataset_path', type=str, default='./', help='Path to the OC dataset')
    parse.add_argument('--predictions_path', type=str, default='./predictions/', help='Path to predictions folder')
    parse.add_argument('--results_path', type=str, default='./results/', help='Path to results folder')
    parse.add_argument('--seed', type=int, default=42, help='the random seeds')
    parse.add_argument('--prediction_file', type=str, default='./predictions/elmo_predictions.json'
                       , help='Path to predictions file')

    args = parse.parse_args()

    return args