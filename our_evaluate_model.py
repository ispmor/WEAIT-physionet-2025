#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d data -o outputs -s scores.csv
#
# where 'data' is a folder containing files with the reference signals and labels for the data, 'outputs' is a folder containing
# files with the outputs from your models, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each data or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import argparse
import numpy as np
import os
import os.path
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(data_folder, output_folder):
    # Find the records.
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No records found.')

    labels = np.zeros(num_records)
    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)

    # Load the labels and model outputs.
    for i, record in enumerate(records):
        label_filename = os.path.join(data_folder, record)
        label = load_label(label_filename)

        output_filename = os.path.join(output_folder, record + '.txt')
        output = load_text(output_filename)
        binary_output = get_label(output, allow_missing=True)
        probability_output = get_probability(output, allow_missing=True)

        # Missing model outputs are interpreted as zero for the binary and probability outputs.
        labels[i] = label
        if not is_nan(binary_output):
            binary_outputs[i] = binary_output
        else:
            binary_outputs[i] = 0
        if not is_nan(probability_output):
            probability_outputs[i] = probability_output
        else:
            probability_outputs[i] = 0

    # Evaluate the model outputs.
    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    return challenge_score, auroc, auprc, accuracy, f_measure


def print_confusion_matrix(labels, outputs, max_fraction_positive=0.05):
    """
    Compute and print the confusion matrix (TP, FP, FN, TN) at the highest
    threshold that yields at most max_fraction_positive * N positive predictions.
    """
    # --- copy of your threshold‐selection logic ---
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    num_instances = len(labels)
    max_pos = int(max_fraction_positive * num_instances)

    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]

    idx = np.argsort(outputs)[::-1]

    # initialize
    tp = np.zeros(len(thresholds), dtype=int)
    fp = np.zeros(len(thresholds), dtype=int)
    fn = np.zeros(len(thresholds), dtype=int)
    tn = np.zeros(len(thresholds), dtype=int)

    tp[0] = 0
    fp[0] = 0
    fn[0] = int((labels == 1).sum())
    tn[0] = int((labels == 0).sum())

    i = 0
    for j in range(1, len(thresholds)):
        tp[j] = tp[j-1]
        fp[j] = fp[j-1]
        fn[j] = fn[j-1]
        tn[j] = tn[j-1]

        while i < num_instances and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]] == 1:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # find the “best” threshold index k
    k = len(thresholds) - 1
    for j in range(1, len(thresholds)):
        if tp[j] + fp[j] > max_pos:
            k = j - 1
            break

    # now print
    chosen_thr = thresholds[k]
    print(f"Chosen threshold = {chosen_thr!r}")
    print()
    print("Confusion matrix (rows = actual, cols = predicted)")
    print(f"{'':10s}Pred +    Pred -")
    print(f"Actual + {tp[k]:8d} {fn[k]:8d}")
    print(f"Actual - {fp[k]:8d} {tn[k]:8d}")


# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    challenge_score, auroc, auprc, accuracy, f_measure = evaluate_model(args.data_folder, args.output_folder)

    output_string = \
        f'Challenge score: {challenge_score:.3f}\n' + \
        f'AUROC: {auroc:.3f}\n' \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' \
        f'F-measure: {f_measure:.3f}\n'

    # Output the scores to screen and/or a file.
    if args.score_file:
        save_text(args.score_file, output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
