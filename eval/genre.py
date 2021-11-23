#!/usr/bin/python3

import argparse, itertools, logging, os, sys

from collections import defaultdict, Counter, OrderedDict

# local imports
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from data.utils import *


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Evaluate Domain Assignments')
    arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
    arg_parser.add_argument('exp_path', help='path to experiment directory')
    arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
    arg_parser.add_argument('-sn', '--split_name', help='name of split to use for clustering (default: None - all splits)')
    return arg_parser.parse_args()


def get_overlap_error(metrics1, metrics2):
    genres1, genres2 = metrics1['genres'], metrics2['genres']
    # get overlapping domains
    genre_overlap = set(genres1) & set(genres2)

    # calculate distributions
    dist1 = metrics1['counts'] / np.sum(metrics1['counts'])
    dist2 = metrics2['counts'] / np.sum(metrics2['counts'])

    # get Bhattacharyya coefficient
    overlap = np.sum(np.sqrt(dist1 * dist2))

    # calculate expected distributions
    # (equiprobable across present domains, e.g. [.5 .5 0] and [0 .5 .5] for 3 domains with 1 overlap)
    exp_dist1, exp_dist2 = np.zeros_like(dist1), np.zeros_like(dist2)  # (num_genres, )
    labels1 = np.arange(len(genres1))  # 0: genre0, 1: genre1, ...
    labels2 = np.arange(
        (len(genres1) - len(genre_overlap)),
        (len(genres1) - len(genre_overlap) + len(genres2)))  # from not-shared -> end
    exp_dist1[labels1] = 1. / len(genres1)  # 1/num_genres1 (equiprobable)
    exp_dist2[labels2] = 1. / len(genres2)  # 1/num_genres2 (equiprobable)
    # get expected Bhattacharyya coefficient
    exp_overlap = np.sum(np.sqrt(exp_dist1 * exp_dist2))

    error = np.abs(exp_overlap - overlap)
    return error


def main():
    args = parse_arguments()

    # check if experiment dir exists
    if not os.path.exists(args.exp_path):
        print(f"[Error] Could not find experiment directory '{args.exp_path}'. Exiting.")
        return

    # setup logging
    setup_logging(os.path.join(args.exp_path, 'eval.log'))

    # load data split definition (if supplied)
    relevant_idcs = None
    if args.split:
        with open(args.split, 'rb') as fp:
            split_idcs = pickle.load(fp)
        # create set of relevant indices from named split
        if args.split_name:
            relevant_idcs = set(split_idcs[args.split_name])
        # create set of relevant indices from all splits
        else:
            relevant_idcs = set().union(*[split_idcs[name] for name in split_idcs])
        logging.info(f"Loaded {len(relevant_idcs)} relevant data from provided splits.")

    # load predictions
    with open(os.path.join(args.exp_path, 'All.pkl'), 'rb') as fp:
        predictions = pickle.load(fp)
    pred_distribution = predictions['domain_dist'] # (num_sents, num_genres)
    pred_names = predictions.get('domains', [f'cluster-{idx}' for idx in range(pred_distribution.shape[1])])
    logging.info(f"Loaded predictions for {pred_distribution.shape[0]} instances across {len(pred_names)} labels ({', '.join([pn[:50] for pn in pred_names])}).")

    # load UD
    ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
    genres = ud.get_domains()
    logging.info(f"Loaded {ud} with {len(genres)} genres ({', '.join(genres)}).")

    # aggregate metrics across UD
    cur_file_idx = 0
    num_files = len(ud.get_treebanks())
    logging.info(f"Aggregating metrics across {num_files} treebank files:")
    # initialize aggregation variables
    aggregate_metrics = defaultdict(dict)
    tbf_keys = []
    # iterate over sentences per treebank file
    for tbf, sentences in ud.get_sentences_by_file():
        # gather global indices of current treebank file
        tbf_idcs = [s.idx for s in sentences]
        # gather gold genres of current treebank file
        tbf_genres = ud.get_domains_of_index(sentences[0].idx)
        # gather other metadata
        tbf_language = ud.get_language_of_index(sentences[0].idx)
        tbf_treebank = ud.get_treebank_name_of_index(sentences[0].idx)

        # reduce indices to relevant ones
        if relevant_idcs is None:
            tbf_rel_idcs = tbf_idcs
        else:
            tbf_rel_idcs = set(tbf_idcs) & relevant_idcs
            # skip if not relevant to evaluation
            if len(tbf_rel_idcs) < 1:
                logging.info(f"[{cur_file_idx+1}/{num_files}] No relevant data in {tbf}. Skipped.")
                cur_file_idx += 1
                continue
        tbf_rel_idcs = sorted(tbf_rel_idcs)

        # calculate predicted distribution of current file
        tbf_predictions = pred_distribution[tbf_rel_idcs, :] # (num_rel, num_genres)
        tbf_labels = np.argmax(tbf_predictions, axis=1) # (num_rel,)
        # gather overall domain distribution (num_rel, num_genres)
        tbf_label_counts = np.zeros((tbf_predictions.shape[0], tbf_predictions.shape[1]))
        tbf_label_counts[np.arange(tbf_predictions.shape[0]), tbf_labels] = 1
        tbf_label_counts = np.sum(tbf_label_counts, axis=0) # (num_genres,)
        tbf_label_distribution = tbf_label_counts / np.sum(tbf_label_counts)
        tbf_max_label = tbf_label_distribution.argmax() # get predicted genre with maximum number of assignments

        # aggregate metrics
        tbf_keys.append(f'UD/{tbf_language}/{tbf_treebank}/{tbf}')
        for agg_key in [
            'UD',
            f'UD/{tbf_language}',
            f'UD/{tbf_language}/{tbf_treebank}',
            f'UD/{tbf_language}/{tbf_treebank}/{tbf}'
        ]:
            # aggregate gold genres
            if 'genres' not in aggregate_metrics[agg_key]:
                aggregate_metrics[agg_key]['genres'] = set()
            aggregate_metrics[agg_key]['genres'] |= set(tbf_genres)

            # aggregate counts
            if 'counts' not in aggregate_metrics[agg_key]:
                aggregate_metrics[agg_key]['counts'] = np.zeros_like(tbf_label_counts)
            aggregate_metrics[agg_key]['counts'] += tbf_label_counts

            # aggregate purity and assigned label (single-genre only)
            if len(tbf_genres) == 1:
                # aggregate purity
                if 'purity' not in aggregate_metrics[agg_key]:
                    aggregate_metrics[agg_key]['purity'] = []
                tbf_purity = tbf_label_distribution[tbf_max_label]
                aggregate_metrics[agg_key]['purity'].append(tbf_purity)

                # aggregate agreement
                if 'agreement' not in aggregate_metrics[agg_key]:
                    aggregate_metrics[agg_key]['agreement'] = defaultdict(list)
                aggregate_metrics[agg_key]['agreement'][tbf_genres[0]].append(tbf_max_label)

            # aggregate number of in-genre classifications (classification only)
            if set(pred_names) <= set(genres):
                if 'correct' not in aggregate_metrics[agg_key]:
                    aggregate_metrics[agg_key]['correct'] = 0
                positive_masks = np.array([tbf_labels == genres.index(tbfg) for tbfg in tbf_genres])
                tbf_num_correct = np.sum(np.logical_or.reduce(positive_masks))
                aggregate_metrics[agg_key]['correct'] += tbf_num_correct

        logging.info(f"[{cur_file_idx+1}/{num_files}] Computed metrics on {len(tbf_rel_idcs)}/{len(tbf_idcs)} instances from '{tbf}'.")
        cur_file_idx += 1

    # calculate overlap errors (across all treebank file combinations)
    logging.info(f"Calculating overlap errors between {(len(tbf_keys)**2)//2 - (len(tbf_keys)//2)} treebank file combinations...")
    tbf_pairs = itertools.combinations(tbf_keys, 2)
    for tbf_key1, tbf_key2 in tbf_pairs:
        _, tbf_language1, tbf_treebank1, _ = tbf_key1.split('/')
        _, tbf_language2, tbf_treebank2, _ = tbf_key2.split('/')
        # calculate error for file pair
        cur_overlap_error = get_overlap_error(aggregate_metrics[tbf_key1], aggregate_metrics[tbf_key2])
        # gather keys for which to aggregate the error
        agg_keys = ['UD']
        if tbf_language1 == tbf_language2:
            agg_keys.append(f'UD/{tbf_language1}')
            if tbf_treebank1 == tbf_treebank2:
                agg_keys.append(f'UD/{tbf_language1}/{tbf_treebank1}')
        # aggregate error across relevant levels
        for agg_key in agg_keys:
            if 'overlap' not in aggregate_metrics[agg_key]:
                aggregate_metrics[agg_key]['overlap'] = []
            aggregate_metrics[agg_key]['overlap'].append(cur_overlap_error)

    # print aggregate metrics
    logging.info('-'*10)
    for agg_key in sorted(aggregate_metrics):
        # gather metrics
        cur_genres = aggregate_metrics[agg_key]['genres']
        cur_counts = aggregate_metrics[agg_key]['counts']
        cur_num_instances = np.sum(cur_counts)
        cur_distribution = cur_counts / cur_num_instances

        # print predicted distribution
        logging.info(f"{agg_key} (n={int(cur_num_instances)}):")
        logging.info(f"  Genres (n={len(cur_genres)}): {', '.join(sorted(cur_genres))}")
        for label_idx, label in enumerate(pred_names):
            logging.info(
                f"    {'[!] ' if pred_names[label_idx] in cur_genres else ''}{pred_names[label_idx][:50]}: "
                f"{cur_distribution[label_idx] * 100:.4f}% (n={int(cur_counts[label_idx])})"
            )

        # print purity of single-genre files
        if 'purity' in aggregate_metrics[agg_key]:
            cur_purities = aggregate_metrics[agg_key]['purity']
            logging.info(f"  Purity (n={len(cur_purities)}): {np.mean(cur_purities) * 100:.2f}%")

        # print agreement over single-genre files
        if 'agreement' in aggregate_metrics[agg_key]:
            num_agreements, num_comparisons = 0, 0
            cur_agreements = aggregate_metrics[agg_key]['agreement'] # {genre: [label, label, ...]}
            for cur_genre in cur_agreements:
                assignment_counts = Counter(cur_agreements[cur_genre])
                max_label, max_label_count = assignment_counts.most_common(1)[0]
                num_assignments = sum(assignment_counts.values())
                cur_agreement = max_label_count/num_assignments
                num_agreements += max_label_count
                num_comparisons += num_assignments
                logging.info(
                    f"  Agreement ({cur_genre} -> {pred_names[max_label][:50]}): "
                    f"{cur_agreement * 100:.2f}% (n={max_label_count}/{num_assignments})"
                )
            avg_agreement = num_agreements / num_comparisons
            logging.info(f"  Agreement (mean): {avg_agreement * 100:.2f}% (n={num_agreements}/{num_comparisons})")

        # print overlap errors
        if 'overlap' in aggregate_metrics[agg_key]:
            cur_overlap_errors = aggregate_metrics[agg_key]['overlap']
            logging.info(
                f"  Inter-file overlap error (n={len(cur_overlap_errors)}): {np.mean(cur_overlap_errors):.4f} mean "
                f"({np.std(cur_overlap_errors):.4f} stddev, {np.max(cur_overlap_errors):.4f} max, {np.min(cur_overlap_errors):.4f} min)"
            )

        # print ratio of correct classifications
        if 'correct' in aggregate_metrics[agg_key]:
            cur_ratio_correct = aggregate_metrics[agg_key]['correct'] / cur_num_instances
            logging.info(f"  Ratio of in-genre classifications: {cur_ratio_correct * 100:.2f}%")


if __name__ == '__main__':
    main()
