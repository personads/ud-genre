#!/usr/bin/python3

import argparse, json, logging, os, sys

from collections import defaultdict, OrderedDict

# local imports
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from data.utils import *


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Evaluate Sentence Genre Assignments')
    arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
    arg_parser.add_argument('ud_meta', help='path to Universal Dependencies metadata JSON')
    arg_parser.add_argument('mapping', help='path to JSON mapping of treebank genres to UD genres')
    arg_parser.add_argument('exp_path', help='path to experiment directory')
    arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
    arg_parser.add_argument('-sn', '--split_name', help='name of split to use for clustering (default: None - all splits)')
    return arg_parser.parse_args()


def get_prf_scores(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
    return precision, recall, f1_score


def main():
    args = parse_arguments()

    # check if experiment dir exists
    if not os.path.exists(args.exp_path):
        print(f"[Error] Could not find experiment directory '{args.exp_path}'. Exiting.")
        return

    # setup logging
    setup_logging(os.path.join(args.exp_path, 'eval-sentences.log'))

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
    logging.info(f"Loaded predictions for {pred_distribution.shape[0]} instances across {len(pred_names)} labels ({', '.join(pred_names)}).")

    # load UD metadata
    with open(args.ud_meta, 'r', encoding='utf8') as fp:
        meta = json.load(fp, object_pairs_hook=OrderedDict)

    # load TB->UD genre mapping
    with open(args.mapping, 'r', encoding='utf8') as fp:
        genre_mapping = json.load(fp)
    logging.info(f"Loaded treebank genre to UD genre mapping for {len(genre_mapping)} treebanks.")

    # load UD
    ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
    genres = ud.get_domains()
    logging.info(f"Loaded {ud} with {len(genres)} genres ({', '.join(genres)}).")

    # iterate over sentences per treebank
    ud_metrics = defaultdict(lambda: defaultdict(list))
    ud_purity = []
    ud_baseline_mftb = defaultdict(int)  # baseline: most frequent in-TB genre
    ud_genre_dist = defaultdict(int)
    tb_genre_dists = defaultdict(lambda: defaultdict(int))
    for tb, sentences in ud.get_sentences_by_treebank():
        tb_name = ud.get_treebank_name_of_index(sentences[0].idx)
        tb_language = ud.get_language_of_index(sentences[0].idx)
        tb_genres = ud.get_domains_of_index(sentences[0].idx)
        # check if treebank has group definitions
        if 'groups' not in meta[tb_language]['treebanks'][tb_name]:
            logging.info(f"{tb} contains no subgroups. Skipping.")
            continue
        # check if treebank has genre subgroups
        if 'genres' not in meta[tb_language]['treebanks'][tb_name]['groups']:
            logging.info(f"{tb} contains no 'genre' subgroups. Skipping.")
            continue
        # check if treebank is included in the TB->UD mapping
        if f'{tb_language}/{tb_name}' not in genre_mapping:
            logging.info(f"{tb_language}/{tb_name} has no treebank genre to UD genre mapping. Skipping.")
            continue

        # set up data
        rel_sentences, tb_idcs, rel_idcs, sen_rel_idcs = [], [], [], {}
        for pred_idx, sentence in enumerate(sentences):
            tb_idcs.append(sentence.idx)
            if (relevant_idcs is not None) and (sentence.idx not in relevant_idcs):
                continue
            rel_sentences.append(sentence)
            rel_idcs.append(sentence.idx)
            sen_rel_idcs[sentence.idx] = len(rel_idcs) - 1
        # skip if there are no sentences to cluster for current target
        if len(rel_idcs) < 1:
            logging.info(f"{tb} contains no relevant data. Skipping.")
            continue
        # gather genres in relevant subsample
        rel_genres = set()
        for rel_idx in rel_idcs:
            for genre in ud.get_domains_of_index(rel_idx):
                ud_genre_dist[genre] += 1
                rel_genres.add(genre)

        # set up group parser
        grouper = parse_grouper(meta[tb_language]['treebanks'][tb_name]['groups']['genres'])
        if grouper is None:
            logging.info(f"Could not parse group generator for {tb}. Skipping.")
            continue
        # segment sentences into genre groups {'group': [sent_0, sent_1, ...]}
        sentence_groups = grouper(rel_sentences)
        logging.info(f"Segmented {len(rel_idcs)} sentences into {len(sentence_groups)} groups ({', '.join(sentence_groups)}).")

        # calculate predicted distribution of current treebank
        tb_predictions = pred_distribution[rel_idcs, :] # (num_rel, num_genres)
        tb_labels = np.argmax(tb_predictions, axis=1) # (num_rel,)

        # calculate metrics of predictions based on mapping to global UD genre
        logging.info(f"UD/{tb_language}/{tb} (n={len(rel_idcs)}/{len(tb_idcs)}):")
        logging.info(f"  Genres (n={len(rel_genres)}/{len(tb_genres)}): {', '.join(sorted(rel_genres))}")
        tb_metrics = defaultdict(lambda: defaultdict(list))
        tb_purity = []
        tb_max_genre = None
        for tb_genre, genre_sentences in sentence_groups.items():
            # lookup treebank genre to UD genre mapping
            ud_genre = genre_mapping[f'{tb_language}/{tb_name}'][tb_genre]
            ud_genre_idx = genres.index(ud_genre)
            tb_genre_dists[f'{tb_language}/{tb_name}'][ud_genre] += len(genre_sentences)
            logging.info(f"  TB: {tb_genre} -> UD: {ud_genre} (n={len(genre_sentences)}):")
            # skip empty groups
            if len(genre_sentences) < 1:
                logging.info(f"    No relevant data in group {tb_genre}. Skipped.")
                continue
            # create mask of relevant sentences belonging to target genre
            cur_rel_mask = np.zeros(len(rel_idcs))
            cur_rel_mask[[sen_rel_idcs[s.idx] for s in genre_sentences]] = 1
            cur_rel_mask = np.array(cur_rel_mask, dtype=np.bool)
            # calculate purity
            cur_pred_labels = tb_labels[cur_rel_mask]
            cur_label_counts = np.bincount(cur_pred_labels)
            cur_max_label = cur_label_counts.argmax()
            purity = cur_label_counts[cur_max_label] / np.sum(cur_label_counts)
            tb_purity.append(purity)
            # gather global agreement
            tb_metrics['agreement'][ud_genre].append(cur_max_label)
            tb_metrics['confusion'][ud_genre].append(cur_label_counts)
            # gather most frequent TB genre
            if (tb_max_genre is None) or (len(genre_sentences) > len(sentence_groups[tb_max_genre])):
                tb_max_genre = tb_genre
            # calculate precision, recall and F1
            true_positives = np.sum(tb_labels[cur_rel_mask] == ud_genre_idx)
            false_positives = np.sum(tb_labels[np.logical_not(cur_rel_mask)] == ud_genre_idx)
            false_negatives = np.sum(tb_labels[cur_rel_mask] != ud_genre_idx)
            tb_metrics['tp'][ud_genre].append(true_positives)
            tb_metrics['fp'][ud_genre].append(false_positives)
            tb_metrics['fn'][ud_genre].append(false_negatives)
            precision, recall, f1_score = get_prf_scores(true_positives, false_positives, false_negatives)
            tb_metrics['precision'][ud_genre].append(precision)
            tb_metrics['recall'][ud_genre].append(recall)
            tb_metrics['f1'][ud_genre].append(f1_score)
            # print results
            logging.info(f"    Purity: {purity:.4f} (max label: {genres[cur_max_label]})")
            logging.info(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

        # aggregate treebank-level metrics
        for metric in tb_metrics:
            for genre in tb_metrics[metric]:
                ud_metrics[genre][metric] += tb_metrics[metric][genre]
        ud_purity += tb_purity

        # print treebank-level most-frequent baseline metrics (100% TPs, all non-max-genre FPs, all non-max-genre FNs)
        bl_tp = len(sentence_groups[tb_max_genre])
        bl_fp = sum([len(sentence_groups[g]) for g in sentence_groups if g != tb_max_genre])
        bl_fn = bl_fp
        ud_baseline_mftb['tp'] += bl_tp
        ud_baseline_mftb['fp'] += bl_fp
        ud_baseline_mftb['fn'] += bl_fn
        _, _, f1_baseline = get_prf_scores(bl_tp, bl_fp, bl_fn)
        logging.info(f"  F1 (TB-MF-baseline): {f1_baseline:.4f} ({tb_max_genre})")

        # print treebank-level model performance
        true_positives = np.sum([np.sum(v) for v in tb_metrics['tp'].values()])
        false_positives = np.sum([np.sum(v) for v in tb_metrics['fp'].values()])
        false_negatives = np.sum([np.sum(v) for v in tb_metrics['fn'].values()])
        precision, recall, f1_score = get_prf_scores(true_positives, false_positives, false_negatives)
        logging.info(f"  Purity (mean): {np.mean(tb_purity):.4f} ({np.std(tb_purity):.4f} stddev)")
        logging.info(f"  Precision (mean): {precision:.4f}")
        logging.info(f"  Recall (mean): {recall:.4f}")
        logging.info(f"  F1 (mean): {f1_score:.4f}")

    # print UD-level metrics
    ud_tp = np.sum([np.sum(m['tp']) for m in ud_metrics.values()])
    ud_fp = np.sum([np.sum(m['fp']) for m in ud_metrics.values()])
    ud_fn = np.sum([np.sum(m['fn']) for m in ud_metrics.values()])
    logging.info(f"UD (n={ud_tp + ud_fn}/{len(ud)}):")
    # print metrics per genre
    logging.info(f"  Genres (n={len(ud_metrics)}/{len(genres)}): {', '.join(sorted(ud_metrics))}")
    ud_agreement = []
    for genre, metrics in ud_metrics.items():
        true_positives = np.sum(metrics['tp'])
        false_positives = np.sum(metrics['fp'])
        true_negatives = np.sum(metrics['tn'])
        false_negatives = np.sum(metrics['fn'])
        precision, recall, f1_score = get_prf_scores(true_positives, false_positives, false_negatives)
        label_counts = np.bincount(metrics['agreement'])
        max_label = label_counts.argmax()
        agreement = label_counts[max_label] / np.sum(label_counts)
        ud_agreement.append(agreement)
        logging.info(f"  {genre} (n={true_positives + false_positives + true_negatives + false_negatives}):")
        logging.info(
            f"    Agreement (n={np.sum(label_counts)}): {agreement:.4f} "
            f" for {genres[max_label]} with {[(genres[l], c) for l, c in enumerate(label_counts) if c > 0]}"
        )
        logging.info(f"    Precision (mean): {precision:.4f}")
        logging.info(f"    Recall (mean): {recall:.4f}")
        logging.info(f"    F1 (mean): {f1_score:.4f}")

    # print UD-level most-frequent baseline metrics
    logging.info(f"  UD-MF baseline per treebank:")
    bl_tp, bl_fp, bl_fn = 0, 0, 0
    for tb in sorted(tb_genre_dists):
        tb_genres = list(tb_genre_dists[tb])
        counts_in_ud = {tbg: ud_genre_dist[tbg] for tbg in tb_genres}
        max_ud_genre, max_count = max(counts_in_ud.items(), key=lambda el: el[1])
        true_positives = counts_in_ud[max_ud_genre]
        false_positives = sum([count for tbg, count in counts_in_ud.items() if tbg != max_ud_genre])
        false_negatives = false_positives
        _, _, f1_baseline = get_prf_scores(true_positives, false_positives, false_negatives)
        bl_tp += true_positives
        bl_fp += false_positives
        bl_fn += false_negatives
        logging.info(f"    F1 (UD-MF baseline): {f1_baseline:.4f} ({max_ud_genre} in {tb})")
    _, _, f1_baseline = get_prf_scores(bl_tp, bl_fp, bl_fn)
    logging.info(f"  F1 (UD-MF baseline): {f1_baseline:.4f} (micro-avg)")

    # print most-frequent TB-genre baseline metrics (100% TPs, all non-genre FPs, 0% FNs)
    _, _, f1_baseline = get_prf_scores(ud_baseline_mftb['tp'], ud_baseline_mftb['fp'], ud_baseline_mftb['fn'])
    logging.info(f"  F1 (TB-MF baseline): {f1_baseline:.4f} (micro-avg)")

    # print aggregated metrics
    precision, recall, f1_score = get_prf_scores(ud_tp, ud_fp, ud_fn)
    logging.info(f"  Purity (mean): {np.mean(ud_purity):.4f} ({np.std(ud_purity):.4f} stddev)")
    logging.info(f"  Agreement (mean): {np.mean(ud_agreement):.4f} ({np.std(ud_agreement):.4f} stddev)")
    logging.info(f"  Precision (mean): {precision:.4f}")
    logging.info(f"  Recall (mean): {recall:.4f}")
    logging.info(f"  F1 (mean): {f1_score:.4f}")

    # pickle metrics
    with open(os.path.join(args.exp_path, 'eval-sentences.pkl'), 'wb') as fp:
        pickle.dump(dict(ud_metrics), fp)
    logging.info(f"Saved metrics to '{os.path.join(args.exp_path, 'eval-sentences.pkl')}'.")


if __name__ == '__main__':
    main()
