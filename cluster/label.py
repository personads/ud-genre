#!/usr/bin/python3

import argparse, json, logging, os, pickle, sys
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.spatial import distance

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from data.utils import *
from classify.boot import get_schedule


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Cluster Labeling')
    arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
    arg_parser.add_argument('emb_path', help='path to embedding directory')
    arg_parser.add_argument('exp_path', help='path to output directory')
    arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
    arg_parser.add_argument('-sn', '--split_name', help='name of split to use for clustering (default: None - all splits)')
    return arg_parser.parse_args()


def main():
    args = parse_arguments()

    # check if experiment dir exists
    if not os.path.exists(args.exp_path):
        print(f"[Error] Could not find experiment directory '{args.exp_path}'. Exiting.")
        return

    # setup logging
    log_path = os.path.join(args.exp_path, 'cluster-labeling.log')
    setup_logging(log_path)

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

    # load Universal Dependencies
    ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
    logging.info(f"Loaded {ud}.")

    # load pre-computed embeddings if provided
    embeddings = load_embeddings(args.emb_path, 'All')
    logging.info(f"Loaded {embeddings.shape[0]} pre-computed embeddings from '{args.emb_path}'.")
    assert embeddings.shape[0] == len(ud), f"[Error] Number of embeddings (n={embeddings.shape[0]}) and sentences in UD (n={len(ud)}) do not match."

    # iterate over treebank cluster assignments to compute cluster embeddings and genre combinations
    logging.info("-"*10)
    logging.info("Computing cluster embeddings...")
    # {('genre0', 'genre1', ...) -> {'tb_key': [{'idcs': [], 'emb': np.array}, ...]}}
    genre_combination_clusters = defaultdict(dict)
    for tb, sentences in ud.get_sentences_by_treebank():
        # set up data
        tb_idcs, rel_idcs, rel_pred_idcs = [], [], []
        for pred_idx, sentence in enumerate(sentences):
            tb_idcs.append(sentence.idx)
            if (relevant_idcs is not None) and (sentence.idx not in relevant_idcs):
                continue
            rel_idcs.append(sentence.idx)
            rel_pred_idcs.append(pred_idx)
        # skip if there are no sentences to cluster for current target
        if len(rel_idcs) < 1:
            logging.info(f"{tb} contains no relevant data. Skipping.")
            continue

        # gather current genres
        genres = {g for i in rel_idcs for g in ud.get_domains_of_index(i)}
        genre_combination = tuple(sorted(genres))

        # load cluster assignments
        with open(os.path.join(args.exp_path, f'{tb.replace(" ", "_")}.pkl'), 'rb') as fp:
            results = pickle.load(fp)
        # check if results contain cluster assignments
        assert 'domain_dist' in results, f"No predictions found in '{tb.replace(' ', '_')}.pkl'."
        # filter to relevant results
        assignment_dist = results['domain_dist'][rel_pred_idcs, :]
        assignments = assignment_dist.argmax(axis=-1)
        num_clusters = assignment_dist.shape[-1]
        num_sentences = assignment_dist.shape[0]
        assert num_sentences == len(rel_idcs), f"Mismatch between number of clustering results (n={num_sentences}) and relevant sentences from {tb} (n={len(rel_idcs)})."
        # gather sentence IDs per cluster [0 -> [sen0, sen2, ...], ...]
        cluster_sentences = [[] for _ in range(num_clusters)]
        for pred_idx, sen_idx in enumerate(rel_idcs):
            cur_cluster = assignments[pred_idx]
            cluster_sentences[cur_cluster].append(sen_idx)
        # TODO handle clusters without sentences
        assert sum([1 for idcs in cluster_sentences if len(idcs) > 0]) == num_clusters, f"One or more clusters for {tb} are empty."

        # compute cluster embeddings
        genre_combination_clusters[genre_combination][tb] = []
        for cluster, sen_idcs in enumerate(cluster_sentences):
            cluster_embedding = np.mean(embeddings[sen_idcs], axis=0)
            genre_combination_clusters[genre_combination][tb].append({
                'idcs': sen_idcs,
                'emb': cluster_embedding
            })

        logging.info(f"Computed embeddings for {len(cluster_sentences)} clusters in {tb} with genres {genre_combination}.")

    # compute schedule
    logging.info('-'*10)
    schedule = get_schedule(list(genre_combination_clusters.keys()))
    logging.info(f"Created schedule with {len(schedule)} environments:")
    for env_idx, environment in enumerate(schedule):
        logging.info(f"Env {env_idx + 1}:")
        for key in sorted(environment):
            logging.info(f"  {key} ({len(environment[key])}): {environment[key]}")
    if len(schedule[-1]['disjunct']) > 0:
        logging.error(f"[Error] Unable to bootstrap using this schedule. Exiting.")
        return

    # assign labels according to schedule
    logging.info('-'*10)
    for env_idx, environment in enumerate(schedule):
        # gather known genres and compute mean embeddings
        known_embeddings = np.zeros((len(environment['known']), embeddings.shape[1])) # (num_known, emb_dim)
        for kg_idx, known_genre in enumerate(environment['known']):
            kg_embedding = np.mean([
                c['emb'] for tb, clusters in genre_combination_clusters[(known_genre, )].items() for c in clusters
            ], axis=0)
            known_embeddings[kg_idx] = kg_embedding
        logging.info(f"[Env {env_idx + 1}/{len(schedule)}] Labeling mixtures based on {len(environment['known'])} genres.")
        if len(environment['predict']) < 1:
            logging.info(f"No more predictions to make. Schedule complete.")
            break

        # look up closest clusters for each prediction mixture
        prediction_combinations = list(genre_combination_clusters.keys())
        for cmb_idx, genre_combination in enumerate(prediction_combinations):
            # skip single-genre combinations (i.e. already labeled)
            if len(genre_combination) < 2:
                logging.info(f"[Pred {cmb_idx + 1}/{len(prediction_combinations)}] Skipped single-genre combination {genre_combination}.")
                continue

            # gather embeddings of known genres in current combination
            pred_genres, pred_genre_idcs = [], []
            for kg_idx, known_genre in enumerate(environment['known']):
                if known_genre not in genre_combination: continue
                pred_genres.append(known_genre)
                pred_genre_idcs.append(kg_idx)
            unpred_genres = list(set(genre_combination) - set(pred_genres))
            rel_known_embeddings = known_embeddings[pred_genre_idcs, :]
            if len(pred_genres) < 1:
                logging.info(f"[Pred {cmb_idx + 1}/{len(prediction_combinations)}] No predictable genres in {genre_combination}. Skipped (for now).")
                continue
            logging.info(
                f"[Pred {cmb_idx + 1}/{len(prediction_combinations)}] Predicting {len(pred_genres)} genres ({', '.join(pred_genres)}) "
                f"within {len(genre_combination)} ({', '.join(genre_combination)})...")

            # within each treebank, match the closest cluster embeddings
            genre_combination_treebanks = list(genre_combination_clusters[genre_combination].items())
            for tb, tb_clusters in genre_combination_treebanks:
                logging.info(f"{tb}: Predicting {len(pred_genres)}/{len(tb_clusters)} clusters...")
                tb_cluster_embeddings = np.array([c['emb'] for c in tb_clusters])
                cos_distances = distance.cdist(tb_cluster_embeddings, rel_known_embeddings, 'cosine') # (num_clusters, num_rel_genres)

                # compute all minimum cluster-genre distances
                unlabeled_clusters = list(range(len(tb_clusters)))
                unassigned_genres = list(range(len(pred_genre_idcs)))
                while len(unassigned_genres) > 0:
                    rel_cluster_idcs, rel_genre_idcs = np.array(unlabeled_clusters), np.array(unassigned_genres)
                    rel_distances = cos_distances[rel_cluster_idcs[:, None], rel_genre_idcs] # (num_unlabeled, num_unassigned)
                    rel_cluster_idx, rel_genre_idx = np.unravel_index(np.argmin(rel_distances), rel_distances.shape)
                    closest_cluster_idx = unlabeled_clusters[rel_cluster_idx]
                    closest_genre_idx = unassigned_genres[rel_genre_idx]
                    closest_cluster = tb_clusters[closest_cluster_idx]
                    closest_genre = pred_genres[closest_genre_idx]

                    # add to cluster to set of single-genre treebanks
                    if tb not in genre_combination_clusters[(closest_genre, )]:
                        genre_combination_clusters[(closest_genre, )][tb] = []
                    genre_combination_clusters[(closest_genre, )][tb].append(closest_cluster)
                    # remove cluster, genre from unlabeled queue
                    unlabeled_clusters.remove(closest_cluster_idx)
                    unassigned_genres.remove(closest_genre_idx)
                    logging.info(
                        f"Labeled {tb}'s cluster {closest_cluster_idx} as {closest_genre} "
                        f"(cos: {cos_distances[closest_cluster_idx, closest_genre_idx]:.4f}).")

                # if only one unpredicted genre remains, label it as such
                if len(unpred_genres) == 1:
                    inferred_genre = unpred_genres[0]
                    assert len(unlabeled_clusters) == 1, f"[Error] Inferring 1 genre ({inferred_genre}), but {len(unlabeled_clusters)} unlabeled clusters remain."
                    inferred_cluster_idx = unlabeled_clusters[0]
                    # remove cluster from unlabeled queue
                    unlabeled_clusters.remove(inferred_cluster_idx)
                    # add to cluster to set of single-genre treebanks
                    if tb not in genre_combination_clusters[(inferred_genre, )]:
                        genre_combination_clusters[(inferred_genre, )][tb] = []
                    genre_combination_clusters[(inferred_genre,)][tb].append(tb_clusters[inferred_cluster_idx])
                    logging.info(f"Labeled {tb}'s cluster {inferred_cluster_idx} as {inferred_genre} (inferred).")

                # check if any unlabeled clusters remain
                if len(unlabeled_clusters) > 0:
                    # move treebank from old set of genres to the set of remaining, unpredicted ones
                    unpred_combination = tuple(sorted(unpred_genres))
                    if tb not in genre_combination_clusters[unpred_combination]:
                        genre_combination_clusters[unpred_combination][tb] = []
                    # remove labeled clusters from prediction set
                    genre_combination_clusters[unpred_combination][tb] = [
                        cluster for cidx, cluster in enumerate(tb_clusters) if cidx in unlabeled_clusters
                    ]
                    logging.info(f"Moved {tb} from {genre_combination} -> {unpred_combination}.")
                # remove treebank from old set of genres
                del(genre_combination_clusters[genre_combination][tb])

            # remove current genre combination if it has been resolved
            if len(genre_combination_clusters[genre_combination]) < 1:
                del(genre_combination_clusters[genre_combination])
                logging.info(f"Completed labeling of {len(pred_genres)}/{len(genre_combination)} genres in treebanks with mixture {genre_combination}.")

    # save predictions
    logging.info("-"*10)
    logging.info("Saving label predictions...")
    # initialize genre predictions with zeros
    domain_dist = np.zeros((len(ud), len(ud.get_domains())))
    # gather all predictions
    genres = sorted(ud.get_domains())
    genre_idcs = {g: i for i, g, in enumerate(genres)}
    for genre in genre_combination_clusters:
        genre_idx = genre_idcs[genre[0]]
        for tb, tb_clusters in genre_combination_clusters[genre].items():
            for tb_cluster in tb_clusters:
                domain_dist[tb_cluster['idcs'], genre_idx] = 1.
                logging.info(f"Labeled {len(tb_cluster['idcs'])} sentences from {tb} as {genre[0]}.")
                # store results on disk

    # store results to disk
    results_path = os.path.join(args.exp_path, 'All-labeled.pkl')
    with open(results_path, 'wb') as fp:
        pickle.dump(
            {
                'domains': genres,
                'domain_dist': domain_dist
            },
            fp
        )
    logging.info(f"Saved label predictions to '{results_path}'.")


if __name__ == '__main__':
    main()
