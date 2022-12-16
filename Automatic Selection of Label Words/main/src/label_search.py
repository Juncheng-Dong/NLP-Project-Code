"""Automatic label search helpers."""

import itertools
import torch
import tqdm
import multiprocessing
import numpy as np
import scipy.spatial as spatial
import scipy.special as special
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)


def select_likely_words(only_true_label, k_or_p, train_logits, train_labels, false_labels_weight, k_likely=1000, p_likely=0.2, vocab=None, is_regression=False):
    """Pre-select likely words based on conditional likelihood."""
    indices = []
    if is_regression:
        median = np.median(train_labels)
        train_labels = (train_labels > median).astype(np.int)
    num_labels = np.max(train_labels) + 1
    for idx in range(num_labels):
        if only_true_label==True:
            label_logits = train_logits[train_labels == idx]
            scores = label_logits.mean(axis=0)
            logger.info(f"false_scores: 0, not using them at all")
        else:
            true_label_logits = train_logits[train_labels == idx]
            true_scores = true_label_logits.mean(axis=0)
            logger.info(f"true_scores: {true_scores}")

            false_label_logits = train_logits[train_labels != idx]
            logger.info(f"false_label_logits: {false_label_logits}")
            false_scores = false_label_logits.mean(axis=0)
            if len(false_scores.shape)>1: # if more than one false label
                logger.info('false_scores.shape.size=2')
                false_scores = false_scores.mean(axis=0)
            logger.info(f"false_scores: {false_scores}")
            logger.info(f"false_labels_weight: {false_labels_weight}")

            scores = true_scores - false_labels_weight*false_scores
            
        kept = []
        scores_descending = []
        for i in np.argsort(-scores):
            current_score = scores[i]
            text = vocab[i]
            if not text.startswith("Ġ"):
                continue
            kept.append(i)
            scores_descending.append(current_score)
        if k_or_p=='k':
            indices.append(kept[:k_likely])
        else:
            p_sum=0
            current_indices=[]
            p_descending = special.softmax(np.array(scores_descending))
            for j, current_p in enumerate(len(p_descending)):
                if p_sum+current_p < p_likely:
                    current_indices.append(kept[j])
                    p_sum+=current_p
                else:
                    break
            indices.append(current_indices)

    return indices
'''def select_likely_words(train_logits, train_labels, k_likely=1000, vocab=None, is_regression=False):
    """Pre-select likely words based on conditional likelihood."""
    indices = []
    if is_regression:
        median = np.median(train_labels)
        train_labels = (train_labels > median).astype(np.int)
    num_labels = np.max(train_labels) + 1
    for idx in range(num_labels):
        label_logits = train_logits[train_labels == idx]
        scores = label_logits.mean(axis=0)
        kept = []
        for i in np.argsort(-scores):
            text = vocab[i]
            if not text.startswith("Ġ"):
                continue
            kept.append(i)
        indices.append(kept[:k_likely])
    return indices'''


def select_neighbors(distances, k_neighbors, valid):
    """Select k nearest neighbors based on distance (filtered to be within the 'valid' set)."""
    indices = np.argsort(distances)
    neighbors = []
    for i in indices:
        if i not in valid:
            continue
        neighbors.append(i)
    if k_neighbors > 0:
        return neighbors[:k_neighbors]
    return neighbors


def init(train_logits, train_labels):
    global logits, labels
    logits = train_logits
    labels = train_labels


def eval_pairing_acc(pairing):
    global logits, labels
    label_logits = np.take(logits, pairing, axis=-1)
    preds = np.argmax(label_logits, axis=-1)
    correct = np.sum(preds == labels)
    return correct / len(labels)


def eval_pairing_corr(pairing):
    global logits, labels
    if pairing[0] == pairing[1]:
        return -1
    label_logits = np.take(logits, pairing, axis=-1)
    label_probs = special.softmax(label_logits, axis=-1)[:, 1]
    pearson_corr = stats.pearsonr(label_probs, labels)[0]
    return pearson_corr


def find_labels(
    model,
    train_logits,
    train_labels,
    only_true_label,
    k_or_p,
    seed_labels=None,
    false_labels_weight=1,
    k_likely=1000,
    p_likely=0.2,
    k_neighbors=None,
    top_n=-1,
    vocab=None,
    is_regression=False,
):
    # Get top indices based on conditional likelihood using the LM.
    likely_indices = select_likely_words(
        only_true_label=only_true_label,
        k_or_p=k_or_p,
        train_logits=train_logits,
        train_labels=train_labels,
        false_labels_weight=false_labels_weight,
        k_likely=k_likely,
        p_likely=p_likely,
        vocab=vocab,
        is_regression=is_regression)

    '''def find_labels(
        model,
        train_logits,
        train_labels,
        seed_labels=None,
        k_likely=1000,
        k_neighbors=None,
        top_n=-1,
        vocab=None,
        is_regression=False,
    ):
        # Get top indices based on conditional likelihood using the LM.
        likely_indices = select_likely_words(
            train_logits=train_logits,
            train_labels=train_labels,
            k_likely=k_likely,
            vocab=vocab,
            is_regression=is_regression)'''

    logger.info("Top labels (conditional) per class:")
    for i, inds in enumerate(likely_indices):
        logger.info("\t| Label %d: %s", i, ", ".join([vocab[i] for i in inds[:10]]))

    # Convert to sets.
    valid_indices = [set(inds) for inds in likely_indices]

    # If specified, further re-rank according to nearest neighbors of seed labels.
    # Otherwise, keep ranking as is (based on conditional likelihood only).
    if seed_labels:
        assert(vocab is not None)
        seed_ids = [vocab.index(l) for l in seed_labels]
        vocab_vecs = model.lm_head.decoder.weight.detach().cpu().numpy()
        seed_vecs = np.take(vocab_vecs, seed_ids, axis=0)

        # [num_labels, vocab_size]
        label_distances = spatial.distance.cdist(seed_vecs, vocab_vecs, metric="cosine")

        # Establish label candidates (as k nearest neighbors).
        label_candidates = []
        logger.info("Re-ranked by nearest neighbors:")
        for i, distances in enumerate(label_distances):
            label_candidates.append(select_neighbors(distances, k_neighbors, valid_indices[i]))
            logger.info("\t| Label: %s", seed_labels[i])
            logger.info("\t| Neighbors: %s", " ".join([vocab[idx] for idx in label_candidates[i]]))
    else:
        label_candidates = likely_indices

    # Brute-force search all valid pairings.
    pairings = list(itertools.product(*label_candidates))

    if is_regression:
        eval_pairing = eval_pairing_corr
        metric = "corr"
    else:
        eval_pairing = eval_pairing_acc
        metric = "acc"

    # Score each pairing.
    pairing_scores = []
    with multiprocessing.Pool(initializer=init, initargs=(train_logits, train_labels)) as workers:
        with tqdm.tqdm(total=len(pairings)) as pbar:
            chunksize = max(10, int(len(pairings) / 1000))
            for score in workers.imap(eval_pairing, pairings, chunksize=chunksize):
                pairing_scores.append(score)
                pbar.update()

    # Take top-n.
    best_idx = np.argsort(-np.array(pairing_scores))[:top_n]
    best_scores = [pairing_scores[i] for i in best_idx]
    best_pairings = [pairings[i] for i in best_idx]

    logger.info("Automatically searched pairings:")
    for i, indices in enumerate(best_pairings):
        logger.info("\t| %s (%s = %2.2f)", " ".join([vocab[j] for j in indices]), metric, best_scores[i])

    return best_pairings
