import numpy as np
import tensorflow as tf
import random


# ======================================================
#     Uncertainty Scoring  (Entropy)
# ======================================================

def compute_uncertainty_scores(model, X_unlabeled, batch_size=256):
    probs = model.predict(X_unlabeled, batch_size=batch_size, verbose=0)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-10, 1)), axis=1)
    return entropy


# ======================================================
#     Select Top-N Most Uncertain Samples
# ======================================================

def select_most_uncertain(X_unlabeled, scores, budget):
    idx_sorted = np.argsort(scores)[::-1]   # highest entropy first
    selected_idx = idx_sorted[:budget]
    return selected_idx


# ======================================================
#     Update Dataset (Move selected → labeled)
# ======================================================

def update_datasets(X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_idx):
    X_sel = X_unlabeled[selected_idx]
    y_sel = y_unlabeled[selected_idx]

    if X_labeled is None:
        X_labeled = X_sel
        y_labeled = y_sel
    else:
        X_labeled = np.concatenate([X_labeled, X_sel], axis=0)
        y_labeled = np.concatenate([y_labeled, y_sel], axis=0)

    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[selected_idx] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled


# ======================================================
#     Full Active Learning Cycle
# ======================================================

def active_learning_cycle(
    model,
    X_labeled,
    y_labeled,
    X_unlabeled,
    y_unlabeled,
    cycles=5,
    budget=200,
    epochs=3,
    batch_size=256,
):
    """
    main active learning loop:
        1) train student
        2) compute uncertainty on unlabeled
        3) pick top uncertain samples
        4) move to labeled set
        5) repeat
    """

    for cycle in range(cycles):
        print(f"\n========== ACTIVE LEARNING CYCLE {cycle+1}/{cycles} ==========")

        # 1) train student on current labeled set
        model.fit(
            X_labeled, y_labeled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # 2) compute uncertainty on unlabeled
        scores = compute_uncertainty_scores(model, X_unlabeled)

        # 3) select most uncertain samples
        selected_idx = select_most_uncertain(X_unlabeled, scores, budget)

        # 4) update datasets
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = update_datasets(
            X_labeled, y_labeled,
            X_unlabeled, y_unlabeled,
            selected_idx
        )

        print(f"Added {budget} samples to labeled set. Total labeled: {len(X_labeled)}")

    return model, X_labeled, y_labeled, X_unlabeled, y_unlabeled

