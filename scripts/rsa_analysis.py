"""
Representational Similarity Analysis (RSA).

Zero neural networks. Zero training loops. 

Measures raw geometric alignment between fMRI activity (X) and target 
embedding spaces (Y_text, Y_audio) via pairwise distance correlations (Spearman on RDMs). 

Acts as a pre-flight check: tests whether semantic and acoustic structure 
already mirrors neural geometry before a decoder ever learns a map.
"""

import argparse

import numpy as np
from datasets import load_dataset
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def compute_rdm(X: np.ndarray) -> np.ndarray:
    """
    Compute a Representational Dissimilarity Matrix (RDM)
    using cosine distance.

    Returns the condensed upper-triangular form produced
    by scipy.spatial.distance.pdist().
    """
    return pdist(X, metric="cosine")


def rsa_score(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute RSA score using Spearman correlation
    between pairwise distance vectors.
    """
    X_rdm = compute_rdm(X)
    Y_rdm = compute_rdm(Y)

    rho, _ = spearmanr(X_rdm, Y_rdm)

    return float(rho)


def prepare_brain_data(
    X: np.ndarray,
    pca_components: int | None,
) -> np.ndarray:
    """
    Optional PCA followed by normalization.
    """

    if pca_components is not None:
        max_components = min(
            X.shape[0] - 1,
            X.shape[1],
            pca_components,
        )

        if max_components > 0:
            X = PCA(
                n_components=max_components,
                random_state=0,
            ).fit_transform(X)

    return normalize(X)


def load_data(
    sub: str,
    space: str,
    split: str,
    max_samples: int,
):
    ds = load_dataset(
        "karavela/cephalonauts_one",
        name=f"sub-{sub}_{space}",
        split=split,
    )

    n_samples = min(max_samples, len(ds))

    ds = ds.select(range(n_samples))

    X = np.asarray(ds["X"], dtype=np.float32)
    Y_text = np.asarray(ds["Y_text"], dtype=np.float32)
    Y_audio = np.asarray(ds["Y_audio"], dtype=np.float32)

    return X, Y_text, Y_audio


def main():
    parser = argparse.ArgumentParser(
        description="Representational Similarity Analysis (RSA) for Cephalonauts One."
    )

    parser.add_argument(
        "--sub",
        default="0",
        choices=["0", "1", "2"],
        help="Subject identifier.",
    )

    parser.add_argument(
        "--space",
        default="fsaverage5",
        choices=["fsaverage5", "T1w", "MNI"],
        help="Brain representation space.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Maximum number of samples to analyze.",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Number of PCA dimensions for fMRI features. Use 0 to disable PCA.",
    )

    args = parser.parse_args()

    print("Loading dataset...")

    X, Y_text, Y_audio = load_data(
        sub=args.sub,
        space=args.space,
        split=args.split,
        max_samples=args.max_samples,
    )

    pca_components = (
        None
        if args.pca_components <= 0
        else args.pca_components
    )

    print()
    print(f"Subject          : {args.sub}")
    print(f"Space            : {args.space}")
    print(f"Split            : {args.split}")
    print(f"Samples          : {len(X)}")
    print(f"Brain dimensions : {X.shape[1]}")
    print(f"Text dimensions  : {Y_text.shape[1]}")
    print(f"Audio dimensions : {Y_audio.shape[1]}")
    print(
        f"PCA components   : "
        f"{pca_components if pca_components is not None else 'disabled'}"
    )
    print()

    X = prepare_brain_data(
        X,
        pca_components=pca_components,
    )

    Y_text = normalize(Y_text)
    Y_audio = normalize(Y_audio)

    print("Computing RSA scores...")

    text_rsa = rsa_score(X, Y_text)
    audio_rsa = rsa_score(X, Y_audio)

    rng = np.random.default_rng(0)

    shuffled_text_rsa = rsa_score(
        X,
        Y_text[rng.permutation(len(Y_text))],
    )

    shuffled_audio_rsa = rsa_score(
        X,
        Y_audio[rng.permutation(len(Y_audio))],
    )

    print()
    print("=" * 60)
    print("RSA RESULTS")
    print("=" * 60)

    print(
        f"Brain ↔ Text           : "
        f"{text_rsa:.4f}"
    )

    print(
        f"Brain ↔ Audio          : "
        f"{audio_rsa:.4f}"
    )

    print()
    print("Shuffled Controls")

    print(
        f"Brain ↔ Shuffled Text  : "
        f"{shuffled_text_rsa:.4f}"
    )

    print(
        f"Brain ↔ Shuffled Audio : "
        f"{shuffled_audio_rsa:.4f}"
    )

    print("=" * 60)

    print()
    print("Interpretation")
    print(
        "- Positive RSA values indicate shared representational geometry."
    )
    print(
        "- Shuffled baselines provide an estimate of chance-level alignment."
    )
    print(
        "- Large gaps between real and shuffled RSA suggest meaningful structure."
    )
    print(
        "- Differences between text and audio RSA may reveal whether brain activity aligns more strongly with semantic or acoustic representations."
    )


if __name__ == "__main__":
    main()
  
