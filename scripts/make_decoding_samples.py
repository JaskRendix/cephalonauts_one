import argparse
import typing as tp

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from joblib.memory import Memory
from nilearn.signal import clean as clean_signal
from scipy.ndimage import uniform_filter1d
from sentence_transformers import SentenceTransformer
from torchaudio import functional as F
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor

memory = Memory(".cache", verbose=0)


def make_text_samples(transcript: pd.DataFrame, timestamps: np.ndarray, n_words_context: int = 20) -> tp.List[str]:
    offsets = transcript["onset"] + transcript["duration"]
    samples = []
    for timestamp in timestamps:
        mask = offsets <= timestamp
        words = transcript["word"][mask]
        samples.append(" ".join(words[-n_words_context:]))

    return samples


def compute_text_embeddings(texts: tp.List[str], model_name: str = "Qwen/Qwen3-Embedding-8B", **kwargs) -> np.ndarray:
    model = SentenceTransformer(model_name, **kwargs)
    print(f"Embedding {len(texts)} text samples with {model_name} on {model.device}")
    return model.encode(texts, show_progress_bar=True)


def make_audio_samples(
    audio_paths: str | tp.List[str],
    timestamps: np.ndarray | tp.List[np.ndarray],
    audio_context_seconds: float = 10.0,
    target_sr: tp.Optional[int] = None,
) -> tp.Tuple[tp.List[torch.Tensor], tp.List[int]]:
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    if isinstance(timestamps, np.ndarray):
        timestamps = [timestamps]

    waveforms = []
    sampling_rates = []
    for audio_path, run_timestamps in zip(audio_paths, timestamps):
        audio_path = hf_hub_download(repo_id="karavela/cephalonauts_one", filename=audio_path, repo_type="dataset")
        waveform, sr = torchaudio.load(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)  # Aggregate to mono if stereo
        if target_sr is not None and sr != target_sr:
            waveform = F.resample(waveform, sr, target_sr)
            sr = target_sr
        for timestamp in run_timestamps:
            start = int((timestamp - audio_context_seconds) * sr)
            end = int(timestamp * sr)
            waveforms.append(waveform[start:end])
            sampling_rates.append(sr)

    return waveforms, sampling_rates


def compute_audio_embeddings(
    waveforms: tp.List[torch.Tensor],
    sampling_rate: int | tp.List[int],
    model_name: str = "LCO-Embedding/LCO-Embedding-Omni-7B",
    **kwargs,
) -> np.ndarray:
    if isinstance(sampling_rate, int):
        sampling_rate = [sampling_rate] * len(waveforms)

    model = SentenceTransformer(model_name, **kwargs)
    inputs = [{"array": waveform.numpy(), "sampling_rate": sr} for waveform, sr in zip(waveforms, sampling_rate)]
    print(f"Embedding {len(waveforms)} audio samples with {model_name} on {model.device}")
    return model.encode(inputs, show_progress_bar=True)


def audio_samples_wrapper(
    audio_paths: str | tp.List[str],
    timestamps: np.ndarray | tp.List[np.ndarray],
    audio_context_seconds: float = 10.0,
    model_name: str = "LCO-Embedding/LCO-Embedding-Omni-7B",
    **kwargs,
) -> np.ndarray:
    target_sr = AutoFeatureExtractor.from_pretrained(model_name).sampling_rate
    waveforms, sr = make_audio_samples(audio_paths, timestamps, audio_context_seconds, target_sr=target_sr)
    return compute_audio_embeddings(waveforms, sr, model_name, **kwargs)


def preprocess_fmri(X: np.ndarray, tr: float, temporal_avg: int = 4) -> np.ndarray:
    X = clean_signal(X, detrend=True, standardize="zscore_sample", t_r=tr)
    X = uniform_filter1d(X, size=temporal_avg, axis=0)

    return X


def build_XY_dataframe(
    df: pd.DataFrame,
    space: str = "fsaverage5",
    fmri_n_skip: int = 3,
    lag: float = 2.5,
    n_words_context: int = 20,
    text_model_name: str = "Qwen/Qwen3-Embedding-8B",
    audio_context_seconds: float = 10.0,
    audio_model_name: str = "LCO-Embedding/LCO-Embedding-Omni-7B",
    temporal_avg: int = 4,
) -> pd.DataFrame:
    metacols = [
        "sub",
        "ses",
        "run",
        "tr",
        "fmri_offset",
        "audio",
        "audio_offset",
        "audio_duration",
    ]
    # Test if all expected columns are present
    df[metacols + ["transcript", f"{space.lower()}_npy"]]

    results = []
    audio_paths = []
    audio_timestamps = []
    pbar = tqdm(total=len(df), desc=f"Preparing samples for sub-{sub} and preprocessing fMRI in {space} space")
    with pbar:
        for _, row in df.iterrows():
            tr = row["tr"]
            X = np.load(
                hf_hub_download(
                    repo_id="karavela/cephalonauts_one", filename=row[f"{space.lower()}_npy"], repo_type="dataset"
                )
            )
            run_fmri_duration = X.shape[0] * tr
            X = preprocess_fmri(X, tr=tr, temporal_avg=temporal_avg)
            fmri_timestamps = row["fmri_offset"] + np.arange(X.shape[0]) * tr
            transcript = pd.DataFrame(row["transcript"].tolist())  # Columns are ['onset', 'duration', 'word']
            word_offsets = transcript["onset"] + transcript["duration"]

            timestamps = fmri_timestamps[:: (fmri_n_skip + 1)] - lag
            # Skip timestamps that are outside the range of transcript or audio with requested context
            mask = timestamps >= row["audio_offset"] + audio_context_seconds  # Too early to have audio context
            mask = mask & (timestamps <= row["audio_offset"] + row["audio_duration"])  # After the end of the audio
            mask = mask & (
                timestamps >= word_offsets.nsmallest(n_words_context).max()
            )  # Too early to have text context
            mask = mask & (timestamps <= word_offsets.max())  # After the end of the transcript
            timestamps = timestamps[mask]
            X = X[:: (fmri_n_skip + 1)][mask]

            text_samples = make_text_samples(
                transcript=transcript,
                timestamps=timestamps,
                n_words_context=n_words_context,
            )
            run = pd.DataFrame({
                "X": X.tolist(),
                "timestamps": timestamps,
                "text": text_samples,
            })
            for col in metacols:
                run[col] = row[col]
            run["run_fmri_duration"] = run_fmri_duration
            run["fmri"] = row[f"{space.lower()}_npy"]
            results.append(run)
            audio_paths.append(row["audio"])
            audio_timestamps.append(timestamps - row["audio_offset"])
            pbar.update(1)

    results = pd.concat(results, ignore_index=True)

    Y_text = memory.cache(compute_text_embeddings)(texts=results["text"].tolist(), model_name=text_model_name)
    Y_audio = memory.cache(audio_samples_wrapper)(
        audio_paths=audio_paths,
        timestamps=audio_timestamps,
        audio_context_seconds=audio_context_seconds,
        model_name=audio_model_name,
    )

    results["Y_audio"] = Y_audio.tolist()
    results["Y_text"] = Y_text.tolist()

    # Reorder columns
    first_cols = ["timestamps", "text", "X", "Y_text", "Y_audio"]
    results = results[first_cols + [col for col in results.columns if col not in first_cols]]

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sub", type=int, choices=range(3), default=0)
    p.add_argument("--space", type=str, choices=["fsaverage5", "T1w", "MNI"], default="fsaverage5")
    p.add_argument("--val-session", type=int, default=22)
    p.add_argument("--test-session", type=int, default=23)
    p.add_argument("--push", action="store_true", help="Upload generated splits to karavela/cephalonauts_one")

    args = p.parse_args()
    sub = args.sub
    space = args.space
    val_session = args.val_session
    test_session = args.test_session

    ds = load_dataset("karavela/cephalonauts_one", name="timeseries", split="metadata")
    df = ds.to_pandas().sort_values(["sub", "ses", "run"])
    df_XY = build_XY_dataframe(df=df[df["sub"] == sub], space=space)

    df_XY["split"] = "train"
    df_XY.loc[df_XY["ses"] == val_session, "split"] = "val"
    df_XY.loc[df_XY["ses"] == test_session, "split"] = "test"
    ds = DatasetDict({
        split: Dataset.from_pandas(df_XY[df_XY["split"] == split], preserve_index=False)
        for split in ["train", "val", "test"]
    })

    # Save locally unless uploading was explicitly requested.
    if args.push:
        ds.push_to_hub(
            repo_id="karavela/cephalonauts_one",
            config_name=f"sub-{sub}_{space}",
            data_dir=f"derivatives/decoderprep/sub-{sub}/{space}",
            private=True,
        )
    else:
        ds.save_to_disk(f".cache/cephalonauts_one/sub-{sub}_{space}")
