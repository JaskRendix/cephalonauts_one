import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from baseline import train
from joblib.memory import Memory
from tqdm.auto import tqdm

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]

memory = Memory(".cache", verbose=0)

parser = argparse.ArgumentParser()
parser.add_argument("--space", choices=["fsaverage5", "T1w", "MNI"], default="fsaverage5")
args = parser.parse_args()

TRAINING_SESSIONS = [1, 3, 6, 10, 15, 22]
FOLDS = {sub: range(5) for sub in ["0", "1", "2"]}

results = []
pbar = tqdm(
    total=sum(len(FOLDS[sub]) * len(TRAINING_SESSIONS) for sub in FOLDS),
    desc="Running baseline for all subsets, folds and session counts",
)
with pbar:
    for sub in FOLDS:
        for n_train_sessions in TRAINING_SESSIONS:
            for fold in FOLDS[sub]:
                kwargs = {"sub": sub, "n_train_sessions": n_train_sessions, "fold": fold, **vars(args)}
                res = memory.cache(train)(**kwargs, target="both")
                results.append(res | kwargs)
                pbar.update(1)
results = pd.DataFrame(results)
results.to_csv("scaling_laws.csv", index=False)

train_data_col = "Training Data (hours)"
results[train_data_col] = results.groupby(["sub", "n_train_sessions"])[train_data_col].transform("mean")
df = results.melt(
    id_vars=["sub", "fold", "n_train_sessions", train_data_col],
    value_vars=["Median Relative Rank (↓)", "Top-10 Retrieval Accuracy % (↑)", "Cosine Similarity (↑)"],
    var_name="metric", value_name="value",
)
df = df.rename(columns={"sub": "Subject"})
col_order = ["Median Relative Rank (↓)", "Top-10 Retrieval Accuracy % (↑)", "Cosine Similarity (↑)"]

g = sns.relplot(
    data=df,
    x=train_data_col,
    y="value",
    col="metric",
    col_order=col_order,
    hue="Subject",
    kind="line",
    style="Subject",
    dashes=False,
    markers=True,
    errorbar="sd",
    facet_kws={"sharey": None},
    legend=True,
    height=2.35,
    aspect=1,
)
sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, 0.95), ncol=3)
g.set_titles(template="{col_name}", pad=10)
g.set_ylabels("")
for ax in g.axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 5, 10, 20])
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(labelsize=8)
plt.savefig("scaling_laws.pdf", bbox_inches="tight")
