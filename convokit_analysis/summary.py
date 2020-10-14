import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Dict, List
from convokit_analysis.utils import match_qa_pairs, subset_df

plt.style.use(["science", "grid"])


def plot_questions_by_recipient(
    df: pd.DataFrame,
    k: int,
    recipients: List[str] = [],
    question_categories: List[int] = [],
    cluster_labels: Dict = {},
    filename: str = "plot.pdf",
):
    matched_qas = match_qa_pairs(df, include_group=True)
    matched_qas.drop(["a_cluster"], axis=1, inplace=True)

    if recipients:
        matched_qas = matched_qas[matched_qas["group_a"].isin(recipients)]

    if question_categories:
        matched_qas = matched_qas[
            matched_qas["q_cluster"].isin(question_categories)
        ]
        question_labels = question_categories
    else:
        question_labels = list(range(k))

    if cluster_labels:
        legend_labels = [cluster_labels[i] for i in question_labels]
    else:
        legend_labels = question_labels

    matched_qas.rename(
        {
            "group_a": "answererCategory",
            "group_q": "questionerCategory",
            "q_cluster": "qCluster",
        },
        axis="columns",
        inplace=True,
    )

    grouped = (
        matched_qas.groupby(["answererCategory", "qCluster"]).count().unstack()
    )

    # normalize
    grouped["totals"] = grouped[grouped.columns].sum(axis=1)
    grouped_pct = grouped[grouped.columns].div(grouped.totals, axis=0)

    grouped_pct.drop(["totals"], axis=1, inplace=True)

    grouped_pct.plot.bar(stacked=False)
    plt.legend(legend_labels, title="Question categories", loc="upper left")
    plt.ylabel("Proportion of Questions")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    plt.close()


def group_by_cluster_and_speaker(
    df: pd.DataFrame,
    k: int,
    utterance_subset: str,
    cluster_labels: Dict = {},
    savedir: str = "",
    fn: str = "grouped.csv",
):
    """
    @param df dataframe of results
    @param k number of clusters
    @param cluster_labels optional dict mapping int cluster nums to labels
    @param utterance_subset question or answer
    """
    if not cluster_labels:
        for i in range(k):
            cluster_labels[i] = i

    if utterance_subset == "question":
        new_df = df.loc[df["is_question"]]
        new_df = new_df[["q_cluster", "group", "speaker"]]
    elif utterance_subset == "answer":
        new_df = df.loc[df["is_answer"]]
        new_df = new_df[["a_cluster", "group", "speaker"]]

    new_df.rename(
        {"q_cluster": "cluster", "a_cluster": "cluster"},
        axis="columns",
        inplace=True,
    )

    grouped = new_df.groupby(["cluster", "group"]).cluster.count().unstack().T
    grouped.columns = cluster_labels.values()

    grouped["totals"] = grouped[grouped.columns].sum(axis=1)
    grouped = grouped.sort_values(by=["totals"], ascending=False)
    grouped = grouped.fillna(0)

    grouped.to_csv(os.path.join(savedir, fn))

    return grouped


def generate_grouped_plots(
    df: pd.DataFrame,
    k: int,
    utterance_subset: str,
    normalize: bool = True,
    cluster_labels: Dict = {},
    cluster_subset: List = [],
    group_subset: List = [],
    savedir: str = "",
    fn: str = "proportions.pdf",
):
    """
    @param df dataframe of results
    @param k number of clusters
    @param cluster_labels dict mapping int cluster nums to labels
    @param utterance_subset question, answer
    @param savedir dir to save output
    """

    grouped = group_by_cluster_and_speaker(
        df=df,
        k=k,
        utterance_subset=utterance_subset,
        # cluster_labels=cluster_labels,
    )
    # do not use cluster_labels because we need to use the cluster nums
    # in subset_df below

    if normalize:
        speaker_pct = grouped[grouped.columns].div(grouped.totals, axis=0)
        speaker_pct.drop(["totals"], axis=1, inplace=True)
        speaker_pct = subset_df(speaker_pct, cluster_subset, group_subset)
        speaker_pct.rename(cluster_labels, axis="columns", inplace=True)
        speaker_pct.plot.bar(stacked=False)
    else:
        grouped.drop(["totals"], axis=1, inplace=True)
        grouped = subset_df(grouped, cluster_subset, group_subset)
        grouped.rename(cluster_labels, axis="columns", inplace=True)
        grouped.plot.bar(stacked=False)

    plt.show()
    plt.savefig(os.path.join(savedir, fn))
