import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Dict, Optional, List

plt.style.use(["science", "grid"])

def plot_questions_by_recipient(
    df: pd.DataFrame,
    k: int,
    recipients: List = [str],
    question_categories: List = [int],
    cluster_labels: Dict = {},
    filename: str = "plot.pdf"
):
    matched_qas = match_qa_pairs(df, include_group=True)
    matched_qas.drop(["a_cluster"], axis=1, inplace=True)

    if recipients:
        matched_qas = matched_qas[matched_qas["group_a"].isin(recipients)]

    if question_categories:
        matched_qas = matched_qas[matched_qas["q_cluster"].isin(question_categories)]
        question_labels = question_categories
    else:
        question_labels = list(range(k))

    if cluster_labels:
        legend_labels = [cluster_labels[i] for i in question_labels]
    else:
        legend_labels = question_labels

    matched_qas.rename(
        {"group_a": "answererCategory",
         "group_q": "questionerCategory",
        "q_cluster": "qCluster"},
        axis="columns",
        inplace=True,
    )

    grouped = (
        matched_qas.groupby(["answererCategory", "qCluster"])
        .count()
        .unstack()
    )

    # normalize
    grouped["totals"] = grouped[grouped.columns].sum(axis=1)
    grouped_pct = grouped[grouped.columns].div(grouped.totals, axis=0)

    grouped_pct.drop(["totals"], axis=1, inplace=True)

    grouped_pct.plot.bar(stacked=False)
    plt.legend(
        legend_labels,
        title="Question categories",
        loc="upper left",
    )
    plt.ylabel("Proportion of Questions")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    plt.close()




# def generate_results(
#     df: pd.DataFrame,
#     k: int,
#     rank: int,
#     cluster_labels: Dict,
#     utterance_subset: str,
#     savedir: Optional[str] = None,
# ):
#     """
#     @param df dataframe of results
#     @param k number of clusters
#     @param rank rank of SVD reduction
#     @param cluster_labels dict mapping int cluster nums to labels
#     @param utterance_subset question, answer, or all
#     @param savedir dir to save output
#     """

#     if utterance_subset == "question":
#         df = df[df["is_question"] is True]
#     elif utterance_subset == "answer":
#         df = df[df["is_question"] == 0]

#     if savedir is None:
#         savedir = os.getcwd()
#     if os.path.exists(savedir):
#         print(f"Warning: directory {savedir} exists")
#     else:
#         os.mkdir(savedir)

#     grouped = (
#         df.groupby(["cluster_num", "category"]).cluster_num.count().unstack().T
#     )
#     grouped.columns = cluster_labels.values()

#     grouped["totals"] = grouped[grouped.columns].sum(axis=1)
#     grouped = grouped.sort_values(by=["totals"], ascending=False)

#     print(f"Raw totals for {utterance_subset}")
#     display(grouped)
#     grouped.to_csv(
#         os.path.join(
#             savedir, f"k{k}_rank{rank}_{utterance_subset}_raw_counts.csv"
#         )
#     )
#     grouped.drop(["totals"], axis=1).plot.barh(
#         stacked=True,
#         figsize=(15, 10),
#         title=f"Raw counts for {utterance_subset}",
#     )
#     plt.savefig(
#         os.path.join(
#             savedir, f"k{8}_rank{rank}_{utterance_subset}_raw_counts.pdf"
#         )
#     )
#     plt.show()

#     print(f"{utterance_subset} breakdown by speaker category")
#     speaker_pct = grouped[grouped.columns].div(grouped.totals, axis=0)
#     display(speaker_pct)
#     speaker_pct.to_csv(
#         os.path.join(
#             savedir, f"k{k}_rank{rank}_{utterance_subset}_pct_by_speaker.csv"
#         )
#     )
#     speaker_pct.drop(["totals"], axis=1).plot.barh(
#         stacked=True,
#         figsize=(15, 10),
#         title=f"{utterance_subset} Percents by Speaker Category",
#     )
#     plt.savefig(
#         os.path.join(
#             savedir, f"k{k}_rank{rank}_{utterance_subset}_pct_by_speaker.pdf"
#         )
#     )
#     plt.show()

#     print(f"{utterance_subset} breakdown by cluster")
#     cluster_pct = grouped.drop(["totals"], axis=1).T
#     cluster_pct["totals"] = cluster_pct[cluster_pct.columns].sum(axis=1)
#     cluster_pct = cluster_pct[cluster_pct.columns].div(
#         cluster_pct.totals, axis=0
#     )
#     display(cluster_pct)
#     cluster_pct.to_csv(
#         os.path.join(
#             savedir, f"k{k}_rank{rank}_{utterance_subset}_pct_by_cluster.csv"
#         )
#     )
#     cluster_pct.drop(["totals"], axis=1).plot.barh(
#         stacked=True,
#         figsize=(15, 10),
#         title=f"{utterance_subset} Percents by Cluster",
#     )
#     plt.savefig(
#         os.path.join(
#             savedir, f"k{k}_rank{rank}_{utterance_subset}_pct_by_cluster.pdf"
#         )
#     )
#     plt.show()
#     plt.close()


# def generate_results_group(
#     df: pd.DataFrame,
#     speaker_categories: List,
#     k: int,
#     rank: int,
#     cluster_labels: Dict,
#     utterance_subset: str,
#     savedir: str,
# ):
#     """
#     @param df dataframe of results
#     @param speaker_categories list of speaker categories to restrict results to
#     @param k number of clusters
#     @param rank rank of SVD reduction
#     @param cluster_labels dict mapping int cluster nums to labels
#     @param subset question, answer, or all
#     @param savedir dir to save output
#     """
#     df = df[df.category.isin(speaker_categories)]

#     generate_results(df, k, rank, cluster_labels, utterance_subset, savedir)

