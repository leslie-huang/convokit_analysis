import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from typing import Dict, List

from convokit_analysis.utils import match_qa_pairs

plt.style.use(["science", "grid"])


def generate_qa_transition_matrix(df: pd.DataFrame, normalized: bool = True):
    """
    Does not include NAs
    """
    matched_qas = match_qa_pairs(df)
    matched_qas = matched_qas.fillna(-1)
    axis_labels = sorted(matched_qas.a_cluster.unique())
    mat = matched_qas.groupby(["q_cluster", "a_cluster"]).count().unstack()

    if normalized:
        mat["totals"] = mat[mat.columns].sum(axis=1)
        mat = mat[mat.columns].div(mat.totals, axis=0)
        print(mat)
        mat = mat.drop(["totals"], axis=1)
        fmt_type = ".3f"
    else:
        fmt_type = ".1f"

    ax = sns.heatmap(
        mat,
        center=False,
        annot=True,
        linewidths=0.5,
        cmap="YlGnBu",
        fmt=fmt_type,
    )
    ax.tick_params(length=0)
    ax.set(xlabel="answer cluster", ylabel="question cluster")
    ax.set_xticklabels(axis_labels)


##################################################
# Transition matrix methods
##################################################
def subset_hearing(hearing_id: str, df: pd.DataFrame):
    """
    Subsets a df by hearing_id and returns the result as a list of dicts

    @param hearing_id unique id of a hearing (conversation)
    @param df dataframe of results
    @utterance type question or answer
    """

    this_convo = df[df.hearing_id == hearing_id]

    this_convo_dict = this_convo.to_dict(orient="records")

    return this_convo_dict


def generate_empty_transition_dict(
    states: List, add_init_term_states: bool = True
):
    """
    Takes an array of states. Returns a dict where keys = start states
    and values = dicts{keys = end states, values = 0
    [will be count of this transition]}

    @param add_init_term_states add "start" and "end" states
    """
    cluster_names = list(states)

    if add_init_term_states:
        starts = cluster_names + ["start"]
        ends = cluster_names + ["end"]
    else:
        starts = cluster_names
        ends = cluster_names

    transition_counts_dict = {k: {k: 0 for k in ends} for k in starts}

    return transition_counts_dict


def count_transition_probs(
    state_label: str,
    this_hearing_list_of_dicts: List[Dict],
    transition_counts_dict: Dict,
):
    """
    Iterates through this_hearing_list_of_dicts
    and, for each utterance,
    increments the count for the appropriate transition
    in transition_counts_dict

    It is assumed that transition_counts_dict is passed
    in with all values initialized to 0
    """
    num_utterances = len(this_hearing_list_of_dicts)

    for i in range(num_utterances):
        # look at i and the utterance BEFORE i

        this_state = this_hearing_list_of_dicts[i][state_label]

        # handle special start case
        if i == 0:
            prev_state = "start"

        else:
            prev_state = this_hearing_list_of_dicts[i - 1][state_label]

        # Increment the count in the matrix
        transition_counts_dict[prev_state][this_state] += 1

        # Handle end state
        if i == (num_utterances - 1):
            transition_counts_dict[this_state]["end"] += 1

    return transition_counts_dict


def compute_transition_counts(
    df: pd.DataFrame,
    state_label: str,
    unique_convos: List[str],
    utterance_type: str,
):
    """
    Takes a list of states and calls generate_empty_transition_dict()
    to make the transition_dict
    Then iterates through each hearing with subset_hearing()
    and uses count_transition_probs to add to the transition_dict
    """
    if utterance_type == "question":
        df = df[df.is_question == 1]
    else:
        df = df[df.is_question == 0]
    # ignoring any non-fitted utterances within hearings

    states = df[state_label].unique()  # list of possible transition states

    transition_counts_dict = generate_empty_transition_dict(
        states
    )  # initialize

    for hearing_id in unique_convos:
        this_hearing_dict = subset_hearing(hearing_id, df)

        transition_counts_dict = count_transition_probs(
            state_label, this_hearing_dict, transition_counts_dict
        )

    transition_counts_mat = pd.DataFrame.from_dict(
        transition_counts_dict, orient="index"
    )

    return transition_counts_mat


def mat_states_only(transition_counts_mat: pd.DataFrame):
    """
    Drop the start/end states in order to compute normalization
    """

    transition_counts_mat_states_only = transition_counts_mat.drop(
        ["start"], axis=0
    ).drop(["end"], axis=1)

    return transition_counts_mat_states_only


def mat_normalized_rows(transition_counts_mat_states_only: pd.DataFrame):
    """
    Converts counts to percents normalized by row
    Assumes that mat_states_only() has been used to
    drop "start"/"end" prior to this function
    """

    transition_counts_mat_states_only[
        "totals"
    ] = transition_counts_mat_states_only.sum(axis=0)

    for c in transition_counts_mat_states_only.columns:
        transition_counts_mat_states_only[
            c
        ] = transition_counts_mat_states_only[c].div(
            transition_counts_mat_states_only["totals"]
        )

    return transition_counts_mat_states_only.drop(["totals"], axis=1)


def generate_transition_matrix(
    df: pd.DataFrame,
    state_label: str,
    utterance_type: str,
    normalized: bool = False,
):
    """
    Generate a transition matrix of raw counts or normalized percents

    @param df dataframe of results
    @param state_label could be cluster_num or category
    @param normalized Compute transition matrix with raw
    counts or normalized pcts
    """
    conversation_roots = df.hearing_id.unique()
    transition_counts_clusters = compute_transition_counts(
        df, state_label, conversation_roots, utterance_type
    )

    if not normalized:
        display(transition_counts_clusters)

        ax = sns.heatmap(
            transition_counts_clusters,
            center=False,
            annot=True,
            linewidths=0.5,
            cmap="YlGnBu",
            fmt="g",
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0)

    else:
        transition_counts_clusters_states_only = mat_states_only(
            transition_counts_clusters
        )
        norm_mat_clusters = mat_normalized_rows(
            transition_counts_clusters_states_only
        )

        display(norm_mat_clusters)

        ax = sns.heatmap(
            norm_mat_clusters,
            center=False,
            annot=True,
            linewidths=0.5,
            cmap="YlGnBu",
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
