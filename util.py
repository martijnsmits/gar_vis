import pandas as pd
from enum import Enum
import tkinter as tk


class Style(str, Enum):
    TITLE = "bold"
    VARIABLE = "cyan"
    BACKGROUND = "white"
    REL_DOC = "lightgray"
    REL_DOC_NEW = "lime"
    REL_DOC_DUPE = "lightcoral"


def get_color_scale(neighbourhood):
    if 3 in neighbourhood.values.flatten():
        colors = [
            Style.BACKGROUND,
            Style.REL_DOC,
            Style.REL_DOC_DUPE,
            Style.REL_DOC_NEW,
        ]
    elif 1 in neighbourhood.values.flatten():
        colors = [Style.BACKGROUND, Style.REL_DOC]
    else:
        colors = [Style.BACKGROUND]

    return colors


def label_neighbourhood(neighbourhood, original_ranking, qrels):
    df = pd.DataFrame(neighbourhood)
    width = len(df.columns)
    height = len(df.index)

    # The cells are traversed based on manhattan distance, to find first occurences of new documents
    index_pairs = [(i, j) for i in range(height) for j in range(width)]
    index_pairs = sorted(
        index_pairs, key=lambda x: (abs(x[0]) + abs(x[1]), -x[1], -x[0])
    )

    qrel_dict = dict(zip(qrels["docno"], qrels["label"]))

    first_doc = []
    for i, j in index_pairs:
        docno = df.iat[i, j]
        rel = qrel_dict.get(docno, 0)
        label = 0
        if rel > 0:
            label += 1
            if docno not in original_ranking:
                label += 1
                if docno not in first_doc:
                    label += 1
                    first_doc.append(docno)

        df.iat[i, j] = label
    df = df.astype(int)
    return df


def update_stats(neighbourhood, original_ranking, qrels):
    neighbourhood = pd.DataFrame(neighbourhood)
    rel_docs = qrels["docno"].to_list()
    neighbours = neighbourhood.iloc[:, 1:].copy().values.flatten()

    stats = {}

    # Compute Recall
    original_recall = get_recall(rel_docs, original_ranking)
    stats["original_recall"] = original_recall

    neighbour_recall = get_recall(rel_docs, neighbours)
    stats["neighbour_recall"] = neighbour_recall

    total_recall = get_recall(rel_docs, set(original_ranking).union(neighbours))
    stats["total_recall"] = total_recall

    # Compute Document Counts
    docno_counts = pd.Series(neighbours).value_counts().to_dict()
    rel_docs_counts = {}
    is_new_document = {}
    for docno in rel_docs:
        if docno in docno_counts.keys():
            count = docno_counts[docno]
            rel_docs_counts[docno] = count
            is_new_document[docno] = docno not in original_ranking

    stats["rel_docs_counts"] = rel_docs_counts
    stats["is_new_document"] = is_new_document

    return stats


def get_recall(rel_docs, docs):
    return float(len(set(rel_docs).intersection(docs)) / len(set(rel_docs)))


def create_label(stats_label, title, variable):
    stats_label.insert(tk.END, f"{title}: ", Style.TITLE)
    stats_label.insert(tk.END, f"{variable}\n", Style.VARIABLE)
