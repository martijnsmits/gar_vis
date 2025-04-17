import numpy as np
import h5py
import pyterrier as pt
from pyterrier_adaptive import CorpusGraph


def create_neighbourhood(
    corpus_graph: CorpusGraph,
    retriever: pt.Transformer,
    dataset: pt.datasets.Dataset,
    k: int = 128,
    run_id: str = "default",
    save_dir: str = "neighbourhoods/",
):
    topics = dataset.get_topics()
    corpus_graph = corpus_graph.to_limit_k(k)
    qids = topics["qid"].to_list()

    file_path = f"{save_dir}/{run_id}.h5"
    with h5py.File(file_path, "w") as fp:
        ranking = retriever.transform(topics)
        for qid in qids:
            neighbourhood = _append_neighbours(ranking, qid, corpus_graph, k)
            fp.create_dataset(qid, data=neighbourhood)

    return file_path


def _get_neighbours(docid, corpus_graph):
    neighbours = [docid] + [int(n) for n in corpus_graph.neighbours(docid)]
    return neighbours


def _append_neighbours(ranking, qid, corpus_graph, k):
    df = ranking[ranking["qid"] == qid]
    neighbourhood = np.array(
        [_get_neighbours(docno, corpus_graph) for docno in df["docno"]]
    ).astype("S21")
    return neighbourhood
