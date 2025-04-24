import os
import h5py
import pyterrier as pt
from pyterrier_adaptive import CorpusGraph

from .util import append_neighbours


class GarVis:
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

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = f"{save_dir}/{run_id}.h5"
        with h5py.File(file_path, "w") as fp:
            ranking = retriever.transform(topics)
            for qid in qids:
                neighbourhood = append_neighbours(ranking, qid, corpus_graph)
                fp.create_dataset(qid, data=neighbourhood)

        return file_path

    def visualise_neighbourhood(file_path, dataset, min_rel=1):
        # Import it here, so that Python versions without Tkinter don't need to install it
        from .neighbour_viewer import NeighbourViewer

        app = NeighbourViewer(file_path, dataset, min_rel=min_rel)
        app.mainloop()
