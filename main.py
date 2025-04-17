import pyterrier as pt
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex

from create_neighbourhood import create_neighbourhood
from neighbour_viewer import NeighbourViewer

if __name__ == "__main__":
    corpus_graph = CorpusGraph.from_hf("macavaney/msmarco-passage.corpusgraph.bm25.128")
    bm25 = PisaIndex.from_dataset("msmarco_passage").bm25()
    dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-2019/judged")

    file_path = create_neighbourhood(corpus_graph, bm25, dataset)

    app = NeighbourViewer(file_path, dataset, min_rel=2)
    app.mainloop()
