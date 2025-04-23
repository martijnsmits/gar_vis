# gar_vis
Visualisation tool for [Corpus Graphs](https://arxiv.org/abs/2208.08942)

## Installation
Install the package via `pip`:
```bash
pip install gar-vis
```

## Getting Started

Run the following code for a simple demo:
```bash
import pyterrier as pt
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex

from gar_vis import GarVis

if __name__ == "__main__":
    corpus_graph = CorpusGraph.from_hf("macavaney/msmarco-passage.corpusgraph.bm25.128")
    bm25 = PisaIndex.from_dataset("msmarco_passage").bm25()
    dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-2019/judged")

    file_path = GarVis.create_neighbourhood(corpus_graph, bm25, dataset)
    GarVis.visualise_neighbourhood(file_path, dataset, min_rel=2)
```

Or use it with a different corpus graph, retriever and/or dataset (including qrels):
```python
import pyterrier as pt
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex

from gar_vis import GarVis

if __name__ == "__main__":
    corpus_graph = ...
    retriever = ...
    dataset = ...

    # Create a new neighbourhood
    file_path = GarVis.create_neighbourhood(corpus_graph, bm25, dataset, k=num_neighbours, run_id="file_name", save_dir = "path_to_file")

    # Or use an existing neighbourhood
    file_path = "path_to_file/file_name.h5"

    # And start the visulisation tool (min_rel is used to set the minimum relevance label to consider the document relevant)
    GarVis.visualise_neighbourhood(file_path, dataset, min_rel=2)
```

## Citation
```bibtex
@inproceedings{rear2025,
    title = {Resource Efficient Adaptive Retrieval},
    author = {Martijn Smits},
    year = {2025},
}
```