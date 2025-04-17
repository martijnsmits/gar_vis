# gar_vis
Visualisation tool for [Corpus Graphs](https://arxiv.org/abs/2208.08942)

## Getting Started
Clone the respository and install the requirements:
```bash
git clone https://github.com/martijnsmits/gar_vis.git
pip install -r requirements.txt
```

Run main for a simple demo.
```bash
python -m main
```

Or use it with a different corpus graph, retriever and/or dataset (including qrels):
```python
import pyterrier as pt
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex

from create_neighbourhood import create_neighbourhood
from neighbour_viewer import NeighbourViewer

if __name__ == "__main__":
    corpus_graph = ...
    retriever = ...
    dataset = ...

    # Create a new neighbourhood
    file_path = create_neighbourhood(corpus_graph, retriever, dataset, k=num_neighbours, run_id="file_name", save_dir = "path_to_file")

    # Or use an existing neighbourhood
    file_path = "path_to_file/file_name.h5"

    # And start the visulisation tool (min_rel )
    app = NeighbourViewer(file_path, dataset, min_rel=2)
    app.mainloop()
```

## Citation
```bibtex
    TODO
```