import numpy as np
import h5py
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import seaborn as sns

from . import util
from .util import Style


class NeighbourViewer(tk.Tk):
    def __init__(self, file_path, dataset, min_rel):
        super().__init__()

        # Initialise parameters
        self.dataset = h5py.File(file_path, "r")
        topics = dataset.get_topics()
        self.topics = topics.set_index("qid")["query"].to_dict()
        self.qids = topics["qid"].tolist()
        self.qrels = dataset.get_qrels()
        self.min_rel = min_rel

        # Initial config
        self.current_index = 0
        self.num_rows = 100
        self.num_columns = 16

        # Create interface
        self.title("Neighbourhood viewer")
        self.geometry("1600x900")
        self.create_frames()
        self.create_control_panel()
        self.update_interface()

    def update_interface(self):
        neighbourhood, stats = self._update_display()
        self.display_neighbourhood(neighbourhood)
        self.display_stats(stats)

    def next_query(self):
        if len(self.qids) > self.current_index + 1:
            self.current_index = self.current_index + 1
            self.update_interface()

    def prev_query(self):
        if self.current_index - 1 >= 0:
            self.current_index = self.current_index - 1
            self.update_interface()

    def update_row_columns(self):
        self.num_rows = int(self.row_entry.get())
        if self.num_rows < 1:
            self.num_rows = 1
        if self.num_rows > self._get_num_rows():
            self.num_rows = self._get_num_rows()
        self.row_entry.delete(0, tk.END)
        self.row_entry.insert(0, self.num_rows)

        self.num_columns = int(self.column_entry.get())
        if self.num_columns < 1:
            self.num_columns = 1
        if self.num_columns > self._get_num_columns():
            self.num_columns = self._get_num_columns()
        self.column_entry.delete(0, tk.END)
        self.column_entry.insert(0, self.num_columns)

        self.update_interface()

    def _update_display(self):
        qid = self.qids[self.current_index]

        # Gather Selected Data
        neighbourhood = self.dataset[qid][: self.num_rows, : self.num_columns].astype(
            "U21"
        )
        qrels = self.qrels[
            (self.qrels["qid"] == qid) & (self.qrels["label"] >= self.min_rel)
        ]
        original_ranking = neighbourhood[:, 0].tolist()

        # Update the neighbourhood and stats
        labeled_neighbourhood = util.label_neighbourhood(
            neighbourhood, original_ranking, qrels
        )
        stats = util.update_stats(neighbourhood, original_ranking, qrels)

        return labeled_neighbourhood, stats

    def _get_num_rows(self):
        qid = self.qids[self.current_index]
        return self.dataset[qid][:].shape[0]

    def _get_num_columns(self):
        qid = self.qids[self.current_index]
        return self.dataset[qid][:].shape[1]

    def create_frames(self):
        # Frame for the heatmap
        self.heatmap_frame = tk.Frame(self)
        self.heatmap_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas = None

        # Frame for the stats
        self.table_frame = tk.Frame(self)
        self.table_frame.grid(row=0, column=1, sticky="nsew")
        self.stats_label = tk.Text(
            self.table_frame, height=20, wrap="none", padx=10, pady=10
        )
        self.stats_label.pack(fill="both", expand=True)
        self._configure_style()

        # Control panel
        self.footer_frame = tk.Frame(self)
        self.footer_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

    def create_control_panel(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Previous and Next buttons in footer
        self.prev_button = tk.Button(
            self.footer_frame, text="Previous", command=self.prev_query
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(
            self.footer_frame, text="Next", command=self.next_query
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Input field for row selection in footer
        self.row_label = tk.Label(self.footer_frame, text="Rows:")
        self.row_label.pack(side=tk.LEFT, padx=5)

        self.row_entry = tk.Entry(self.footer_frame, width=5)
        self.row_entry.insert(0, str(self.num_rows))
        self.row_entry.pack(side=tk.LEFT, padx=5)

        # Input field for column selection in footer
        self.column_label = tk.Label(self.footer_frame, text="Columns:")
        self.column_label.pack(side=tk.LEFT, padx=5)

        self.column_entry = tk.Entry(self.footer_frame, width=5)
        self.column_entry.insert(0, str(self.num_columns))
        self.column_entry.pack(side=tk.LEFT, padx=5)

        # Update row and column
        self.update_button = tk.Button(
            self.footer_frame, text="Update", command=self.update_row_columns
        )
        self.update_button.pack(side=tk.LEFT, padx=5)

    def display_neighbourhood(self, neighbourhood):
        # Create heatmap plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = util.get_color_scale(neighbourhood)
        cmap = ListedColormap(colors)
        sns.heatmap(neighbourhood, cmap=cmap, ax=ax, cbar=False)
        ax.set_ylabel("Documents")
        ax.set_xlabel("Neighbours")

        x_ticks = np.arange(0, neighbourhood.shape[1], 10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha="right")

        y_ticks = np.arange(0, neighbourhood.shape[0], 25)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)

        legend_elements = [
            Patch(facecolor=Style.REL_DOC_NEW, label="New Relevant Document"),
            Patch(facecolor=Style.REL_DOC_DUPE, label="Duplicate Relevant Document"),
            Patch(facecolor=Style.REL_DOC, label="Relevant Document"),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.08),
            frameon=True,
        )

        # Add plot to canvas
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        matplotlib.pyplot.close(fig)

    def display_stats(self, stats):
        self.stats_label.delete("1.0", tk.END)

        qid = self.qids[self.current_index]
        original_recall = stats["original_recall"]
        nb_recall = stats["neighbour_recall"]
        combined_recall = stats["total_recall"]

        self._create_label("Query ID", qid)
        self._create_label("Query", self.topics[qid])
        self._create_label("Original Recall", original_recall)
        self._create_label("Neighbour Recall", nb_recall)
        self._create_label("Total Recall", combined_recall)

        rel_docs_counts = stats["rel_docs_counts"]
        is_new_document = stats["is_new_document"]

        sorted_keys = sorted(
            rel_docs_counts, key=lambda x: rel_docs_counts[x], reverse=True
        )

        self.stats_label.insert(tk.END, f"\nDocuments:\n", Style.TITLE)
        for docno in sorted_keys:
            self.stats_label.insert(tk.END, "Docno ")
            self.stats_label.insert(
                tk.END,
                f"{docno}",
                (Style.REL_DOC_NEW if is_new_document[docno] else Style.REL_DOC),
            )
            self.stats_label.insert(tk.END, f" occurs ")
            self.stats_label.insert(tk.END, f"{rel_docs_counts[docno]}", Style.VARIABLE)
            self.stats_label.insert(tk.END, f" times amongst neighbours\n")

    def _configure_style(self):
        self.stats_label.configure(
            background="gray10", foreground="white", font=("Arial", 12)
        )
        self.stats_label.tag_configure("bold", font=("Arial", 12, "bold"))
        self.stats_label.tag_configure("cyan", foreground="cyan")

        self.stats_label.tag_configure("lightgray", foreground="lightgray")
        self.stats_label.tag_configure("lime", foreground="lime")
        self.stats_label.tag_configure("lightcoral", foreground="light coral")

    def _create_label(self, title, variable):
        self.stats_label.insert(tk.END, f"{title}: ", Style.TITLE)
        self.stats_label.insert(tk.END, f"{variable}\n", Style.VARIABLE)
