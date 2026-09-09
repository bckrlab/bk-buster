from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def generate_data(
        num_nodes_per_class,
        g1_dist_params=(0.5, 0.1),
        g2_dist_params=(0.8, 0.1),
        num_graphs=10,
        seed=42
    ):

    rng = np.random.RandomState(seed)

    # prepare data
    g1_mu, g1_sigma = g1_dist_params
    g2_mu, g2_sigma = g2_dist_params
    X = np.zeros((num_graphs, 2*num_nodes_per_class))
    X[:, :num_nodes_per_class] = rng.normal(g1_mu, g1_sigma, (num_graphs, num_nodes_per_class))
    X[:, num_nodes_per_class:] = rng.normal(g2_mu, g2_sigma, (num_graphs, num_nodes_per_class))
    
    # prepare labels
    y = np.zeros((num_graphs, 2*num_nodes_per_class), dtype=int)
    y[:, :num_nodes_per_class] = 0
    y[:, num_nodes_per_class:] = 1
    
    return X, y

def make_graph(X, y, edge_ops=None):

    # the number of nodes per class, which is the half (for binary classification) of the number of columns in X
    num_nodes = len(X) // 2

    # fully connected graph graph for each class
    index_tuples = list(permutations(range(0, num_nodes), 2)) + list(permutations(range(num_nodes, num_nodes*2), 2))

    # apply edge operations for contamination 
    if edge_ops is not None and len(edge_ops) > 0:
        for edge_op in edge_ops:
            if edge_op[0] == "add":
                index_tuples.append((int(edge_op[1]), int(edge_op[2])))
                index_tuples.append((int(edge_op[2]), int(edge_op[1])))
            elif edge_op[0] == "remove":
                index_tuples = [edge for edge in index_tuples if edge not in [
                    (int(edge_op[1]), int(edge_op[2])), (int(edge_op[2]), int(edge_op[1]))]
                ]

    data = Data(
        x=torch.tensor(X[:, np.newaxis], dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(index_tuples, dtype=torch.long).t().contiguous() if len(index_tuples) > 0 else torch.empty((2, 0), dtype=torch.long)
    )

    data.validate(raise_on_error=True)
    return data
    
def make_graphs_nx(X):
    num_nodes_per_class = X.shape[1] // 2
    graphs = []
    for x in X:
        g1 = nx.complete_graph(num_nodes_per_class)
        for i, node in enumerate(g1.nodes):
            g1.nodes[node]["x"] = x[i]
            g1.nodes[node]["y"] = 0
        for u, v in g1.edges:
            g1.edges[u, v]["weight"] = 1
        g2 = nx.complete_graph(num_nodes_per_class)
        for i, node in enumerate(g2.nodes):
            g2.nodes[node]["x"] = x[i + num_nodes_per_class]
            g2.nodes[node]["y"] = 1
        for u, v in g2.edges:
            g2.edges[u, v]["weight"] = 1
        graphs.append(nx.union(g1, g2, rename=("A", "B")))
    return graphs

def visualize_graph(
        data,
        node_colors=None,
        layout="circular",
        draw_node_labels=True,
        draw_borders=True,
        ax=None,
        kwargs={}
    ):
    graph = to_networkx(data, to_undirected=True, node_attrs=["x", "y"])
    font_color = "white"
    if node_colors is None:
        node_colors = data["x"]
        font_color = "black"
    # labels = {node: f'{data["x"][0]:.2f}' for node, data in graph.nodes(data=True)} if draw_node_labels else None
    labels = {node: f"{data["x"][0]:.2f}".replace("-0.", "-.").lstrip("0") for node, data in graph.nodes(data=True)} if draw_node_labels else None
    if layout == "circular":
        pos = nx.circular_layout(graph)
        theta = np.pi / 2
        rotated_pos = {
            node: (
                x * np.cos(theta) - y * np.sin(theta),
                x * np.sin(theta) + y * np.cos(theta)
            )
            for node, (x, y) in pos.items()
        }
        pos = rotated_pos
    elif layout == "spring":
        pos = nx.spring_layout(graph, k=1.5, seed=42)
    elif layout == "graphviz":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=draw_node_labels,
        labels=labels,
        node_color=node_colors,
        cmap=plt.cm.coolwarm,
        vmin=0,
        vmax=1,
        edge_color=kwargs.get("edge_color", "darkgray"),
        width=kwargs.get("edge_width", 0.5),
        node_size=kwargs.get("node_size", 500),
        font_color=font_color,
        font_size=kwargs.get("font_size", 10),
        font_weight="bold",
        node_shape="o",
        clip_on=False,
        ax=ax,
    )

    # draw borders: blue for class 0, red for class 1
    if draw_borders:
        palette = sns.color_palette("deep", n_colors=2)
        node_border_colors = [palette[0] if data["y"] == 0 else palette[1] for node, data in graph.nodes(data=True)]
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            node_color="none",
            edgecolors=node_border_colors,
            node_size=kwargs.get("node_size", 500) + 20,
            linewidths=kwargs.get("node_border_width", 1),
            ax=ax,
        )

    return ax