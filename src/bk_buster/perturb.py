import random
from typing import Callable

import networkx as nx
import numpy as np


def get_delta(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Calculate the delta between two graphs. Delta is defined as the number of different edges divided by the maximum
    number of different edges. Delta is meaningful for rewired graphs.
    Parameters:
        G1 (nx.Graph): The first graph.
        G2 (nx.Graph): The second graph.
    Returns:
        float: The delta between the two graphs.
    """

    # this seems to be faster than nx.symmetric_difference as it does not create a new graph (~0.2s on CPDB)
    num_diff_edges = len(set(G1.edges()) ^ set(G2.edges()))
    max_diff_edges = 2 * G1.number_of_edges()
    delta = num_diff_edges / max_diff_edges
    return delta


def get_aspl(G: nx.Graph) -> float:
    """
    Calculate the average shortest path length (ASPL) of a graph. Supports disconnected graphs as well.
    There are different ways to calculate the ASPL for disconnected graphs, some of them can be listed as:
    - Calculate ASPL for each component and return the average.
    - Calculate ASPL for each component with at least n nodes and return the average.
    - Ignore disconnected components and calculate ASPL only for the largest component.
    This function implements the second option with n=2 to discard isolated nodes.
    Parameters:
        G (nx.Graph): The input graph.
    Returns:
        float: The average shortest path length of the graph.
    """

    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        avg_path_lengths = []
        for component in list(nx.connected_components(G)):
            if len(component) < 2:
                continue
            subgraph = G.subgraph(component)
            avg_path_lengths.append(nx.average_shortest_path_length(subgraph))
        return np.mean(avg_path_lengths)


def is_swap_valid(G: nx.Graph, e1: tuple, e2: tuple) -> bool:
    """
    Check if the double edge swap is valid. A double edge swap is valid if:
    - Both edges exist in the graph.
    - The new edges do not already exist in the graph.
    - The edges do not share any nodes.
    Parameters:
        G (nx.Graph): The input graph.
        e1 (tuple): The first edge to swap.
        e2 (tuple): The second edge to swap.
    Returns:
        bool: True if the swap is valid, False otherwise.
    """

    # one or both edges do not exist in the graph
    if not G.has_edge(*e1) or not G.has_edge(*e2):
        return False

    # the new edges already exist in the graph
    if G.has_edge(e1[0], e2[0]) or G.has_edge(e1[1], e2[1]):
        return False

    # edges share a node
    if len({*e1, *e2}) < 4:
        return False

    return True


def do_edge_swap(
    G: nx.Graph,
    e1: tuple,
    e2: tuple,
    check_validity: bool = True,
    keep_weights: bool = True,
) -> None:
    """
    Perform a double edge swap on the graph. Swap validity check is optional. Keeps the weights of the edges by default.
    Parameters:
        G (nx.Graph): The input graph.
        e1 (tuple): The first edge to swap.
        e2 (tuple): The second edge to swap.
        check_validity (bool, optional): If True, check if the swap is valid. Default is True.
        keep_weights (bool, optional): If True, keep the weights of the edges. Default is True.
    Raises:
        ValueError: If the edge swap is not valid.
    """

    if check_validity:
        if not is_swap_valid(G, e1, e2):
            raise ValueError("Edge swap is not valid.")

    w1, w2 = None, None
    if keep_weights:
        if nx.is_weighted(G, edge=e1):
            w1 = G[e1[0]][e1[1]]["weight"]
        if nx.is_weighted(G, edge=e2):
            w2 = G[e2[0]][e2[1]]["weight"]

    # remove the edges from the graph
    G.remove_edge(*e1)
    G.remove_edge(*e2)

    # add the new edges to the graph
    if w1 is not None:
        G.add_edge(e1[0], e2[0], weight=w1)
    else:
        G.add_edge(e1[0], e2[0])

    if w2 is not None:
        G.add_edge(e1[1], e2[1], weight=w2)
    else:
        G.add_edge(e1[1], e2[1])


# random perturbations --------------------------------------------


def add_random_edges(
    G: nx.Graph, n_edges: int, inplace: bool = False, seed: int | None = None
) -> nx.Graph:
    """
    Add a specified number of random edges to a graph.
    Parameters:
        G (nx.Graph): The input graph to which edges will be added.
        n_edges (int): The number of edges to add.
        inplace (bool, optional): If True, add edges to the original graph. If False, create a copy of the graph with
            the new edges. Default is False.
        seed (int | None, optional): Seed for the random number generator to ensure reproducibility. Default is None.
    Returns:
        nx.Graph: The graph with the added edges. If inplace is True, returns the original graph with new edges.
            Otherwise, returns a new graph with the added edges.
    Raises:
        ValueError: If the number of edges to add exceeds the number of non-edges in the graph.
    """

    # get all non-edges in the graph
    non_edges = list(nx.non_edges(G))
    if n_edges > len(non_edges):
        raise ValueError(
            f"Cannot add {n_edges} edges to a graph with {len(non_edges)} non-edges"
        )

    # randomly select n_edges non-edges to add
    rng = np.random.default_rng(seed)
    edges_to_add = rng.choice(non_edges, n_edges, replace=False)

    # add edges to the graph
    G_new = G if inplace else G.copy()
    G_new.add_edges_from(edges_to_add)
    return G_new


def remove_random_edges(
    G: nx.Graph, n_edges: int, inplace: bool = False, seed: int | None = None
) -> nx.Graph:
    """
    Remove random edges from a graph.
    Parameters:
        G (nx.Graph): The input graph from which edges will be removed.
        n_edges (int): The number of edges to remove.
        inplace (bool, optional): If True, remove edges from the original graph. If False, create a copy of the graph
            with the specified edges removed. Default is False.
        seed (int | None, optional): Seed for the random number generator to ensure reproducibility. Default is None.
    Returns:
        nx.Graph: The graph with the specified edges removed. If inplace is True, returns the original graph with
            edges removed. Otherwise, returns a new graph with the specified edges removed.
    Raises:
        ValueError: If the number of edges to remove exceeds the number of edges in the graph.
    """

    # check if the number of edges to remove is valid
    num_edges = G.number_of_edges()
    if n_edges > num_edges:
        raise ValueError(
            f"Cannot remove {n_edges} edges from a graph with {num_edges} edges"
        )

    # get a list of edges to remove
    rng = np.random.default_rng(seed)
    edges_to_remove = rng.choice(list(G.edges), n_edges, replace=False)

    # remove edges from the graph
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(edges_to_remove)
    return G_new


def isolate_random_nodes(
    G: nx.Graph, n_nodes: int, inplace: bool = False, seed: int | None = None
) -> nx.Graph:
    """
    Isolate (remove all edges) a specified number of random nodes in a graph.
    Parameters:
        G (nx.Graph): The input graph from which nodes will be isolated.
        n_nodes (int): The number of nodes to isolate.
        inplace (bool, optional): If True, isolate nodes in the original graph. If False, create a copy of the graph
            with the specified nodes isolated. Default is False.
        seed (int | None, optional): Seed for the random number generator to ensure reproducibility. Default is None.
    Returns:
        nx.Graph: The graph with the specified nodes isolated. If inplace is True, returns the original graph with
            nodes isolated. Otherwise, returns a new graph with the specified nodes isolated.
    Raises:
        ValueError: If the number of nodes to isolate exceeds the number of nodes in the graph.
    """

    # check if the number of nodes to isolate is valid
    num_nodes = G.number_of_nodes()
    if n_nodes > num_nodes:
        raise ValueError(
            f"Cannot isolate {n_nodes} nodes from a graph with {num_nodes} nodes"
        )

    G_init = G.copy()
    G_new = G if inplace else G.copy()

    # randomly select n_nodes nodes to isolate and get a list of edges to remove
    rng = np.random.default_rng(seed)
    nodes_to_isolate = rng.choice(list(G.nodes), n_nodes, replace=False)
    edges_to_remove = [
        (node, neighbor)
        for node in nodes_to_isolate
        for neighbor in list(G_init.neighbors(node))
    ]

    # isolate nodes in the graph
    G_new.remove_edges_from(edges_to_remove)

    return G_new


def add_noise_to_edge_weights(
    G: nx.Graph,
    noise_func: Callable[[float, int | None], float],
    inplace: bool = False,
    seed: int | None = None,
) -> nx.Graph:
    """
    Adds noise to the edge weights of a given graph.
    Parameters:
        G (nx.Graph): Input graph with weighted edges.
        noise_func (Callable[[float, int | None], float]): A function that takes a weight and returns a noisy weight.
            Optionally, the function can take a seed parameter for reproducibility.
        inplace (bool, optional): If True, update the edge weights in the original graph. If False, create a copy of the
            graph with updated weights. Default is False.
        seed (int | None, optional): Seed for the random number generator to ensure reproducibility. Default is None.
    Returns:
        nx.Graph: A new graph with updated weights.
    """

    G_new = G if inplace else G.copy()

    rng = np.random.default_rng(seed)

    # add noise to the edge weights
    for u, v, data in G_new.edges(data=True):
        if "weight" in data:
            original_weight = data["weight"]
            local_seed = rng.integers(0, 2**32 - 1) if seed is not None else None
            data["weight"] = noise_func(original_weight, seed=local_seed)
            # print(f"Edge ({u}, {v}): {original_weight} -> {data['weight']}")

    return G_new


# structure preserving perturbations --------------------------------------------


def node_degree_preserving_rewiring(
    G: nx.Graph,
    deltas: list[float],
    delta_check_freq: int = 100,
    n_max_attempts: int = 1000,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[list[nx.Graph], list[float]]:
    """
    Perform node degree preserving rewiring on a graph to achieve specified deltas.
    """

    G_new = G.copy()
    rng = np.random.default_rng(seed)

    deltas = sorted(deltas)
    delta_cp_index = 0
    Gs_new = []
    deltas_actual = []

    # perform double edge swaps to achieve the desired ratio
    n_attempts = 0
    delta_current = 0
    while delta_current < deltas[-1]:
        # randomly pick two edges without replacement
        edges = list(G_new.edges)
        (a, b), (c, d) = rng.choice(edges, 2)

        # perform swap if it is valid
        if is_swap_valid(G_new, (a, b), (c, d)):
            do_edge_swap(G_new, (a, b), (c, d), check_validity=False, keep_weights=True)

        n_attempts += 1
        if n_attempts % delta_check_freq == 0:
            delta_current = get_delta(G, G_new)
            if verbose:
                print(f"Attempt {n_attempts}, delta_current: {delta_current}")
            if delta_cp_index < len(deltas):
                if delta_current >= deltas[delta_cp_index]:
                    if verbose:
                        print(f"Achieved delta checkpoint: {delta_current}")
                    Gs_new.append(G_new.copy())
                    deltas_actual.append(delta_current)
                    delta_cp_index += 1

        # :(
        if n_attempts >= n_max_attempts:
            print(
                f"Could not achieve desired deltas after {n_attempts} attempts. Final delta: {delta_current}"
            )
            break

    if verbose:
        print(f"Achieved deltas {[f'{delta:.2f}' for delta in deltas_actual]} after {n_attempts} attempts")

    return Gs_new, deltas_actual


def node_degree_dist_preserving_rewiring(
    G: nx.Graph, inplace: bool = False, seed: int | None = None
) -> tuple[nx.Graph, float]:
    rng = np.random.default_rng(seed)
    for i in range(1000):
        # print(f"Attempt {i + 1}/1000")
        degree_sequence = rng.permutation([i[1] for i in G.degree])
        local_seed = int(rng.integers(1e9)) if seed is not None else None
        try:
            G_new = nx.random_degree_sequence_graph(
                degree_sequence, seed=local_seed, tries=10
            )
        except nx.NetworkXError:
            # print(f"Failed to generate graph at i={i}, retrying...")
            continue

        delta = len(nx.symmetric_difference(G, G_new).edges) / (G.number_of_edges() * 2)
        if delta < 0.9:
            # print(f"Graph generated at i={i}, delta: {delta}")
            continue
        break

    return G_new, delta


def randomly_relabel_nodes(
    G: nx.Graph, inplace: bool = False, seed: int | None = None
) -> tuple[nx.Graph, float]:
    G_new = G if inplace else G.copy()

    rng = np.random.default_rng(seed)
    mapping = {
        node: new_node for node, new_node in zip(G.nodes(), rng.permutation(G.nodes()))
    }
    G_new = nx.relabel_nodes(G_new, mapping)

    delta = len(nx.symmetric_difference(G, G_new).edges) / (G.number_of_edges() * 2)

    return G_new, delta


def do_swaps(
    G: nx.Graph,
    delta: list[float],
    n_swaps_per_attempt: int = 1,
    n_max_attempts: int = 1000,
    what_to_preserve: str = "aspl",
    tolerance: float = 0.01,
    inplace: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[list[nx.Graph], list[float], list[float]]:
    """
    TBA
    """
    # TODO: add a global batch log not to undo swaps

    if what_to_preserve == "aspl":
        # measure_fn = nx.average_shortest_path_length
        measure_fn = get_aspl  # this custom aspl works for disconnected graphs as well
    elif what_to_preserve == "acc":
        measure_fn = nx.average_clustering
    elif what_to_preserve == "transitivity":
        measure_fn = nx.transitivity
    else:
        raise ValueError(f"Unknown measure to preserve: {what_to_preserve}")

    if verbose:
        print(f"Preserving {what_to_preserve} with delta:{delta}")

    # calculate the initial measure
    m_init = measure_fn(G)
    if verbose:
        print(f"Initial {what_to_preserve}: {m_init:.4f}, max abs diff: {(m_init * tolerance):.4f}")

    G_new = G.copy()
    G_new.__networkx_cache__ = None  # do not cache this graph as it will be changed
    rng = np.random.default_rng(seed)

    delta = sorted(delta)
    delta_cp_index = 0
    Gs_new = []
    delta_actual = []
    m_vals = []
    # m_vals.append(m_init)

    n_attempts = 0
    n_accepted_swaps = 0
    delta_current = 0
    while delta_current < delta[-1]:
        # a number of unique swaps are collected and applied
        added_edges = set()
        removed_edges = set()
        n_swaps_inner = 0
        while n_swaps_inner < n_swaps_per_attempt:
            edges = list(G_new.edges)

            # randomly pick two edges without replacement
            (a, b), (c, d) = rng.choice(edges, 2)

            # ensure edges are not already added to the batch collection
            if (a, b) in added_edges or (c, d) in added_edges:
                continue

            # check swap validity
            if not is_swap_valid(G_new, (a, b), (c, d)):
                continue

            # at this point, the swap should be valid: remove old edges and add new ones
            do_edge_swap(G_new, (a, b), (c, d), check_validity=False, keep_weights=True)

            # add the edges to the batch collection
            added_edges.add((a, c))
            added_edges.add((b, d))
            removed_edges.add((a, b))
            removed_edges.add((c, d))
            n_swaps_inner += 1

        # measure the deviation after swaps
        m_new = measure_fn(G_new)
        m_diff = abs(m_new - m_init) / abs(m_init)

        # if the deviation is within the tolerance, accept the swaps
        if m_diff <= tolerance:
            # recalculate delta
            n_accepted_swaps += n_swaps_per_attempt
            # delta_current = len(nx.symmetric_difference(G, G_new).edges) / (n_edges * 2)
            # delta_current = (n_accepted_swaps * 4) / (n_edges * 2)  # this is a cheaper way but not always accurate
            delta_current = get_delta(G, G_new)

            # check if a delta checkpoint is reached, if so, store the graph, delta and m
            if delta_cp_index < len(delta):
                if delta_current >= delta[delta_cp_index]:
                    if verbose:
                        print(f"  Achieved delta checkpoint: delta:{delta_current:.4f}, m: {m_new:.4f}")
                    Gs_new.append(G_new.copy())
                    delta_actual.append(delta_current)
                    m_vals.append(m_new)
                    delta_cp_index += 1
                if delta_cp_index >= len(delta):
                    if verbose:
                        print("  All deltas are reached, terminating...")
                    break  # all deltas are reached

        # revert changes if deviation is too much
        else:
            for edge in removed_edges:
                G_new.add_edge(*edge)
            for edge in added_edges:
                G_new.remove_edge(*edge)

        n_attempts += 1
        if verbose:
            print(f"Attempt {n_attempts}/{n_max_attempts}, accepted swaps:{n_accepted_swaps}, delta_cur:{delta_current:.4f}")

        # check if max attempts are reached
        if n_attempts >= n_max_attempts:
            if verbose:
                print(f"Reached max attempts, achieved deltas: {delta_actual}")
            break

    return Gs_new, delta_actual, m_vals


def find_gm_candidate(
    G: nx.Graph, n_trials: int = 5, seed: int | None = None
) -> tuple[int, int, int, int] | None:
    """
    Finds a 4-node subgraph that can undergo a GM switch. Returns (a, c, b, d) if found, else None.
    """
    edges = list(G.edges)
    edge_set = set(edges)  # Use set for O(1) edge existence checks

    rng = np.random.default_rng(seed)

    for _ in range(n_trials):
        rng.shuffle(edges)  # Randomize edge order
        for a, c in edges:
            for b, d in edges:
                if a != b and c != d and len({a, b, c, d}) == 4:
                    if (a, d) not in edge_set and (b, c) not in edge_set:
                        return a, c, b, d  # Found valid switchable structure

    return None  # No valid switch found


def godsil_mckay_switch(
    G: nx.Graph, n_trials: int = 5, inplace: bool = False, seed: int | None = None
) -> nx.Graph:
    """
    GM switch using precomputed edge pairs. Hopefully works efficiently on large graphs.
    """

    G_new = G.copy()
    rng = np.random.default_rng(seed)

    candidate = find_gm_candidate(G_new, n_trials=5, seed=rng.integers(0, 2**32 - 1))
    if candidate is None:
        print("No GM switchable structure found.")
        return None

    # Perform the GM switch
    a, c, b, d = candidate
    G_new.remove_edge(a, c)
    G_new.remove_edge(b, d)
    G_new.add_edge(a, d)
    G_new.add_edge(b, c)

    return G_new


def spectrum_preserving_rewiring(
    G: nx.Graph,
    edited_edge_ratio: int = 0.1,
    n_trials: int = 1000,
    inplace: bool = False,
    seed: int | None = None,
) -> nx.Graph:
    # check if the edited_edge_ratio is valid
    if edited_edge_ratio < 0 or edited_edge_ratio > 1:
        raise ValueError("edited_edge_ratio must be in [0, 1]")

    rng = np.random.default_rng(seed)

    G_init = G.copy()
    G_new = G if inplace else G.copy()
    num_edges = G_new.number_of_edges()
    current_ratio = 0
    num_trials = 0
    while current_ratio < edited_edge_ratio:
        # perform GM switch
        local_seed = rng.integers(0, 2**32 - 1) if seed is not None else None
        G_new = godsil_mckay_switch(
            G=G_new, n_trials=n_trials, inplace=inplace, seed=local_seed
        )

        # compute the ratio of edited edges
        current_ratio = len(nx.symmetric_difference(G_init, G_new).edges) / (
            num_edges * 2
        )
        # print(f"Current ratio: {current_ratio}")

        num_trials += 1
        if num_trials >= n_trials:
            raise ValueError(
                f"Could not achieve desired ratio after {num_trials} trials"
            )

    # compute the actual ratio of edited edges
    actual_ratio = len(nx.symmetric_difference(G_init, G_new).edges) / (num_edges * 2)
    print(f"Desired ratio: {edited_edge_ratio}")
    print(f"Actual ratio: {actual_ratio}")

    return G_new


def community_preserving_rewiring(
    G: nx.Graph,
    n_swaps: int,
    community_structure: list,
    n_trials: int = 1000,
    inplace: bool = False,
    seed: int | None = None,
) -> nx.Graph:
    G_new = G.copy()

    rng = np.random.default_rng(seed)

    for i, community in enumerate(community_structure):
        print(f"Community {i}: {community}")

        # print(f"Edges in community {i}: {edge_list}")
        # if len(edge_list) < 2:
        #     print(f"Not enough edges in community {i} to perform swaps.")
        #     continue

        for _ in range(n_swaps):
            attempts = 0
            while attempts < n_trials:
                edge_list = [
                    (u, v)
                    for u in community
                    for v in community
                    if u != v and G_new.has_edge(u, v)
                ]
                # Randomly pick two distinct edges
                (a, b), (c, d) = rng.choice(edge_list, 2)
                # Ensure new edges do not create self-loops or duplicates
                if (
                    len({a, b, c, d}) < 4
                    or G_new.has_edge(a, c)
                    or G_new.has_edge(b, d)
                ):
                    attempts += 1
                    continue

                # Remove old edges and add new ones
                G_new.remove_edge(a, b)
                G_new.remove_edge(c, d)
                G_new.add_edge(a, c)
                G_new.add_edge(b, d)

                # Check if the graph is still connected
                if not nx.is_connected(G):
                    G_new.remove_edge(a, c)
                    G_new.remove_edge(b, d)
                    G_new.add_edge(a, b)
                    G_new.add_edge(c, d)
                    attempts += 1
                    continue
                else:
                    attempts += 1
                    break

    return G_new


# targeted perturbations --------------------------------------------


def isolate_high_degree_nodes(
    G: nx.Graph, n_nodes: int, inplace: bool = False
) -> nx.Graph:
    # check if the number of nodes to isolate is valid
    if n_nodes > G.number_of_nodes():
        raise ValueError(
            f"Cannot isolate {n_nodes} nodes from a graph with {G.number_of_nodes()} nodes"
        )

    # get the n_nodes nodes with the highest degree
    high_degree_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:n_nodes]
    # print(high_degree_nodes)

    # get a list of edges to remove
    edges_to_remove = [
        (node, neighbor)
        for node, _ in high_degree_nodes
        for neighbor in list(G.neighbors(node))
    ]

    # isolate nodes in the graph
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(edges_to_remove)

    return G_new


def isolate_low_degree_nodes(
    G: nx.Graph, n_nodes: int, inplace: bool = False
) -> nx.Graph:
    # check if the number of nodes to isolate is valid
    if n_nodes > G.number_of_nodes():
        raise ValueError(
            f"Cannot isolate {n_nodes} nodes from a graph with {G.number_of_nodes()} nodes"
        )

    # get the n_nodes nodes with the lowest degree (excluding isolated nodes)
    degree_dict = {node: deg for node, deg in G.degree() if deg > 0}
    low_degree_nodes = sorted(degree_dict, key=degree_dict.get)[:n_nodes]
    # print(low_degree_nodes)

    # get a list of edges to remove
    edges_to_remove = [
        (node, neighbor)
        for node in low_degree_nodes
        for neighbor in list(G.neighbors(node))
    ]

    # isolate nodes in the graph
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(edges_to_remove)

    return G_new


def isolate_nodes_with_degree(
    G: nx.Graph,
    degree: int,
    n_nodes: int | None = None,
    inplace: bool = False,
    seed: int | None = None,
) -> nx.Graph:
    # get the nodes with the specified degree
    nodes_with_degree = [node for node, deg in G.degree() if deg == degree]
    print(f"Nodes with degree {degree}: {nodes_with_degree}")

    # check if any nodes with the specified degree exist
    if len(nodes_with_degree) == 0:
        raise ValueError(f"No nodes with degree {degree} found in the graph")

    # select n_nodes nodes to isolate (if n_nodes is None, isolate all nodes with the specified degree)
    if n_nodes is None or n_nodes >= len(nodes_with_degree):
        nodes_to_isolate = nodes_with_degree
    else:
        rng = np.random.default_rng(seed)
        nodes_to_isolate = rng.choice(nodes_with_degree, n_nodes, replace=False)
    print(f"Isolating nodes: {nodes_to_isolate}")

    # get a list of edges to remove
    edges_to_remove = [
        (node, neighbor)
        for node in nodes_to_isolate
        for neighbor in list(G.neighbors(node))
    ]

    # isolate nodes in the graph
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(edges_to_remove)

    return G_new


def isolate_high_betweenness_nodes(
    G: nx.Graph, n_nodes: int, inplace: bool = False, seed: int | None = None
) -> nx.Graph:
    # compute node betweenness centrality
    node_betweenness = nx.betweenness_centrality(G)
    high_betweenness_nodes = sorted(
        node_betweenness, key=node_betweenness.get, reverse=True
    )[:n_nodes]
    print(high_betweenness_nodes)

    # get a list of edges to remove
    edges_to_remove = [
        (node, neighbor)
        for node in high_betweenness_nodes
        for neighbor in list(G.neighbors(node))
    ]

    # isolate nodes in the graph
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(edges_to_remove)

    return G_new


def remove_high_betweenness_edges(
    G: nx.Graph, n_edges: int, inplace: bool = False, seed: int | None = None
) -> nx.Graph:
    # compute edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    high_betweenness_edges = sorted(
        edge_betweenness, key=edge_betweenness.get, reverse=True
    )[:n_edges]
    print(high_betweenness_edges)

    # remove high betweenness edges
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(high_betweenness_edges)

    return G_new


def spectral_distortion_edge_importance(G: nx.Graph) -> dict:
    eigenvalues = nx.laplacian_spectrum(G)

    importance = {}

    for u, v in G.edges():
        # Perturb the graph by removing edge (u, v)
        G_perturbed = G.copy()
        G_perturbed.remove_edge(u, v)

        # Compute new spectrum
        perturbed_eigenvalues = nx.laplacian_spectrum(G_perturbed)

        # Compute spectral distortion (sum of absolute eigenvalue changes)
        distortion = np.sum(np.abs(eigenvalues - perturbed_eigenvalues))

        importance[(u, v)] = distortion

    # Sort edges by importance
    importance = dict(
        sorted(importance.items(), key=lambda item: item[1], reverse=True)
    )

    return importance


def remove_high_spectral_distortion_edges(
    G: nx.Graph,
    n_edges: int,
    inplace: bool = False,
) -> nx.Graph:
    # compute edge importance
    edge_importance = spectral_distortion_edge_importance(G)
    high_importance_edges = list(edge_importance.keys())[:n_edges]
    print(high_importance_edges)

    # remove high importance edges
    G_new = G if inplace else G.copy()
    G_new.remove_edges_from(high_importance_edges)

    return G_new


def add_inter_community_edges(
    G: nx.Graph,
    n_edges: int,
    community_structure: list,
    inplace: bool = False,
    seed: int | None = None,
) -> nx.Graph:
    G_new = G if inplace else G.copy()

    rng = random.Random(42)

    added_edges = 0
    while added_edges < n_edges:
        # randomly select two distinct communities
        comm1, comm2 = rng.sample(community_structure, 2)
        # print(f"Communities selected: {comm1}, {comm2}")

        # randomly select nodes from each community
        node1 = rng.choice(list(comm1))
        node2 = rng.choice(list(comm2))

        # add edge if it doesn't already exist
        if not G_new.has_edge(node1, node2):
            G_new.add_edge(node1, node2)
            added_edges += 1

    # print(f"Added {added_edges} edges between communities.")

    return G_new
