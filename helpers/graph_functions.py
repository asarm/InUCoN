import networkx as nx
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")


def build_graph(similarities, col_list, textual_similarities, stock2id, threshold_metric="mean",
                ts_w=0.55, text_w=0.45, threshold=None):
    graph1 = nx.Graph()
    graph1.add_nodes_from(col_list)

    for edge in list(similarities.keys()):
        # time-series için distance hesapladık, text için similarity hesapladık. Bunu da distance'a çeviriyoruz
        try:
            text_dist = 1 - textual_similarities[stock2id.loc[edge[0]].item()][
                stock2id.loc[edge[1]].item()]
        except:
            id1 = np.where(stock2id.T.columns == edge[0])[0].item()
            id2 = np.where(stock2id.T.columns == edge[1])[0].item()

            text_dist = 1 - textual_similarities[id1][id2]

        ts_dist = similarities[edge]

        graph1.add_edge(
                edge[0], edge[1], weight=(ts_w * ts_dist + text_w * text_dist)
            )

    # remove self loops
    graph1.remove_edges_from(nx.selfloop_edges(graph1))
    edge_weights = [graph1[edge[0]][edge[1]]['weight'] for edge in graph1.edges(data=True)]

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_weights = scaler.fit_transform(np.array(edge_weights).reshape(-1, 1)).flatten()

    for (i, (u, v)) in enumerate(graph1.edges()):
        graph1[u][v]['weight'] = normalized_weights[i]

    edge_weights = [graph1[edge[0]][edge[1]]['weight'] for edge in graph1.edges(data=True)]
    std = np.std(edge_weights)

    if threshold is None:
        if threshold_metric == "mean":
            threshold = np.mean(edge_weights)
        elif threshold_metric == "1_sigma":
            threshold = np.mean(edge_weights) - (1.0 * std)
        elif threshold_metric == "1_25_sigma":
            threshold = np.mean(edge_weights) - (1.25 * std)
        elif threshold_metric == "1_5_sigma" or threshold_metric == "custom":
            threshold = np.mean(edge_weights) - (1.5 * std)
            # print("threshold:", threshold, "mean:", np.mean(edge_weights), "std:", std)

    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(col_list)
    filtered_edges = [(u, v, w['weight']) for u, v, w in graph1.edges(data=True) if w['weight'] < threshold]

    for edge in filtered_edges:
        filtered_graph.add_edge(
            edge[0], edge[1], weight=edge[2]
        )

    return filtered_graph


def calculate_graph_metrics(g):
    degrees = dict(g.degree()).values()
    avg_degree = np.mean(list(degrees))
    std_degree = np.std(list(degrees))

    avg_density = nx.density(g)
    avg_transitivity = nx.transitivity(g)

    clustering = nx.clustering(g)
    avg_clustering = np.mean(list(clustering.values()))
    std_clustering = np.std(list(clustering.values()))

    metrics = {"avg_degree": avg_degree, "density": avg_density, "transitivity": avg_transitivity,
               "avg_clustering": avg_clustering}
    stds = {"std_degree": std_degree, "std_clustering": std_clustering}

    return metrics, stds


def get_clique_intersection(graph: nx.Graph):
    cliques = nx.find_cliques(graph)

    lens = []
    for c in cliques:
        lens.append(len(c))

    # clique_threshold = int(np.mean(lens) + np.std(lens))
    clique_threshold = int(np.max(lens))
    cliques = nx.find_cliques(graph)
    clique_elements = []
    for c in cliques:
        if len(c) >= clique_threshold:
            clique_elements.append(np.array(c))

    intersections = clique_elements[0]
    for arr in clique_elements[1:]:
        intersections = np.intersect1d(intersections, arr)

    return intersections

def get_all_clique_items(graph: nx.Graph):
    cliques = nx.find_cliques(graph)
    clique_lens = []
    for c in cliques:
        clique_lens.append(len(c))

    # clique_threshold = int(np.mean(clique_lens) + np.std(clique_lens))
    clique_threshold = np.max(clique_lens)
    cliques = nx.find_cliques(graph)
    clique_elements = set()
    for c in cliques:
        if len(c) == clique_threshold:
            for item in c:
                clique_elements.add(item)

    return clique_elements