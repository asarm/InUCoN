import networkx as nx
import numpy as np
import pandas as pd
import itertools
import random
import igraph as ig

def get_non_edges(graph):
    non_edges = []
    nodes = set(graph.vs["name"])
    for node_pair in itertools.combinations(nodes, 2):
        if not graph.are_connected(*node_pair):
            non_edges.append(node_pair)
    return non_edges


def graph_consistency(g: ig.Graph, stock_list, p_h=0.2):
    A = np.array(g.get_adjacency().data)
    A_eig = np.linalg.eig(A)

    node_count = g.vcount()
    link_count = g.ecount()

    removed_link_count = int(link_count * p_h)
    links2remove = random.sample(list(g.get_edgelist()), k=removed_link_count)

    g_perturbated = g.copy()
    g_perturbated.delete_edges(links2remove)

    A_R = np.array(g_perturbated.get_adjacency().data)
    XLX = np.linalg.eig(A_R)
    dlt_A = A - A_R  # unremoved edges

    eigen = XLX[1]
    eigen_vals = XLX[0]
    eigen_t = eigen.T

    XLX_dlt_A = np.dot(eigen_t, dlt_A)
    XLXtimesEigen = np.dot(XLX_dlt_A, eigen_t)
    rowSum = np.sum(XLXtimesEigen, axis=1)
    colSum = np.sum(eigen * eigen, axis=0)

    dlt_l = rowSum / colSum
    A_perturbated = np.dot(eigen, np.dot(np.diag(eigen_vals + dlt_l), eigen.T))

    non_edges = get_non_edges(g_perturbated)

    prediction = pd.DataFrame({
        'nodeA': [edge[0] for edge in non_edges],
        'nodeB': [edge[1] for edge in non_edges],
        'spm': A_perturbated[[stock_list.loc[edge[0]].values[0] for edge in non_edges],
                             [stock_list.loc[edge[1]].values[0] for edge in non_edges]]
    }).sort_values(by='spm', ascending=False)

    prediction = prediction.head(removed_link_count)
    score = np.sum([g.are_connected(nodeA, nodeB) for nodeA, nodeB in
                    zip(prediction['nodeA'], prediction['nodeB'])]) / removed_link_count
    return score