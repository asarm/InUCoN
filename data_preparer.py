import pandas as pd
import numpy as np
import networkx as nx

from helpers.time_series_functions import calculate_historical_ts_similarities
from helpers import graph_functions, data_preparation, nlp_functions

import copy

import igraph as ig
from igraph import Graph

class DataPreparer:
    def __init__(self,
                 data_path="cold_data/description_vectors.pkl",
                 textual_path="cold_data/textual_information.csv",
                 stock2idPath="cold_data/stock2id.csv",
                 stockDataPath="cold_data/stock_data.csv"
                 ):
        self.textual_information, self.stock2id, self.stockData = data_preparation.load_files(
            textual_path, stock2idPath, stockDataPath
        )
        self.text_similarity_matrix = nlp_functions.prepare_textual_similarity_matrix(
            vectors_path=f"{data_path}")
        self.historical_ts_similarities = None

    def calculate_historical_ts_similarities(self, start_date="2023-01-01", end_date="2023-06-01",
                                             ts_similarity_metric="euclidean", window_size=30, overlap=7):

        self.ts_similarities = calculate_historical_ts_similarities(
            stockData=self.stockData,
            start_date=start_date, end_date=end_date,
            similarity_metric=ts_similarity_metric,
            window_size=window_size, overlap=overlap
        )

    def get_historical_ts_similarities(self):
        return self.ts_similarities

    def prepare_snapshots(self, weight_combination={"ts": 0.55, "tx": 0.45}):
        if self.ts_similarities is None:
            print("Calculating TS similarities")
            self.calculate_historical_ts_similarities()

        snapshot_count = len(self.ts_similarities.keys())
        self.snapshots = []

        ts_weight, textual_weight = weight_combination["ts"], weight_combination["tx"]
        for time_idx in range(snapshot_count):
            nx_snapshot = graph_functions.build_graph(
                similarities=self.ts_similarities[list(self.ts_similarities.keys())[time_idx]],
                textual_similarities=self.text_similarity_matrix,
                stock2id=self.stock2id,
                col_list=list(self.stockData.columns),
                threshold_metric="custom",
                add_text_sim=True,
                ts_w=ts_weight,
                text_w=textual_weight
            )

            # Remove self-loops in the NetworkX graph
            nx_snapshot.remove_edges_from(nx.selfloop_edges(nx_snapshot))

            # Convert NetworkX graph to igraph
            ig_snapshot = ig.Graph(directed=nx_snapshot.is_directed())
            ig_snapshot.add_vertices(list(nx_snapshot.nodes()))

            edges = list(nx_snapshot.edges(data=True))
            edge_list = [(u, v) for u, v, data in edges]
            weights = [data['weight'] for u, v, data in edges]

            ig_snapshot.add_edges(edge_list)
            ig_snapshot.es['weight'] = weights

            self.snapshots.append(ig_snapshot)

    def get_snapshots(self):
        self.snapshot_dict = {}
        date_ranges = list(self.ts_similarities.keys())
        for date_id, date_range in enumerate(date_ranges):
            self.snapshot_dict[date_range] = self.snapshots[date_id]

        return self.snapshot_dict

    def calculate_similarity_matrix(self, node_count, snapshot_start_id=0, snapshot_end_id=5):
        cluster_rank_matrix = []
        combination_snapshots = copy.deepcopy(self.snapshots)[snapshot_start_id:snapshot_end_id]

        for _ in np.arange(0, node_count):
            cluster_rank_matrix.append([0] * node_count)

        for s_id, snapshot in enumerate(combination_snapshots):
            communities = snapshot.community_infomap(edge_weights="weight", trials=30)
            # communities = snapshot.community_multilevel(weights='weight')
            # print("Community Modularity:", communities.modularity)

            for com in communities:
                com = list(com)
                # print(com)
                if len(com) > 1:
                    for node_id in range(len(com) - 1):
                        src = com[node_id]
                        trgt = com[node_id + 1]

                        cluster_rank_matrix[src][trgt] += 1
                        cluster_rank_matrix[trgt][src] += 1

        self.distance_matrix = 1 / (1 + np.array(cluster_rank_matrix))
        self.sim_matrix = 1 - self.distance_matrix

    def get_similarity_matrix(self):
        return self.sim_matrix

    def np2igraph(self, network_matrice):
        network = Graph.Weighted_Adjacency(network_matrice, mode="undirected")
        return network

    def calculate_final_communes(self):
        network = self.sim_matrix.tolist()
        network = Graph.Weighted_Adjacency(network, mode="undirected")

        self.communities = network.community_infomap(edge_weights="weight", trials=20)
        # print("Final Community Modularity:", self.communities.modularity)

    def get_communities(self):
        return self.communities

    def calculate_node_metrics(self):
        network_array = self.sim_matrix
        network = self.np2igraph(network_array)

        betweenness_scores = network.betweenness(directed=False, weights="weight")
        pagerank_scores = network.pagerank(directed=False, weights="weight")
        centrality_scores = network.closeness(weights="weight")

        # Create the metrics DataFrame
        metrics = pd.DataFrame({
            "betweenness": betweenness_scores,
            "pagerank": pagerank_scores,
            "centrality": centrality_scores,
            "commune": None
        })

        for commune_id, commune_nodes in enumerate(list(self.communities)):
            for node in commune_nodes:
                metrics.loc[int(node), "commune"] = int(commune_id)

        return metrics

    def choose_nodes(self, metric_df, by="betweenness", how="lower"):
        selected_universe = None
        if how == "lower":
            selected_universe = metric_df.groupby(by="commune")[by].idxmin()
        elif how == "higher":
            selected_universe = metric_df.groupby(by="commune")[by].idxmax()

        return selected_universe

    def id2stock(self, target_id):
        try:
            return self.stock2id[self.stock2id.stock_id == target_id].index.item()
        except:
            return None