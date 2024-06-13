from data_preparer import DataPreparer
import pickle

with open("calculated_data/historical_ts_sim.pkl", "rb") as f:
    ts_sim = pickle.load(f)

preparer = DataPreparer()
preparer.ts_similarities = ts_sim

preparer.prepare_snapshots(weight_combination={"ts":0.55, "tx":0.45})
preparer.calculate_similarity_matrix(node_count=200, snapshot_start_id=0, snapshot_end_id=5)
preparer.calculate_final_communes()

node_metrics = preparer.calculate_node_metrics()
investment_universe = preparer.choose_nodes(node_metrics, by="betweenness", how="lower")
print(list(investment_universe))