from tqdm import tqdm
import numpy as np

import dcor
from dtw import *
from sklearn.metrics.pairwise import euclidean_distances
from datetime import timedelta, datetime

import pickle

def calculate_ts_sim(price_history, start_date, end_date, similarity_metric="euclidean"):
    ts_similarities = {}
    col_list = list(price_history.columns)
    for id in tqdm(range(len(col_list))):
        ticker1 = col_list[id]
        date_filtered = price_history[(price_history.index >= start_date) & (price_history.index <= end_date)]

        ticker1_prices = date_filtered[[ticker1]].pct_change()[1:]

        for sec_id in range(len(col_list)):
            ticker2 = col_list[sec_id]
            pair = (ticker1, ticker2)

            if (ticker2, ticker1) in list(ts_similarities.keys()):
                continue
            else:
                ticker2_prices = date_filtered[[ticker2]].pct_change()[1:]

                # Get the indices of the common values
                mask1 = np.isin(list(ticker1_prices.index), list(ticker2_prices.index))
                mask2 = np.isin(list(ticker2_prices.index), list(ticker1_prices.index))
                indices_array1 = np.where(mask1)[0]
                indices_array2 = np.where(mask2)[0]

                ticker1_prices = ticker1_prices.iloc[indices_array1]
                ticker2_prices = ticker2_prices.iloc[indices_array2]

                if similarity_metric == "dtw":
                    # Check if data is valid for DTW calculation
                    ticker1_data = list(ticker1_prices[ticker1])
                    ticker2_data = list(ticker2_prices[ticker2])
                    
                    # Remove NaN values
                    ticker1_data = [x for x in ticker1_data if not np.isnan(x)]
                    ticker2_data = [x for x in ticker2_data if not np.isnan(x)]
                    
                    # Check if we have enough data and data is not constant
                    if (len(ticker1_data) > 1 and len(ticker2_data) > 1 and 
                        np.std(ticker1_data) > 0 and np.std(ticker2_data) > 0):
                        
                        a = dtw(ticker1_data, ticker2_data, keep_internals=False)
                        ts_similarities[pair] = a.distance
                    else:
                        # If data is constant or insufficient, use a fallback value
                        ts_similarities[pair] = float('inf')  # or another appropriate value

                elif similarity_metric == "euclidean":
                    # Remove NaN values for euclidean calculation
                    ticker1_data = np.array(ticker1_prices[ticker1])
                    ticker2_data = np.array(ticker2_prices[ticker2])
                    
                    # Remove NaN values
                    mask1 = ~np.isnan(ticker1_data)
                    mask2 = ~np.isnan(ticker2_data)
                    common_mask = mask1 & mask2
                    
                    if np.sum(common_mask) > 1:
                        ticker1_clean = ticker1_data[common_mask].reshape(1, -1)
                        ticker2_clean = ticker2_data[common_mask].reshape(1, -1)
                        
                        distance = euclidean_distances(ticker1_clean, ticker2_clean)[0].item()
                        ts_similarities[pair] = distance
                    else:
                        ts_similarities[pair] = float('inf')

                elif similarity_metric == "dist_corr":
                    # Remove NaN values for distance correlation
                    ticker1_data = np.array(ticker1_prices[ticker1])
                    ticker2_data = np.array(ticker2_prices[ticker2])
                    
                    # Remove NaN values
                    mask1 = ~np.isnan(ticker1_data)
                    mask2 = ~np.isnan(ticker2_data)
                    common_mask = mask1 & mask2
                    
                    if np.sum(common_mask) > 1:
                        ticker1_clean = ticker1_data[common_mask]
                        ticker2_clean = ticker2_data[common_mask]
                        
                        similarity = dcor.distance_correlation(ticker1_clean, ticker2_clean)
                        distance = 1-similarity
                        ts_similarities[pair] = distance
                    else:
                        ts_similarities[pair] = 1.0  # Maximum distance when no correlation possible

                elif similarity_metric == "pearson":
                    # Remove NaN values for pearson correlation
                    ticker1_data = np.array(ticker1_prices[ticker1])
                    ticker2_data = np.array(ticker2_prices[ticker2])
                    
                    # Remove NaN values
                    mask1 = ~np.isnan(ticker1_data)
                    mask2 = ~np.isnan(ticker2_data)
                    common_mask = mask1 & mask2
                    
                    if np.sum(common_mask) > 2:  # Need at least 3 points for correlation
                        ticker1_clean = ticker1_data[common_mask]
                        ticker2_clean = ticker2_data[common_mask]
                        
                        # Check if data has variance (not constant)
                        if np.std(ticker1_clean) > 0 and np.std(ticker2_clean) > 0:
                            # Calculate correlation manually to avoid issues
                            mean1 = np.mean(ticker1_clean)
                            mean2 = np.mean(ticker2_clean)
                            
                            numerator = np.sum((ticker1_clean - mean1) * (ticker2_clean - mean2))
                            denominator = np.sqrt(np.sum((ticker1_clean - mean1)**2) * np.sum((ticker2_clean - mean2)**2))
                            
                            if denominator > 0:
                                similarity = numerator / denominator
                                # Clamp correlation to [-1, 1] range
                                similarity = np.clip(similarity, -1, 1)
                                distance = 1 - similarity
                            else:
                                distance = 1.0  # Maximum distance when denominator is 0
                        else:
                            distance = 1.0  # Maximum distance when data is constant
                    else:
                        distance = 1.0  # Maximum distance when insufficient data
                    
                    ts_similarities[pair] = distance

    return ts_similarities


def get_similiariy(similarity_dict, ticker1, ticker2):
    sim = similarity_dict.get((ticker1, ticker2), None)
    if sim is not None:
        return round(similarity_dict[(ticker1, ticker2)], 3)

    elif sim is None:
        try:
            return round(similarity_dict[(ticker2, ticker1)], 3)
        except:
            return None


def generate_time_range_list(start_date, target_end_date, window_size, overlap=7):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=window_size)

    target_end_date = datetime.strptime(target_end_date, "%Y-%m-%d")
    range_list = []
    while end_date < target_end_date:
        range_list.append((str(start_date)[:10], str(end_date)[:10]))

        start_date = end_date - timedelta(days=overlap)
        end_date = start_date + timedelta(days=window_size)

    return range_list


def calculate_historical_ts_similarities(stockData,
                                         start_date="2023-01-01", end_date="2024-04-01",
                                         save=False, path="range_similarities.pkl",
                                         similarity_metric="euclidean", window_size=30, overlap=7):
    range_list = generate_time_range_list(start_date=start_date, target_end_date=end_date, window_size=window_size, overlap=overlap)
    print("Date Range Count:", len(range_list))

    range_similarities = {}
    for time_range in range_list:
        start_date = time_range[0]
        end_date = time_range[1]

        ts_sim = calculate_ts_sim(price_history=stockData, start_date=start_date, end_date=end_date,
                                  similarity_metric=similarity_metric)
        range_similarities[time_range] = ts_sim

    if save:
        with open(path, 'wb') as file:
            pickle.dump(range_similarities, file)

    return range_similarities