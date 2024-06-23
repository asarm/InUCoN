import yfinance as yf
import pandas as pd
from tqdm import tqdm

def stockListFromURL(market="usa"):
    if market == "usa":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    elif market == "tr":
        url = "https://www.isyatirim.com.tr/en-us/analysis/stocks/Pages/bist-data-table.aspx?endeks=01#page-1"
    elif market == "london":
        url = "https://en.wikipedia.org/wiki/FTSE_250_Index"

    tables = pd.read_html(url)
    if market == "usa":
        stock_list = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
        return stock_list.Symbol
    elif market == "london":
        stock_list = tables[3]["Ticker"]
        stock_list = [s + ".L" for s in stock_list]
        return stock_list

    else:
        stock_list = list(tables[2].Stock)
        stock_list = [s+".IS" for s in stock_list]
        return stock_list


def stockDataFromYf(stock_list):
    stock_data = yf.Tickers(list(stock_list)).history(start="2021-06-01", end="2024-06-01")

    return stock_data


def filterStocksByVolume(stock_data, largest_n=200):
    stock_data_filtered = stock_data.dropna(axis=1)
    volume_sorted = stock_data_filtered["Volume"].mean().sort_values(ascending=False)
    selected_tickers = volume_sorted.index[:largest_n]

    return selected_tickers


def fetchTextualInformation(stock_list, stock2id):
    info = {}

    for ticker_name in tqdm(stock_list):
        ticker = yf.Ticker(ticker_name)
        try:
            ticker_info = ticker.info
            desc = ticker_info["longBusinessSummary"]
            sector = ticker_info["sector"]
            industry = ticker_info["industry"]
            name = ticker_info["shortName"]

            info[ticker_name] = {
                "ticker_id": stock2id[ticker_name],
                "code": ticker_name,
                "name": name,
                "sector": sector,
                "industry": industry,
                "description": desc
            }
        except Exception as e:
            pass

    return info


def stocks2id(stock_list):
    id_dict = {}

    for stock in stock_list:
        id_dict[stock] = len(id_dict)

    return id_dict


def id2stock(target_id, stock2id):
    if target_id <= len(stock2id):
        return stock2id[stock2id.stock_id == target_id].index.item()
    else:
        f"Size match error, target_id: {target_id}, len dict: {len(stock2id)}"


def prepare_data(save_data=False):
    stock_list = stockListFromURL()
    stock_data = stockDataFromYf(stock_list=stock_list)
    filtered_stock_list = filterStocksByVolume(stock_data, largest_n=200)
    filtered_stock_data = stockDataFromYf(filtered_stock_list)
    stock_id_list = stocks2id(filtered_stock_list)
    textual_information = fetchTextualInformation(filtered_stock_list, stock_id_list)

    if save_data:
        filtered_stock_data["Close"].to_csv("stock_data.csv", index=True)
        pd.DataFrame(textual_information).T.to_csv("textual_information.csv", index=False,
                                                   columns=["ticker_id", "code", "name", "sector", "industry",
                                                            "description"])
        pd.DataFrame(stock_id_list, index=["id"]).to_csv("cold_data/stock2id.csv", index=False)

    return filtered_stock_data, textual_information, stock_id_list


def load_files(textual_path="cold_data/textual_information.csv",
               stock2idPath="cold_data/stock2id.csv",
               stockDataPath="cold_data/stock_data.csv"):
    textual_information = pd.read_csv(textual_path)
    stock2id = pd.read_csv(stock2idPath).T
    stockData = pd.read_csv(stockDataPath)
    stockData.index = stockData.Date
    stockData.drop("Date", inplace=True, axis=1)
    stock2id.columns = ["stock_id"]

    return textual_information, stock2id, stockData

# prepare_data(save_data=True)

# print(pd.read_csv("textual_information.csv"))
# print(pd.read_csv("stock2id.csv"))
# print(pd.read_csv("stock_data.csv"))