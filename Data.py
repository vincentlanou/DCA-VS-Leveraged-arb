import requests
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

data_rates = {}
data_closing_prices = {}

#Bank of Canada Policy Rate
url = "https://www.bankofcanada.ca/valet/observations/V39079/json"
response = requests.get(url)
boc_json = response.json()
# Convert the JSON observations into a Pandas DataFrame
df_boc = pd.DataFrame(boc_json['observations'])
# The API returns dates in column 'd' and values nested inside 'V39079' -> 'v'
df_boc['Date'] = pd.to_datetime(df_boc['d'])
df_boc['Close'] = df_boc['V39079'].apply(lambda x: float(x['v']))
# Set the date as the index and resample exactly like yfinance
df_boc.set_index('Date', inplace=True)
df_boc_monthly = df_boc['Close'].resample('MS').last().dropna()
# A list of tuples with date and monthly rate, converted to monthly returns
boc_rates_list = [
    (date.strftime('%Y-%m-%d'), round(((1 + (float(value) / 100))**(1/12) - 1), 8))
    for date, value in df_boc_monthly.items()
]
data_rates["Policy_Rate_BOC"] = boc_rates_list


#for the exchange rate
df_fx = yf.Ticker("CAD=X").history(period="max")
fx_close = df_fx['Close']
fx_close.index = pd.to_datetime(fx_close.index).tz_localize(None)
fx_monthly = fx_close.resample('MS').last().dropna()
list_exchange_rate = []
for date, value in fx_monthly.items():
    date_str = date.strftime('%Y-%m-%d')
    list_exchange_rate.append((date_str, round(float(value), 6)))

tickers = {
    "SP500": "SPY",       # USD
    "TSX": "XIU.TO",      # CAD
    "Ex_US_Can": "EFA"    # USD
}

data_closing_prices = {}
for name, ticker in tickers.items():
    df = yf.Ticker(ticker).history(period="max")
    df_close = df['Close']
    df_close.index = pd.to_datetime(df_close.index).tz_localize(None)
    df_monthly = df_close.resample('MS').last().dropna()
    # Create a list of tuples with date, monthly closing price
    monthly_list = []
    for date, value in df_monthly.items():
        date_str = date.strftime('%Y-%m-%d')
        monthly_list.append((date_str, round(float(value), 6)))
    if name != "TSX":  # For TSX, we use the closing price directly
        temp_list = []
        for i in range(len(min(monthly_list, list_exchange_rate, key=len))):
            temp_list.insert(0, (monthly_list[-i-1][0], round(monthly_list[-i-1][1] * list_exchange_rate[-i-1][1], 6)))
        monthly_list = temp_list
    data_closing_prices[name] = monthly_list

    rates_list = []
    for dat, value in monthly_list[1:]:
        rates_list.append((dat, round(((value / monthly_list[monthly_list.index((dat, value)) - 1][1])-1), 8)))
    data_rates[name] = rates_list


min_length_backtest = min(len(data_rates["Policy_Rate_BOC"]), len(data_rates["SP500"]), len(data_rates["TSX"]), len(data_rates["Ex_US_Can"]))
rates_for_backtest = {}
closing_for_backtest = {}
for key in data_rates:
    rates_for_backtest[key] = data_rates[key][-min_length_backtest:]
for key in data_closing_prices:
    closing_for_backtest[key] = data_closing_prices[key][-min_length_backtest:]

min_length_bootstrap = min(len(data_rates["SP500"]), len(data_rates["TSX"]), len(data_rates["Ex_US_Can"]))
rates_for_bootstrap = {}
closing_for_bootstrap = {}
for key in tickers.keys():
    rates_for_bootstrap[key] = data_rates[key][-min_length_bootstrap:]
for key in tickers.keys():
    closing_for_bootstrap[key] = data_closing_prices[key][-min_length_bootstrap:]

#To be coherent for the correlation and the sharpe ratio, we will use the same data for the backtest and the calculations.

#Standard deviation of the returns (SP500, TSX, Ex_US_Can)
std = {}
for key in data_closing_prices:
    std[key] = np.std([rates_for_backtest[key][i][1] for i in range(len(rates_for_backtest[key]))])

Pearson_cor = {
    "SP500_TSX": (np.cov([x[1] for x in rates_for_backtest["SP500"]], [x[1] for x in rates_for_backtest["TSX"]])[0, 1]) / (std["SP500"] * std["TSX"]),
    "SP500_Ex_US_Can": (np.cov([x[1] for x in rates_for_backtest["SP500"]], [x[1] for x in rates_for_backtest["Ex_US_Can"]])[0, 1]) / (std["SP500"] * std["Ex_US_Can"]),
    "TSX_Ex_US_Can": (np.cov([x[1] for x in rates_for_backtest["TSX"]], [x[1] for x in rates_for_backtest["Ex_US_Can"]])[0, 1]) / (std["TSX"] * std["Ex_US_Can"])
}
