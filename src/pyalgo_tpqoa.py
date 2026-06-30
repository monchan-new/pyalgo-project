import tpqoa
import pandas as pd

api = tpqoa.tpqoa('src/pyalgo.cfg')

# ヒストリーデータの取得
# df = api.get_history(
#     instrument='USD_JPY',
#     start='2024-01-01',
#     end='2024-01-05',
#     granularity='H1',
#     price='M'
# )


# 成行買い（1万通貨）
# api.create_order(
#     instrument='USD_JPY',
#     units=10000,        # + で buy
# )

# 建玉一覧を確認
positions = api.get_positions()
print(positions)


# 該当の Long（10,000 通貨）だけ決済
# api.create_order(
#     instrument='USD_JPY',
#     units=-20000   # 20,000 通貨のショート → Long の10,000が相殺される
# )







# import configparser
# config = configparser.ConfigParser()
# config.read("src/pyalgo.cfg")

# ACCOUNT_ID = config["oanda"]["account_id"]
# ACCESS_TOKEN = config["oanda"]["access_token"]
# ACCOUNT_TYPE = config["oanda"]["account_type"]

# print(ACCOUNT_ID, ACCESS_TOKEN, ACCOUNT_TYPE)

# import requests
# url = "https://api-fxpractice.oanda.com/labs/v1/candles"
# params = {
#   "instrument": "USD_JPY",
#   "count": 10,
#   "granularity": "H1"
# }

# headers= {
#   "User-Agent": "Mozilla/5.0"
# }

# response = requests.get(url, params=params, headers=headers)
# print(response.status_code)
# print(response.text)