import configparser
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders

# ===== 設定読み込み =====
config = configparser.ConfigParser()
config.read("src/config.ini")

ACCOUNT_ID = config["oanda"]["account_id"]
ACCESS_TOKEN = config["oanda"]["access_token"]

# Practice 環境で初期化
api = API(access_token=ACCESS_TOKEN, environment="practice")
# api = API(access_token=ACCESS_TOKEN, environment="live")


# ===== 現在価格取得 =====
def get_price(instrument: str):
    params = {"instruments": instrument}
    r = pricing.PricingInfo(accountID=ACCOUNT_ID, params=params)
    resp = api.request(r)

    p = resp["prices"][0]
    bid = float(p["bids"][0]["price"])
    ask = float(p["asks"][0]["price"])
    mid = (bid + ask) / 2.0

    return bid, ask, mid

# ===== 成行＋TP/SL（OCO相当） =====
def place_oco_order(instrument: str, units: int, tp_pips: float, sl_pips: float):
    bid, ask, mid = get_price(instrument)

    tp_price = round(mid + tp_pips, 3)
    sl_price = round(mid - sl_pips, 3)

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",

            # ★ 本物の v20 の TP/SL
            "takeProfitOnFill": {
                "price": str(tp_price)
            },
            "stopLossOnFill": {
                "price": str(sl_price)
            }
        }
    }

    r = orders.OrderCreate(accountID=ACCOUNT_ID, data=order_data)
    resp = api.request(r)
    return resp

if __name__ == "__main__":
    instrument = "USD_JPY"
    units = 20000

    resp = place_oco_order(instrument, units, tp_pips=0.20, sl_pips=0.20)
    print(resp)
    

