import configparser

config = configparser.ConfigParser()
config.read("src/pyalgo.cfg")

ACCOUNT_ID = config["oanda"]["account_id"]
ACCESS_TOKEN = config["oanda"]["access_token"]
ACCOUNT_TYPE = config["oanda"]["account_type"]

print(ACCOUNT_ID, ACCESS_TOKEN, ACCOUNT_TYPE)