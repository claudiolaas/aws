#%%
from tenacity import retry
from tenacity.wait import wait_fixed
import ccxt
import datetime
import config
import boto3
import pandas as pd

import io
from time import sleep
import io

cfg = config.get()
exchange = ccxt.binance()
exchange.load_markets

s3_client = boto3.client('s3')


@retry(wait=wait_fixed(5))
def get_price(selector, exchange, market) -> float:
    """Fetched den aktuellen Marktpreis für die jeweilige Seite"""
    try:
        orderbook = exchange.fetch_order_book(market)
    except Exception as e:
        print(f"get_price failed, retrying: {e}")

    bid: float = orderbook["bids"][0][0] if len(orderbook["bids"]) > 0 else None
    ask: float = orderbook["asks"][0][0] if len(orderbook["asks"]) > 0 else None
    spread: float = (ask / bid) if (bid and ask) else None
    avg_price: float = (bid + ask) / 2
    bid_ask: dict[str, float] = {
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "avg_price": avg_price,
    }
    return bid_ask[selector]


def df_to_s3_csv(df,csv_name):
 
    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, index=False)

        response = s3_client.put_object(
            Bucket='crwpl-prices', Key=f"{csv_name}.csv", Body=csv_buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")


prices = []
while True:
    price = get_price(selector='avg_price', exchange=exchange, market='BTC/USDT')
    print(f'price: {price}')
    prices.append(price)
    df = pd.DataFrame({'dt':datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),'prices':prices})
    df_to_s3_csv(df,'prices')
    print('added price')
    sleep(1)
