import math
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tenacity import retry, stop_after_attempt, wait_fixed

from .utils import get_logger, send_email

# from utils import send_email, get_logger


def truncate(n: float, decimals: int) -> float:
    # TODO: truncate as string
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


@retry(wait=wait_fixed(5))
def fetch_balance(exchange, logger, symbol: str, selector: str = None) -> float:
    balance: float = 0.0
    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        message = f"could not fetch initial base balance due to {e}"
        logger.error(message)

    try:
        if selector is not None:
            balance = balance[symbol][selector]
        else:
            balance = balance[symbol]
    except Exception as e:
        err_msg = f"{symbol} not found in balance"
        logger.error(err_msg)
    return balance


@retry(wait=wait_fixed(5))
def fetch_base(config) -> float:
    base: float = 0.0

    if config["use_mock_data"]:
        return config["mock_wallet"][config["base"]]

    else:
        try:
            base = config["exchange"].fetch_balance()
        except Exception as e:
            message = f"could not fetch base balance due to {e}"
            config["event_logger"].error(message)

        try:
            base = base[config["base"]]["free"]

        except Exception as e:
            message = f"could not fetch free base due to {e}"
            config["event_logger"].error(message)

        return base


@retry(wait=wait_fixed(5))
def fetch_quote(config) -> float:
    quote: float = 0.0
    if config["use_mock_data"]:
        return config["mock_wallet"][config["quote"]]
    else:
        try:
            quote = config["exchange"].fetch_balance()
        except Exception as e:
            message = f"could not fetch quote balance due to {e}"
            config["event_logger"].error(message)

        try:
            quote = quote[config["quote"]]["free"]

        except Exception as e:
            message = f"could not fetch free quote due to {e}"
            config["event_logger"].error(message)

        return quote


@retry(wait=wait_fixed(5))
def trigger_sell_order(config: dict, bot: dict) -> dict:
    # TODO: add sanity check/mutex for race condition if another bot trades
    # between fetching `quote_before_sell` and `quote_after_sell`

    quote_before_sell = fetch_quote(config)
    base_before_sell = fetch_base(config)

    if config["use_mock_data"]:
        sell_order = {}
        sell_order["average"] = bot["current_price"] * (1 - config["slippage"])
        sell_order["cost"] = bot["base_to_sell"] * sell_order["average"]
        sell_order["fee"] = {}
        sell_order["fee"]["cost"] = config["fee"] * sell_order["cost"]
        sell_order["fee"]["currency"] = config["quote"]

        sell_order["amount"] = bot["base_to_sell"]

        config["mock_wallet"][config["base"]] -= bot["base_to_sell"]
        config["mock_wallet"][config["quote"]] += (
            sell_order["cost"] - sell_order["fee"]["cost"]
        )
    else:
        try:
            sell_order = config["exchange"].create_order(
                symbol=config["market"],
                type="market",
                side="sell",
                amount=config["exchange"].amount_to_precision(
                    config["market"], bot["base_to_sell"]
                ),
            )
            # re-assign otherwise the order will be diplayed as 'not filled'
            sell_order = config["exchange"].fetchOrder(sell_order["id"])
        except Exception as e:
            message = f"Bot {bot['bot_num']} in {config['market']} failed to sell {bot['base_to_sell']} base due to {e}"
            config["event_logger"].error(message)
            bot["quote_to_spend"] = 0

            return bot

    quote_after_sell = fetch_quote(config)
    bot["quote_to_spend"] = truncate(quote_after_sell - quote_before_sell, 8)
    bot["base_to_sell"] = 0
    bot["order_object"] = sell_order
    bot["sell_price"] = sell_order["average"]
    bot["last_trade"] = "sell"
    values_before = base_before_sell * sell_order["average"] + quote_before_sell
    value_after = fetch_base(config) * sell_order["average"] + fetch_quote(config)
    bot["last_fee_in_quote"] = values_before - value_after
    bot["last_fee_in_percent"] = bot["last_fee_in_quote"] / sell_order["cost"]

    log_order("sell", bot, config)

    return bot


@retry(wait=wait_fixed(5))
def trigger_buy_order(config: dict, bot: dict) -> dict:

    # TODO: add sanity check/mutex for race condition if another bot trades
    # between fetching `base_before_sell` and `base_after_sell`

    quote_before_sell = fetch_quote(config)
    base_before_sell = fetch_base(config)

    if config["use_mock_data"]:
        buy_order = {}
        buy_order["average"] = bot["current_price"] * (1 + config["slippage"])
        buy_order["cost"] = bot["quote_to_spend"]
        buy_order["fee"] = {}
        buy_order["fee"]["cost"] = config["fee"] * buy_order["cost"]
        buy_order["fee"]["currency"] = config["quote"]

        buy_order["amount"] = bot["quote_to_spend"] / buy_order["average"]

        config["mock_wallet"][config["base"]] += (buy_order["amount"]) * (
            1 - config["fee"]
        )
        config["mock_wallet"][config["quote"]] -= bot["quote_to_spend"]

    else:
        try:
            buy_amount = bot["quote_to_spend"] / get_price("ask", config)
            buy_order = config["exchange"].create_order(
                symbol=config["market"],
                type="market",
                side="buy",
                # quoteOrderQty param only existst for Binance get_price("avg_price", config)
                amount=config["exchange"].amount_to_precision(
                    config["market"], buy_amount
                ),
                price=None,
            )

            # re-assign otherwise the order will be diplayed as 'not filled'
            buy_order = config["exchange"].fetchOrder(buy_order["id"])

        except Exception as e:
            message = f"Bot {bot['bot_num']} in {config['market']} failed to buy {config['base']} for {bot['quote_to_spend']} USDT due to {e}"
            config["event_logger"].error(message)

            return bot

    base_after_sell = fetch_base(config)
    bot["base_to_sell"] = truncate(base_after_sell - base_before_sell, 8)
    bot["quote_to_spend"] = 0
    bot["order_object"] = buy_order
    bot["sell_price"] = buy_order["average"]
    bot["last_trade"] = "buy"

    value_before = base_before_sell * buy_order["average"] + quote_before_sell
    value_after = fetch_base(config) * buy_order["average"] + fetch_quote(config)

    bot["last_fee_in_quote"] = value_before - value_after
    bot["last_fee_in_percent"] = bot["last_fee_in_quote"] / buy_order["cost"]

    log_order("buy", bot, config)

    return bot


def update_price_info(df: DataFrame, config: dict, bot: dict) -> pd.DataFrame:

    df.loc[config["next_row"], "close"] = bot["current_price"]
    df.loc[config["next_row"], "ask"] = get_price("ask", config)
    df.loc[config["next_row"], "bid"] = get_price("bid", config)
    df.loc[config["next_row"], "spread_in_percent"] = bot["spread_in_percent"]

    # get current time
    if config["use_mock_data"]:
        dt = config["mock_data"].loc[config["fetch_index"], "dt"]
        df.loc[config["next_row"], "dt"] = dt
    else:
        df.loc[config["next_row"], "dt"] = config["exchange"].iso8601(
            config["exchange"].milliseconds()
        )

    df["asset_return"] = df["close"] / df["close"].shift()
    df["market"] = config["market"]
    df["bot_num"] = bot["bot_num"]
    df["number_of_bots"] = config["number_of_bots"]

    return df


def update_order_info(df: DataFrame, config: dict, bot: dict) -> pd.DataFrame:
    # update df
    df.loc[config["next_row"], "quote_to_spend"] = bot["quote_to_spend"]
    df.loc[config["next_row"], "base_to_sell"] = bot["base_to_sell"]
    df.loc[config["next_row"], "value_in_quote"] = (
        bot["base_to_sell"] * bot["current_price"] + bot["quote_to_spend"]
    )

    # only update if there was an order
    if not bot["last_trade"] == "":
        df.loc[config["next_row"], f"{bot['last_trade']}_price"] = bot["order_object"][
            "average"
        ]
        df.loc[config["next_row"], "fee_in_quote"] = bot["last_fee_in_quote"]
        df.loc[config["next_row"], "fee_in_percent"] = bot["last_fee_in_percent"]

    return df


def get_df(start: int, end: int, config: dict) -> pd.DataFrame:
    """
    fetch 1-m candles and resample to the indicated timeframe.
    Example:
    just fetching 1 hour candles would result
    in the last candle being inaccurate as the *current* 1h candle does not
    close now but some time 0-1 hour from now.
    timeframe = timeframe e.g. 1 min, 1 hour, 1 day etc. in miliseconds
    lookback = how many of those timeframes back
    """

    timeframe_in_minutes = int(config["timeframe_in_ms"] / 60000)

    if config["use_mock_data"]:

        # round milliseconds to whole minutes and convert back to milliseconds
        start = int(np.round(start / 60000) * 60000)
        end = int(np.round(end / 60000) * 60000)

        mock_data = config["mock_data"]
        start = int(mock_data[mock_data["milliseconds"] == start].index.values[0])
        end = int(mock_data[mock_data["milliseconds"] == end].index.values[0])

        history = mock_data.loc[start:end:timeframe_in_minutes, :]

        history = history.reset_index(drop=True)

        return history

    else:

        all_candles = []
        while start < end:
            candles = config["exchange"].fetchOHLCV(
                config["market"], timeframe="1m", since=start
            )
            if len(candles):
                start = candles[-1][0]
                all_candles += candles
            else:
                break

        df = pd.DataFrame(all_candles)
        df.drop_duplicates(inplace=True)
        df.rename(
            columns={0: "dt", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"},
            inplace=True,
        )
        df["dt"] = df["dt"].apply(lambda x: config["exchange"].iso8601(x))

        # up-sample to hourly data
        df = df.iloc[::timeframe_in_minutes, [0, 4]].reset_index(drop=True)

        return df


@retry(wait=wait_fixed(5))
def get_price(selector, config) -> float:
    """Fetched den aktuellen Marktpreis für die jeweilige Seite"""

    if config["use_mock_data"]:
        price = config["mock_data"].loc[config["fetch_index"], "close"]

        bid_ask: dict[str, float] = {
            "bid": price,
            "ask": price,
            "spread": 0,
            "avg_price": price,
        }
        return bid_ask[selector]
    else:
        try:
            orderbook = config["exchange"].fetch_order_book(config["market"])
        except Exception as e:
            if config["event_logger"]:
                config["event_logger"].error(
                    f"get_price failed, retrying. Exception: {e}"
                )
            else:
                print(f"get_price failed, retrying: {e}")

        # TODO: check if tenacity retries because of the handled exception or because
        # of the unhandled KeyError below (orderbook is unbound)
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


def get_returns(
    df,
    price_col: str = "close",
    position_col: str = "position",
    suffix: str = "",
    fees: float = 0.00075,
    slippage: float = 0.001,
) -> pd.DataFrame:

    # nan might appear if slow==fast, in that case repeat last position
    df[position_col] = df[position_col].fillna(method="ffill")

    buys = (df.loc[:, position_col].shift(1) == "short") & (
        df.loc[:, position_col] == "long"
    )
    sells = (df.loc[:, position_col].shift(1) == "long") & (
        df.loc[:, position_col] == "short"
    )
    # put fee at every order, otherwise 1
    df[f"trade{suffix}"] = np.where((buys | sells), 1, 0)
    df[f"fees{suffix}"] = np.where((buys | sells), 1 - fees, 1)

    # calculate slippage
    conditions = [buys, sells]
    choices = [df[price_col] * (1 + slippage), df[price_col] * (1 - slippage)]

    df[f"slipped_close{suffix}"] = np.select(conditions, choices, default=df[price_col])

    df[f"asset_return{suffix}"] = df["slipped_close"] / df["slipped_close"].shift()

    conditions = [
        (df[position_col].shift() == "long"),
        (df[position_col].shift() == "short"),
    ]
    choices = [df["asset_return"], 1]

    df[f"bot_return{suffix}"] = (
        np.select(conditions, choices, default=np.nan) * df[f"fees{suffix}"]
    )
    df[f"buy_prices{suffix}"] = np.where(buys, df[f"slipped_close{suffix}"], np.nan)
    df[f"sell_prices{suffix}"] = np.where(sells, df[f"slipped_close{suffix}"], np.nan)
    ix = df[price_col].first_valid_index()
    df[f"bot_eq_curve{suffix}"] = df.loc[:, price_col] / df.loc[ix, price_col]

    return df

    # position | position.shift() | asset_return | bot_return * fees
    # long               na              0.95           1         1
    # long               long            0.99           0.99      1
    # long               long            1.01           1.01      1
    # short              long            0.88           0.88      0.99925
    # short              short           0.95           1         1
    # long               short           1.05           1         0.99925
    # long               long            1.02           1.02      1


def update_mas(df: pd.DataFrame, config: dict) -> pd.DataFrame:

    df["smoothed"] = df["close"].rolling(window=config["smoothing"]).mean()
    df["fast"] = df["smoothed"].rolling(window=config["fast_lookback"]).mean().diff()
    df["slow"] = df["smoothed"].rolling(window=config["slow_lookback"]).mean().diff()

    df["fast_ma"] = df["smoothed"].rolling(window=config["fast_lookback"]).mean()
    df["slow_ma"] = df["smoothed"].rolling(window=config["slow_lookback"]).mean()

    return df


def update_multi_df(multi_df: pd.DataFrame, config: dict):
    number_of_bots = config["number_of_bots"]

    multi_df["dt"] = pd.to_datetime(multi_df["dt"])
    multi_df = multi_df.reset_index(drop=True)
    multi_df = multi_df.sort_values("dt", ascending=True)

    multi_df["total_value_in_quote"] = (
        multi_df["value_in_quote"].rolling(number_of_bots).sum()
    )
    multi_df["total_base_to_sell"] = (
        multi_df["base_to_sell"].rolling(number_of_bots).sum()
    )
    multi_df["total_quote_to_spend"] = (
        multi_df["quote_to_spend"].rolling(number_of_bots).sum()
    )

    start_index = multi_df["total_value_in_quote"].first_valid_index()
    total_start_value_in_quote = multi_df.loc[start_index, "total_value_in_quote"]

    multi_df["bot_return"] = (
        multi_df["total_value_in_quote"] / multi_df["total_value_in_quote"].shift()
    )
    multi_df["bot_eq_curve"] = (
        multi_df["total_value_in_quote"] / total_start_value_in_quote
    )
    multi_df["asset_eq_curve"] = (
        multi_df.loc[start_index:, "close"] / multi_df.loc[start_index, "close"]
    )

    return multi_df


def get_min_base_order_amount(config) -> float:

    price = get_price("avg_price", config)

    if config["quote"] == "USDT":
        return 10 / price

    elif config["quote"] == "BNB":
        return 0.05 / price

    elif config["quote"] == "BTC":
        return 0.0001 / price

    elif config["quote"] == "ETH":
        return 0.005 / price

    elif config["quote"] == "BUSD":
        return 10 / price
    else:
        config["event_logger"].info(f"Unknown quote currency {config['quote']}.")


def log_order(type, bot, config):
    order_return = np.round(((bot["sell_price"] / bot["buy_price"]) - 1) * 100, 4)

    amount = bot["order_object"]["amount"]
    cost = bot["order_object"]["cost"]
    price = np.round(bot["order_object"]["average"], 3)

    if type == "sell":
        config["event_logger"].info(
            f"Bot: {bot['bot_num']} - SELL - amount: {amount} {config['base']} - cost: {cost} {config['quote']} - price: {price} {config['market']} - return: {order_return} %"
        )
    elif type == "buy":
        config["event_logger"].info(
            f"Bot: {bot['bot_num']} - BUY - amount: {amount} {config['base']} - cost: {cost} {config['quote']} - price: {price} {config['market']} - return: {order_return} %"
        )
