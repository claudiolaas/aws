import argparse
import sys
import time
from datetime import datetime as dt
from pathlib import Path
from time import sleep
from typing import List, TypedDict

import ccxt
import numpy as np
import pandas as pd

from . import trading_functions as f
from .utils import (
    get_git_revision_hash,
    get_git_revision_short_hash,
    read_credentials,
    save_config,
    str2bool,
)

# import trading_functions as f
# from utils import save_config, read_credentials, str2bool

__version__ = "2021.12.1"


def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--market",
        type=str,
        default="BTC/USDT",
        help="Trading symbols, e.g. BTC/USDT - only cryptos are valid",
    )
    parser.add_argument(
        "--timeframe_in_ms",
        type=int,
        default=int(60 * 60 * 1000),  # (min * sec * mili)
        help="time intervall the bot trades at in milliseconds",
    )
    parser.add_argument(
        "--fast_lookback",
        type=int,
        default=17,
        help="lookback period for the fast MA - is a multiple of timeframe",
    )
    parser.add_argument(
        "--slow_lookback",
        type=int,
        default=24,
        help="lookback period for the slow MA",
    )  # ...
    parser.add_argument(
        "--smoothing",
        type=int,
        default=6,
        help="lookback period for the MA that smoothes the raw prices",
    )
    parser.add_argument(
        "--return_lookback",
        type=int,
        default=200,
        help="lookback period for the MA that smoothes the returns of the asset and the bot",
    )
    parser.add_argument(
        "--start_amt_in_base",
        default=0.1,
        type=float,
        help="how much of the base currency the bot should start trading with",
    )
    parser.add_argument(
        "--use_all_base_for_trading",
        type=str2bool,
        default=False,
        help="if true, uses all available base curreny for trading, irrespective"
        " of --start_amt_in_base",
    )
    parser.add_argument(
        "--number_of_bots",
        type=int,
        default=2,
        help="number of bots launched in equally spaced timeframes to match the"
        " given timeframe param.",
    )
    parser.add_argument(
        "--use_mock_data",
        type=str2bool,
        default=True,
        help="if true uses mock data and wallet",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2020-01-01 00:00:00",
        help="start date of mock trading",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2022-01-20 00:00:00",
        help="end date of mock trading",
    )
    parser.add_argument(
        "--mock_data_url",
        type=str,
        default="csvs/complete_history_BTCUSDT_1m.csv",
        help="start date of mock trading",
    )
    parser.add_argument(
        "--slippage",
        type=str,
        default="0.001",
        help="slippage used for mock trading",
    )
    parser.add_argument(
        "--fees",
        type=str,
        default="0.00063175",
        help="fees used for mock trading",
    )

    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--use_api_key", type=str2bool, default=True)
    args = parser.parse_args()

    print(args.market)
    config = vars(args)

    timestr = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    config["timestr"] = timestr
    symbol = config["market"].replace("/", "-")
    config["log_pth"] = str(Path(config["log_dir"]) / Path(symbol) / timestr)
    config["base"] = symbol[: symbol.find("-")]
    config["quote"] = symbol[symbol.find("-") + 1 :]

    return config


def run_bot():
    config = get_args()
    config["version"] = __version__
    config["git_short_hash"] = get_git_revision_short_hash()
    config["git_hash"] = get_git_revision_hash()
    save_config(config)
    print("INFO: running the bot!")
    if config["use_mock_data"]:
        print("Using mock data!")
    else:
        print("Using real data!")

    mock_data = pd.read_csv(config["mock_data_url"])
    mock_wallet = {config["base"]: 10000, config["quote"]: 10000}
    config["mock_data"] = mock_data
    config["mock_wallet"] = mock_wallet
    config["slippage"] = 0.001
    config["fee"] = 0.00075

    exchange_id = "ftx"
    exchange_class = getattr(ccxt, exchange_id)
    exchange_settings = {"timeout": 30000, "enableRateLimit": True}

    if config["use_api_key"] is True:
        creds = read_credentials()
        if creds is not False:
            exchange_settings = {**exchange_settings, **creds}
            creds.clear()

    exchange = exchange_class(exchange_settings)

    start_index = int(
        mock_data[
            mock_data["milliseconds"] == exchange.parse8601(config["start_date"])
        ].index.values[0]
    )
    end_index = int(
        mock_data[
            mock_data["milliseconds"] == exchange.parse8601(config["end_date"])
        ].index.values[0]
    )
    fetch_index = int(start_index)
    config["fetch_index"] = fetch_index

    print(mock_data.loc[start_index, "dt"])
    config["exchange"] = exchange
    market = config["market"]
    timeframe_in_ms = config["timeframe_in_ms"]
    timeframe_in_sec: int = int(timeframe_in_ms / 1000)
    timeframe_in_min: int = int(timeframe_in_ms / 60000)
    start_amt_in_base = config["start_amt_in_base"]

    # initialising Loggers
    minute_logger = f.get_logger("minutes", market, config)
    hour_logger = f.get_logger("hours", market, config)
    event_logger = f.get_logger("event", market, config)
    bot_return_logger = f.get_logger("total_bot_returns", market, config)
    alpha_logger = f.get_logger("alpha", market, config)
    asset_return_logger = f.get_logger("total_asset_returns", market, config)
    spread_logger = f.get_logger("spreads", market, config)
    quote_logger = f.get_logger("quote_amounts", market, config)
    base_logger = f.get_logger("base_amounts", market, config)
    value_logger = f.get_logger("total_value_in_quote", market, config)

    config["event_logger"] = event_logger
    asset_start_price = f.get_price("avg_price", config)
    balance_start = exchange.fetch_balance()
    balance_start_in_usd = sum(
        [float(x["usdValue"]) for x in balance_start["info"]["result"]]
    )

    event_logger.info(str(config))

    number_of_bots: int = config["number_of_bots"]

    ########################################################################################################################
    ## SANITY CHECK TRADE AMOUNTS
    ########################################################################################################################

    available_base = f.fetch_base(config)

    if config["use_all_base_for_trading"] is True:
        start_amt_in_base = available_base

    # enough total base available?
    if available_base < start_amt_in_base:
        event_logger.info(
            f"Not enough base available to start trading. You requested {start_amt_in_base} but only have {available_base}. Quitting now."
        )
        sys.exit()

    # base amount per bot big enough?
    min_base_order_amount = f.get_min_base_order_amount(config)

    start_amt_in_base_per_bot = f.truncate((start_amt_in_base / number_of_bots), 8)

    if min_base_order_amount > start_amt_in_base_per_bot:
        event_logger.info(
            f"Base amount per bot too small. You requested {start_amt_in_base_per_bot} but need at least {min_base_order_amount}. Quitting now."
        )
        sys.exit()

    event_logger.info(f"Total start amount: {start_amt_in_base} {config['base']}.")
    event_logger.info(
        f"Start amount per bot: {start_amt_in_base_per_bot} {config['base']}."
    )
    event_logger.info(
        f"{np.round(((start_amt_in_base_per_bot/min_base_order_amount)-1)*100,2)} % away from not being able to trade."
    )

    ########################################################################################################################
    ## INITIALIZE BOTS
    ########################################################################################################################

    if config["use_mock_data"]:
        now = exchange.parse8601(config["start_date"])
    else:
        now = exchange.milliseconds()

    # build dict that hold the individual amounts and the dataframes
    bots: list[dict] = []
    start_minute = time.time()
    for i in range(number_of_bots):
        end_minute = time.time()
        minute_logger.info(end_minute - start_minute)
        start_minute = time.time()
        event_logger.info(f"Initializing bot number {i}.")

        total_lookback_in_ms = (
            config["slow_lookback"]
            + config["smoothing"]
            + config["return_lookback"]
            - 2
        ) * config["timeframe_in_ms"]

        # in milliseconds
        end = now - timeframe_in_ms + (i * (timeframe_in_ms / number_of_bots))
        start = end - total_lookback_in_ms

        # load history
        df = f.get_df(start, end, config)

        # these columns need to exist for plotting even without actual trades
        df["buy_price"] = np.nan
        df["sell_price"] = np.nan

        start_price = f.get_price("avg_price", config)

        bot_dict = {
            "base_to_sell": start_amt_in_base_per_bot,
            "quote_to_spend": 0,
            "df": df,
            "start_price": start_price,
            "sell_price": start_price,
            "buy_price": start_price,
            "order_object": None,
            "bot_num": i,
        }

        bots.append(bot_dict)

        # save df
        df_name = f"df_{i}.csv"
        csv_path = Path(config["log_pth"]) / df_name 
        f.df_to_s3_csv(df,csv_path)
        # df.to_csv(csv_path)

    # 420logit
    event_logger.info("Bots initialized")
    event_logger.info(f"start price {f.get_price('avg_price', config)}")
    event_logger.info(f"start time: {dt.now().strftime('%d.%m.%Y - %H:%M:%S')}")

    ########################################################################################################################
    ## WHILE LOOP
    ########################################################################################################################
    start_hour = time.time()
    while True:
        end_hour = time.time()
        hour_logger.info(end_hour - start_hour)
        start_hour = time.time()

        start_minute = time.time()
        dfs: list[pd.DataFrame] = []

        for i in range(number_of_bots):
            print("entered while")
            end_minute = time.time()
            minute_logger.info(end_minute - start_minute)
            start_minute = time.time()

            #############################
            ## Update df
            #############################
            df = bots[i]["df"]
            config["next_row"] = len(df)

            current_price = f.get_price("avg_price", config)
            current_spread = f.get_price("spread", config)
            bots[i]["spread_in_percent"] = (current_spread - 1) * 100
            bots[i]["current_price"] = current_price

            df = f.update_price_info(df, config, bots[i])
            df = f.update_mas(df, config)

            #############################
            ## MA Crossover
            #############################

            conditions = [
                (df["fast"] > df["slow"]),
                (df["slow"] > df["fast"]),
            ]
            choices = ["long", "short"]

            df["sim_position"] = np.select(conditions, choices, default=np.nan)

            df = f.get_returns(
                df,
                position_col="sim_position",
                suffix="_sim",
                fees=config["fees"],
                slippage=config["slippage"],
            )

            ##############################
            ## Return Crossover
            ##############################
            df["sim_bot_return_ma"] = (
                df["sim_bot_return"].rolling(window=config["return_lookback"]).mean()
            )
            df["asset_return_ma"] = (
                df["asset_return"].rolling(window=config["return_lookback"]).mean()
            )

            conditions = [
                (df["sim_bot_return_ma"] > df["asset_return_ma"]),
                (df["asset_return_ma"] > df["sim_bot_return_ma"]),
            ]
            choices = ["long", "short"]

            df["position"] = np.select(conditions, choices, default=np.nan)

            df = f.get_returns(
                df,
                position_col="position",
                suffix="_act",
                fees=config["fees"],
                slippage=config["slippage"],
            )

            ########################################################################################################################
            ##    O    R    D    E    R    S
            ########################################################################################################################
            # used for filtering
            # will be set to 'buy' or 'sell' in trigger_order functions
            bots[i]["last_trade"] = ""

            # Buy & hold better than the bot, dont trade, just buy.
            if df["asset_return_ma"].iloc[-1] > df["sim_bot_return_ma"].iloc[-1]:
                event_logger.info(f"Buy & hold better than the bot {i}")

                if bots[i]["quote_to_spend"] > 0:
                    bots[i] = f.trigger_buy_order(config, bots[i])

            # Bot is beating the asset, time to trrrrrrrade
            else:
                event_logger.info(f"Bot {i} is beating the asset")
                # slow over fast --> sell
                if df["slow"].iloc[-1] > df["fast"].iloc[-1]:

                    # base_to_sell will be set to 0 by a successfull sell order to stop repetetive sell orders
                    if bots[i]["base_to_sell"] > 0:
                        bots[i] = f.trigger_sell_order(config, bots[i])

                # fast over slow --> buy
                elif df["slow"].iloc[-1] < df["fast"].iloc[-1]:

                    if bots[i]["quote_to_spend"] > 0:
                        bots[i] = f.trigger_buy_order(config, bots[i])

            # update df
            df = f.update_order_info(df, config, bots[i])

            dfs.append(df)

            # sleep for x so that the bots start in equaliy spaced delays
            if config["use_mock_data"]:
                config["fetch_index"] += int(timeframe_in_min / number_of_bots)
                if config["fetch_index"] > end_index:
                    for i in range(number_of_bots):
                        df = bots[i]["df"]
                        # loggin / saving
                        df_name = f"df_{i}.csv"
                        csv_path = Path(config["log_pth"]) / df_name 
                        f.df_to_s3_csv(df,csv_path)
                        # df.to_csv(csv_path)

                        multi_df = pd.concat(dfs)
                        multi_df = f.update_multi_df(multi_df, config)
                        multi_csv_path = Path(config["log_pth"]) / "multi_df.csv"
                        f.df_to_s3_csv(multi_df,multi_csv_path)
                        # multi_df.to_csv(multi_csv_path)

                        print("backetest finished")
                        sys.exit()
            else:
                csv_path = Path(config["log_pth"]) / df_name 
                f.df_to_s3_csv(df,csv_path)
                # df.to_csv(csv_path)
                sleep(timeframe_in_sec / number_of_bots)
                print(f"Iterated bot number {i}.")

        if not config["use_mock_data"]:
            multi_df = pd.concat(dfs)
            multi_df = f.update_multi_df(multi_df, config)
            multi_csv_path = Path(config["log_pth"]) / "multi_df.csv"
            f.df_to_s3_csv(multi_df,multi_csv_path)
            # multi_df.to_csv(multi_csv_path)

        # log total asset return
        asset_current_price = f.get_price("avg_price", config)
        total_asset_return = asset_current_price / asset_start_price
        total_asset_return_percent = (total_asset_return - 1) * 100
        asset_return_logger.info(f"{np.round(total_asset_return_percent,4)} %")

        # log total bot value
        current_price = f.get_price("avg_price", config)
        total_current_value_in_quote = sum(
            [
                bot["base_to_sell"] * current_price + bot["quote_to_spend"]
                for bot in bots
            ]
        )
        value_logger.info(total_current_value_in_quote)

        # log total bot return
        total_start_value_in_quote = sum(
            [start_amt_in_base_per_bot * bot["start_price"] for bot in bots]
        )
        total_bot_return = total_current_value_in_quote / total_start_value_in_quote
        total_bot_return_percent = (total_bot_return - 1) * 100
        bot_return_logger.info(f"{np.round(total_bot_return_percent,4)} %")

        # log alpha
        alpha_percent = total_bot_return_percent - total_asset_return_percent
        alpha_logger.info(f"{np.round(alpha_percent,4)} %")

        # log amounts
        current_quote_amount = sum([bot["quote_to_spend"] for bot in bots])
        current_base_amount = sum([bot["base_to_sell"] for bot in bots])
        base_logger.info(current_base_amount)
        quote_logger.info(current_quote_amount)
        spread_logger.info(
            f"{np.average([bot['spread_in_percent'] for bot in bots])} %"
        )


if __name__ == "__main__":
    run_bot()
