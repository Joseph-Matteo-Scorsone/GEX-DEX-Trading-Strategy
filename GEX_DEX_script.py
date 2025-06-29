import databento as db
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
DATABENTO_KEY = os.getenv("DATABENTO_KEY")
CLIENT = db.Historical(DATABENTO_KEY)

async def get_df(dataset, symbol, option_symbol, start, end):

    start_safe = start.replace(":", "_").replace("T", "_")
    end_safe = end.replace(":", "_").replace("T", "_")

    dataset_safe = dataset.replace(".", "_")
    symbol_safe = symbol.replace(".", "_")
    
    parquet_path = f"opt_underlying_{dataset_safe}_{symbol_safe}_{start_safe}_{end_safe}.parquet"

    start_dt = datetime.fromisoformat(start)
    start_YMD = start_dt.strftime("%Y-%m-%d")
    start_YMD_p1 = (start_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = pd.read_parquet(parquet_path)

        df.index = pd.to_datetime(df.index, utc=True)
        df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

        return df

    except Exception as e:

        # Get symbol mapping for the continuous contract
        symbol_map = CLIENT.symbology.resolve(
            dataset=dataset,
            symbols=symbol,
            stype_in="continuous",
            stype_out="instrument_id",
            start_date=start_YMD,
        )

        # Get instrument ID for the front month future
        front_month_id = int(symbol_map["result"][symbol][0]["s"])

        # Get all option definitions
        def_data_task = CLIENT.timeseries.get_range_async(
            dataset=dataset,
            schema="definition",
            symbols=[option_symbol],
            stype_in="parent",
            start=start_YMD_p1,
        )

        def_res = await def_data_task

        # Convert to DataFrame
        def_df = def_res.to_df()

        # Filter for options on the front month future
        opt_def_df = def_df[
            (def_df["instrument_class"].isin(("C", "P"))) &
            (def_df["underlying_id"] == front_month_id)
        ]

        # Get trades data for options on the front month future
        opt_trades_df_task = CLIENT.timeseries.get_range_async(
            dataset=dataset,
            schema="trades",
            symbols=opt_def_df["instrument_id"].to_list(),
            stype_in="instrument_id",
            start=start,
            end=end,
        )

        # Get MBP-1 data for the front month future
        fut_mbp_df_task = CLIENT.timeseries.get_range_async(
            dataset=dataset,
            schema="mbp-1",
            symbols=front_month_id,
            stype_in="instrument_id",
            start=start,
            end=end,
        )

        opt_trades_df_store, fut_mbp_df_store = await asyncio.gather(opt_trades_df_task, fut_mbp_df_task)
        opt_trades_df = opt_trades_df_store.to_df()
        fut_mbp_df = fut_mbp_df_store.to_df()

        fut_mbp_df = fut_mbp_df[["bid_px_00", "ask_px_00"]]
        fut_mbp_df = fut_mbp_df.sort_index()

        # Join options with their definitions
        opt_df = opt_trades_df.merge(
            opt_def_df,
            on="instrument_id",
            how="inner",
            suffixes=("", "_def"),
        ).set_index("ts_event")
        opt_df = opt_df.sort_index()

        # Join most recent underlying bid/ask with options trades
        df = pd.merge_asof(
            opt_df,
            fut_mbp_df,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        # Rename the columns
        df = df.rename(
            columns={
                "ask_px_00": "underlying_ask",
                "bid_px_00": "underlying_bid",
            },
        )

        df.to_parquet(parquet_path)

        return df

def d1_d2(spot, K, r, sigma, T):
    # Input validation
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return np.nan, np.nan
    
    # Avoid division by zero or overflow in denominator
    denom = sigma * np.sqrt(T)
    if denom < 1e-10:
        return np.nan, np.nan
    
    # Calculate d1 and d2
    d1 = (np.log(spot / K) + (r + (sigma ** 2) / 2) * T) / denom
    d2 = d1 - sigma * np.sqrt(T)
    
    # Check for extreme values
    if not np.isfinite(d1) or not np.isfinite(d2):
        return np.nan, np.nan
    
    return d1, d2

def blackScholesPrice(spot, K, r, sigma, T, is_call):
    d1, d2 = d1_d2(spot, K, r, sigma, T)
    
    if not np.isfinite(d1) or not np.isfinite(d2):
        return np.nan
    
    if is_call:
        return spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

def est_IV(premium, spot, K, r, T, is_call):
    sigma = 0.20
    tol = 1e-5
    max_iter = 100
    
    if T <= 0 or premium <= 0 or spot <= 0 or K <= 0:
        return np.nan
    
    for _ in range(max_iter):
        price = blackScholesPrice(spot, K, r, sigma, T, is_call)
        d1, _ = d1_d2(spot, K, r, sigma, T)
        
        if not np.isfinite(d1):
            return np.nan
        
        vega = spot * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
        diff = premium - price
        
        # Handle if close to value
        if abs(diff) < tol:
            break
        if abs(vega) < 1e-10: # Handle divide by zero
            break
        
        sigma += diff / vega # NR step
    
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan

def delta(d1, is_call):
    if is_call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def gamma(d1, spot, sigma, T):
    return norm.pdf(d1) / (spot * sigma * np.sqrt(T))

def calculate_gex_dex(df, contract_multiplier, path, r=0.05, default_iv=0.20):

    try:
        df = pd.read_parquet(path)

        return df

    except Exception as e:
        result_df = df.copy()
        
        print(f"Initial dataframe shape: {result_df.shape}")
        
        # Parse strike price and option type
        result_df["option_type"] = result_df["symbol_def"].str.extract(r"([PC])")
        result_df["strike_raw"] = result_df["symbol_def"].str.extract(r"[PC](\d+)").astype(float)
        result_df["strike_price"] = result_df["strike_raw"] / 100
        
        # Calculate spot price
        result_df["spot_price"] = (result_df["underlying_bid"] + result_df["underlying_ask"]) / 2
        
        # Calculate time to expiry
        result_df["time_to_expiry"] = (result_df["expiration"] - result_df.index).dt.total_seconds() / (365.25 * 24 * 3600)
        
        # Filter out invalid or extreme cases
        before_filter = len(result_df)
        result_df = result_df[
            (result_df["spot_price"] > 0) &
            (result_df["strike_price"] > 0) &
            (result_df["price"] > 0) &
            # Filter deep out-of-the-money options
            (result_df["strike_price"].between(result_df["spot_price"] * 0.5, result_df["spot_price"] * 1.5))
        ]
        after_filter = len(result_df)
        print(f"Filtered out {before_filter - after_filter} invalid/extreme options, {after_filter} remaining")
        
        if len(result_df) == 0:
            print("ERROR: No options remaining after filtering!")
            return result_df
        
        # Determine call/put
        result_df["is_call"] = result_df["option_type"] == "C"
        
        # Calculate implied volatility
        result_df["implied_vol"] = default_iv
        valid_prices = (result_df["price"] > 0) & (result_df["time_to_expiry"] > 0)
        
        iv_calculated = 0
        for idx in result_df[valid_prices].index:
            try:
                row = result_df.loc[idx]
                iv = est_IV(
                    premium=row["price"],
                    spot=row["spot_price"],
                    K=row["strike_price"],
                    r=r,
                    T=row["time_to_expiry"],
                    is_call=row["is_call"]
                )
                if np.isfinite(iv):
                    result_df.loc[idx, "implied_vol"] = iv
                    iv_calculated += 1
            except Exception as e:
                if iv_calculated < 5:
                    print(f"IV calculation failed for idx {idx}: {e}")
        
        print(f"Successfully calculated IV for {iv_calculated} options")
        
        # Calculate Greeks
        result_df["d1"] = np.nan
        result_df["option_delta"] = np.nan
        result_df["option_gamma"] = np.nan
        
        valid_greeks = (result_df["time_to_expiry"] > 0) & (result_df["implied_vol"] > 0)
        
        greeks_calculated = 0
        for idx in result_df[valid_greeks].index:
            try:
                row = result_df.loc[idx]
                d1, _ = d1_d2(
                    spot=row["spot_price"],
                    K=row["strike_price"],
                    r=r,
                    sigma=row["implied_vol"],
                    T=row["time_to_expiry"]
                )
                
                if np.isfinite(d1):
                    result_df.loc[idx, "d1"] = d1
                    result_df.loc[idx, "option_delta"] = delta(d1, row["is_call"])
                    result_df.loc[idx, "option_gamma"] = gamma(d1, row["spot_price"], row["implied_vol"], row["time_to_expiry"])
                    greeks_calculated += 1
            except Exception as e:
                if greeks_calculated < 5:
                    print(f"Greeks calculation failed for idx {idx}: {e}")
        
        print(f"Calculated Greeks for {greeks_calculated} options")
        
        # Remove rows with NaN Greeks
        before_clean = len(result_df)
        result_df = result_df.dropna(subset=["option_delta", "option_gamma"])
        print(f"Removed {before_clean - len(result_df)} rows with NaN Greeks")
        
        # Calculate exposures
        result_df["delta_exposure"] = result_df["option_delta"] * result_df["size"] * contract_multiplier * result_df["spot_price"]
        result_df["gamma_exposure"] = result_df["option_gamma"] * result_df["size"] * contract_multiplier * result_df["spot_price"] * result_df["spot_price"] / 100
        
        # Adjust for buy/sell side
        result_df.loc[result_df["side"] == "A", "delta_exposure"] *= -1
        result_df.loc[result_df["side"] == "A", "gamma_exposure"] *= -1

        result_df.to_parquet(path)
        
        return result_df

async def figs_and_details(df):
    # GEX/DEX over time
    fig1 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Delta Exposure Over Time", "Gamma Exposure Over Time", "Underlying Spot Price")
    )

    # DEX over time
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df["DEX"],
            name="DEX",
            line=dict(color="#1f77b4"),
            hovertemplate="Time: %{x}<br>DEX: $%{y:,.2f}"
        ),
        row=1, col=1
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)

    # GEX over time
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df["GEX"],
            name="GEX",
            line=dict(color="#ff0000"),
            hovertemplate="Time: %{x}<br>GEX: $%{y:,.2f}"
        ),
        row=2, col=1
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)

    # Spot price
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df["spot_price"],
            name="Spot Price",
            line=dict(color="#00ff00"),
            hovertemplate="Time: %{x}<br>Spot: $%{y:.2f}"
        ),
        row=3, col=1
    )

    # Update layout for dark theme
    fig1.update_layout(
        template="plotly_dark",
        height=800,
        width=1000,
        title_text="GEX/DEX and Spot Price Over Time",
        showlegend=False,
        xaxis3_title="Time",
        yaxis_title="Delta Exposure ($)",
        yaxis2_title="Gamma Exposure ($)",
        yaxis3_title="Spot Price ($)",
        xaxis3_tickangle=45,
        margin=dict(t=100, b=100)
    )

    # Update axes for grid
    for i in range(1, 4):
        fig1.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)", row=i, col=1)

    fig1.show()

    # GEX/DEX by strike price
    fig2 = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.1,
        subplot_titles=("Gamma Exposure by Strike Price", "Delta Exposure by Strike Price")
    )

    # Filter significant exposures
    significant_strikes = strike_exposure[abs(strike_exposure["gamma_exposure"]) > 1000]
    significant_delta_strikes = strike_exposure[abs(strike_exposure["delta_exposure"]) > 10000]

    # Current spot price
    current_spot = df["spot_price"].iloc[-1]

    # Gamma exposure
    fig2.add_trace(
        go.Bar(
            x=significant_strikes.index,
            y=significant_strikes["gamma_exposure"],
            name="Gamma Exposure",
            marker_color="#ff0000",
            opacity=0.7,
            hovertemplate="Strike: $%{x:.2f}<br>GEX: $%{y:,.2f}"
        ),
        row=1, col=1
    )
    fig2.add_vline(x=current_spot, line_dash="dash", line_color="#00ff00", row=1, col=1,
                annotation_text=f"Spot: ${current_spot:.2f}", annotation_position="top")
    fig2.add_hline(y=0, line_color="white", row=1, col=1)

    # Delta exposure
    fig2.add_trace(
        go.Bar(
            x=significant_delta_strikes.index,
            y=significant_delta_strikes["delta_exposure"],
            name="Delta Exposure",
            marker_color="#1f77b4",
            opacity=0.7,
            hovertemplate="Strike: $%{x:.2f}<br>DEX: $%{y:,.2f}"
        ),
        row=2, col=1
    )
    fig2.add_vline(x=current_spot, line_dash="dash", line_color="#00ff00", row=2, col=1,
                annotation_text=f"Spot: ${current_spot:.2f}", annotation_position="top")
    fig2.add_hline(y=0, line_color="white", row=2, col=1)

    # Update layout for dark theme
    fig2.update_layout(
        template="plotly_dark",
        height=700,
        width=1000,
        title_text="GEX/DEX by Strike Price",
        showlegend=False,
        xaxis2_title="Strike Price ($)",
        yaxis_title="Gamma Exposure ($)",
        yaxis2_title="Delta Exposure ($)",
        margin=dict(t=100, b=50)
    )

    # Update axes for grid
    for i in range(1, 3):
        fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)", row=i, col=1)
        fig2.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)", row=i, col=1)

    fig2.show()

async def backtest_GEX_DEX(df, contract_multiplier, initial_balance, percent_per_trade):
    GEX_THRESHOLD = 10_000
    DEX_THRESHOLD = 100_000

    account_balance = initial_balance
    number_of_contracts = 0  # Positive for long, negative for short, 0 for flat
    entry_price = 0.0
    realized_pnl = 0.0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0

    results = pd.DataFrame({
        "spot_price": df["spot_price"],
        "GEX": df["GEX"],
        "DEX": df["DEX"],
        "number_of_contracts": 0,
        "entry_price": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "account_balance": initial_balance,
    }, index=df.index)

    for i in range(1, len(df)):
        prev_dex = df["DEX"].iloc[i - 1]
        prev_gex = df["GEX"].iloc[i - 1]
        dex = df["DEX"].iloc[i]
        gex = df["GEX"].iloc[i]
        current_spot = df["spot_price"].iloc[i]

        pos_flip = (
            prev_gex > GEX_THRESHOLD and gex < GEX_THRESHOLD and dex > DEX_THRESHOLD
        )
        neg_flip = (
            prev_gex < -GEX_THRESHOLD and gex > -GEX_THRESHOLD and dex < -DEX_THRESHOLD
        )
        
        # Calculate unrealized PnL in dollars
        unrealized_pnl = 0.0
        if number_of_contracts != 0 and entry_price != 0:
            if number_of_contracts > 0:  # Long
                unrealized_pnl = (current_spot - entry_price) * contract_multiplier * number_of_contracts
            elif number_of_contracts < 0:  # Short
                unrealized_pnl = (entry_price - current_spot) * contract_multiplier * abs(number_of_contracts)

        # Position management
        realized_pnl = 0.0
        if pos_flip and number_of_contracts <= 0:
            # Close short position
            if number_of_contracts < 0:
                realized_pnl = (entry_price - current_spot) * contract_multiplier * abs(number_of_contracts)
                account_balance += realized_pnl
                total_trades += 1
                if realized_pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

            # Open long position
            risk_per_trade = account_balance * (percent_per_trade / 100)
            number_of_contracts = max(1, int(risk_per_trade / (current_spot * contract_multiplier)))
            entry_price = current_spot

        elif neg_flip and number_of_contracts >= 0:
            # Close long position
            if number_of_contracts > 0:
                realized_pnl = (current_spot - entry_price) * contract_multiplier * number_of_contracts
                account_balance += realized_pnl
                total_trades += 1
                if realized_pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

            # Open short position
            risk_per_trade = account_balance * (percent_per_trade / 100)
            number_of_contracts = -int(risk_per_trade / (current_spot * contract_multiplier))
            number_of_contracts = min(-1, number_of_contracts)  # Ensure at least -1 contract
            entry_price = current_spot

        total_pnl = realized_pnl + unrealized_pnl

        # Update results DataFrame
        results.iloc[i, results.columns.get_loc("number_of_contracts")] = number_of_contracts
        results.iloc[i, results.columns.get_loc("entry_price")] = entry_price
        results.iloc[i, results.columns.get_loc("realized_pnl")] = realized_pnl
        results.iloc[i, results.columns.get_loc("unrealized_pnl")] = unrealized_pnl
        results.iloc[i, results.columns.get_loc("total_pnl")] = total_pnl
        results.iloc[i, results.columns.get_loc("account_balance")] = account_balance + unrealized_pnl

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    return results, win_rate, total_trades

async def plot_account_balance(df):
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["account_balance"],
            mode="lines",
            name="Account Balance",
            line=dict(color="blue")
        )
    )
    
    fig.update_layout(
        title="Account Balance Over Time",
        xaxis_title="Date",
        yaxis_title="Account Balance ($)",
        template="plotly_dark",
    )
    
    fig.show()

if __name__ == "__main__":

    dataset = "GLBX.MDP3"
    symbol = "CL.c.0"
    option_symbol = "LO.OPT"
    start = "2025-06-01T13:00:00"
    end = "2025-06-23T17:00:00"
    contract_multiplier = 1_000
    
    df = asyncio.run(get_df(dataset, symbol, option_symbol, start, end))

    print(df[["strike_price", "instrument_class", "price", "underlying_bid", "underlying_ask"]].head())

    # Calculate GEX and DEX
    start_safe = start.replace(":", "_").replace("T", "_")
    end_safe = end.replace(":", "_").replace("T", "_")

    dataset_safe = dataset.replace(".", "_")
    symbol_safe = symbol.replace(".", "_")
    gex_dex_path = f"opt_gex_dex_iv_{symbol_safe}_{dataset_safe}_{start_safe}_{end_safe}.parquet"

    df_with_greeks = calculate_gex_dex(df, contract_multiplier, gex_dex_path)

    # Aggregate exposures by time
    gex_dex_summary = df_with_greeks.groupby(df_with_greeks.index).agg({
        "delta_exposure": "sum",
        "gamma_exposure": "sum",
        "spot_price": "mean"  # Average spot price for that timestamp
    }).rename(columns={
        "delta_exposure": "DEX",
        "gamma_exposure": "GEX"
    })

    print("GEX and DEX Summary:")
    print(gex_dex_summary.head())

    # Aggregate by strike price to see exposure distribution
    strike_exposure = df_with_greeks.groupby("strike_price").agg({
        "delta_exposure": "sum",
        "gamma_exposure": "sum",
        "size": "sum"
    }).sort_values("strike_price")

    print("\nExposure by Strike Price:")
    print(strike_exposure.head(10))

    # Net exposures across all positions
    total_dex = df_with_greeks["delta_exposure"].sum()
    total_gex = df_with_greeks["gamma_exposure"].sum()

    print(f"\nTotal DEX: ${total_dex:,.2f}")
    print(f"Total GEX: ${total_gex:,.2f}")

    asyncio.run(figs_and_details(gex_dex_summary))

    results_df, win_rate, total_trades = asyncio.run(backtest_GEX_DEX(gex_dex_summary, contract_multiplier, 100_000.00, 0.50))
    
    print(f"Total Trades: {total_trades}.\nWin rate: {win_rate}.")
    asyncio.run(plot_account_balance(results_df))
