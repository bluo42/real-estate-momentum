import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Set wide mode for the app and an appropriate title
st.set_page_config(page_title="Real Estate Momentum Strategy", layout="wide")

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load the transformed CSV data.
    Expected columns include:
      - Metro
      - Date
      - Price
      - SizeRank (optional, for filtering)
    """
    df = pd.read_csv("transformed_price_data.csv", parse_dates=["Date"])
    return df

df = load_data()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_last_price_on_or_before(metro_df, target_date):
    """
    Returns the last available price on or before the target_date.
    If no such price exists, returns None.
    """
    subset = metro_df[metro_df["Date"] <= target_date]
    if subset.empty:
        return None
    return subset.iloc[-1]["Price"]

def get_first_price_on_or_after(metro_df, target_date):
    """
    Returns the first available price on or after the target_date.
    If no such price exists, returns None.
    """
    subset = metro_df[metro_df["Date"] >= target_date]
    if subset.empty:
        return None
    return subset.iloc[0]["Price"]

def compute_momentum_at_date(data, asof_date):
    """
    Computes 1-year and 3-year backward momentum as well as forward growth (1-year and 3-year)
    for each metro based on a specified asof_date. If the forward price is not available,
    NA is returned.
    Returns a DataFrame with the columns:
      Metro, 1Y Momentum, 3Y Momentum, 1Y Fwd Growth %, 3Yr Fwd Growth
    """
    results = []
    for metro in data["Metro"].unique():
        metro_df = data[data["Metro"] == metro].sort_values("Date")
        # Current price as of asof_date (last price on or before the date)
        price_current = get_last_price_on_or_before(metro_df, asof_date)
        if price_current is None:
            continue

        # Backward prices
        price_1y_back = get_last_price_on_or_before(metro_df, asof_date - relativedelta(years=1))
        price_3y_back = get_last_price_on_or_before(metro_df, asof_date - relativedelta(years=3))
        
        # Forward prices (first price on or after target)
        price_1y_fwd = get_first_price_on_or_after(metro_df, asof_date + relativedelta(years=1))
        price_3y_fwd = get_first_price_on_or_after(metro_df, asof_date + relativedelta(years=3))
        
        momentum_1y = ((price_current / price_1y_back) - 1) * 100 if price_1y_back is not None else None
        momentum_3y = ((price_current / price_3y_back) - 1) * 100 if price_3y_back is not None else None
        fwd_growth_1y = ((price_1y_fwd / price_current) - 1) * 100 if price_1y_fwd is not None else None
        fwd_growth_3y = ((price_3y_fwd / price_current) - 1) * 100 if price_3y_fwd is not None else None
        
        results.append({
            "Metro": metro,
            "1Y Momentum": momentum_1y,
            "3Y Momentum": momentum_3y,
            "1Y Fwd Growth %": fwd_growth_1y,
            "3Yr Fwd Growth": fwd_growth_3y
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------
# Layout with Tabs: Dashboard & Historical Backtest
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Dashboard", "Historical Backtest"])

# ----------------------
# Dashboard Tab
# ----------------------
with tab1:
    
    st.title("Real Estate Momentum Strategy")
    col_left, _, col_right = st.columns([2, 0.1, 2])
    # Select Asof Date for momentum view (default to the latest date in the dataset)
    default_asof_date = df["Date"].max().date()


    with col_left:
        asof_date = st.date_input("Select Asof Date", value=date(2021, 12, 31), format="MM/DD/YYYY")
        asof_date = pd.to_datetime(asof_date)
        # Compute momentum view based on asof_date
        metros_data = df[(df["Metro"] != "United States") & (df["SizeRank"] <= 200)]
        momentum_view_df = compute_momentum_at_date(metros_data, asof_date)
        
        st.subheader(f"Momentum Data as of {asof_date.date()}")
        st.dataframe(momentum_view_df, hide_index=True)

        
        
    with col_right:
        # Primary and additional metro selections for detailed drilldown
        all_metros = sorted(metros_data["Metro"].unique())

        selected_metros = st.multiselect("Select Metros for Analysis", all_metros, default=['Los Angeles, CA'])

        fig_combined = go.Figure()
        for metro in selected_metros[:5]:
            metro_data = df[df["Metro"] == metro].sort_values("Date")
            fig_combined.add_trace(go.Scatter(
                x=metro_data["Date"],
                y=metro_data["Price"],
                mode="lines",
                name=metro
            ))
        fig_combined.update_layout(
            title="Price Trends Comparison",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        st.plotly_chart(fig_combined, use_container_width=True)
        # ---------------------------------------------------------
    # Download Data Section (remains at the bottom)
    # ---------------------------------------------------------
    st.header("Download Price Data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="transformed_price_data.csv",
        mime="text/csv",
    )
    st.markdown("Download the complete dataset.")
# ----------------------
# Historical Backtest Tab
# ----------------------
with tab2:
    st.title("Historical Backtest")
    
    # Create a layout with two columns - one for form, one for results
    col_form, col_results = st.columns([1, 2])
    
    with col_form:
        universe_option = st.selectbox("Select Universe", ["All", "State"])
        with st.form("backtest_form"):
            # Arrange the form inputs in two columns.
            col1, col2 = st.columns(2)
            with col1:
                
                if universe_option == "State":
                    states = sorted(df["StateName"].dropna().unique())
                    state_selected = st.selectbox("Select State", states)
                    universe_data = df[df["StateName"] == state_selected]
                else:
                    universe_data = df.copy()
                benchmark_metro = st.selectbox(
                    "Benchmark", 
                    sorted(df["Metro"].unique()),
                    index=sorted(df["Metro"].unique()).index("United States") if "United States" in df["Metro"].unique() else 0
                )
                percentile_cutoff = st.number_input("Percentile Cutoff", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
                
            
            with col2:
                
                from_date = st.date_input("From Date", value=df["Date"].min().date(), format="MM/DD/YYYY")
                to_date   = st.date_input("To Date", value=df["Date"].max().date(), format="MM/DD/YYYY")
                rebalance_freq = st.number_input("Rebalance frequency in months", min_value=1, value=3, step=1)
            
            submitted = st.form_submit_button("Run Backtest")
    
    if submitted:
        # Convert input dates to Timestamps.
        period_start = pd.to_datetime(from_date)
        period_end   = pd.to_datetime(to_date)
        
        # Pivot the data: rows are dates, columns are metros.
        price_pivot = universe_data.pivot(index="Date", columns="Metro", values="Price").sort_index()
        all_price_pivot = df.pivot(index="Date", columns="Metro", values="Price").sort_index()

        # Prepare lists to store period returns and rebalancing dates.
        top_returns = []
        bottom_returns = []
        avg_returns = []
        benchmark_returns = []
        rebalancing_dates = []
        
        # Generate rebalancing dates from period_start to period_end.
        current_date = period_start
        while current_date <= period_end:
            rebalancing_dates.append(current_date)
            current_date += relativedelta(months=rebalance_freq)
        
        # Loop through each rebalancing date and calculate vectorized returns.
        for T in rebalancing_dates:
            T_minus = T - relativedelta(years=1)
            T_plus = T + relativedelta(months=rebalance_freq)
            
            try:
                price_T = price_pivot.loc[:T].ffill().iloc[-1]
                price_T_minus = price_pivot.loc[:T_minus].ffill().iloc[-1]
            except IndexError:
                continue  # Skip if data not available
            
            # For T_plus, use backward fill.
            temp = price_pivot.loc[T_plus:]
            if not temp.empty:
                price_T_plus = temp.bfill().iloc[0]
            else:
                price_T_plus = pd.Series(np.nan, index=price_T.index)
            
            # Vectorized momentum and forward return calculations.
            momentum = (price_T / price_T_minus) - 1
            fwd_return = (price_T_plus / price_T) - 1
            
            # Determine number of metros to include based on the percentile cutoff.
            top_count = max(1, int(len(momentum) * (percentile_cutoff / 100)))
            sorted_momentum = momentum.sort_values(ascending=False)
            top_portfolio_return = fwd_return.loc[sorted_momentum.index[:top_count]].mean()
            bottom_portfolio_return = fwd_return.loc[sorted_momentum.index[-top_count:]].mean()
            universe_avg_return = fwd_return.mean()
            
            # Benchmark return calculation (for the selected benchmark metro).
            bench_series = all_price_pivot[benchmark_metro]
            try:
                bench_price_T = bench_series.loc[:T].ffill().iloc[-1]
            except IndexError:
                bench_price_T = np.nan
            bench_temp = bench_series.loc[T_plus:]
            if not bench_temp.empty:
                bench_price_T_plus = bench_temp.bfill().iloc[0]
            else:
                bench_price_T_plus = np.nan
            if pd.notnull(bench_price_T) and pd.notnull(bench_price_T_plus) and bench_price_T != 0:
                bench_return = (bench_price_T_plus / bench_price_T) - 1
            else:
                bench_return = np.nan
            
            # Append returns only if the top return is valid.
            if not np.isnan(top_portfolio_return):
                top_returns.append(top_portfolio_return)
                bottom_returns.append(bottom_portfolio_return)
                avg_returns.append(universe_avg_return)
                benchmark_returns.append(bench_return)
        
        # Convert lists to arrays and drop any periods with NA values.
        top_arr = np.array(top_returns)
        bottom_arr = np.array(bottom_returns)
        avg_arr = np.array(avg_returns)
        bench_arr = np.array(benchmark_returns)
        dates_arr = np.array(rebalancing_dates[:len(top_arr)])  # Ensure same length
        
        valid_mask = (~np.isnan(top_arr)) & (~np.isnan(bottom_arr)) & (~np.isnan(avg_arr)) & (~np.isnan(bench_arr))
        if valid_mask.sum() == 0:
            st.error("No valid backtest periods found with complete data.")
        else:
            top_arr = top_arr[valid_mask]
            bottom_arr = bottom_arr[valid_mask]
            avg_arr = avg_arr[valid_mask]
            bench_arr = bench_arr[valid_mask]
            dates_arr = dates_arr[valid_mask]
            
            # Create a DataFrame with returns and dates
            returns_df = pd.DataFrame({
                "Date": dates_arr,
                "Top": top_arr,
                "Universe Average": avg_arr,
                "Benchmark": bench_arr,
                "Bottom": bottom_arr
            })
            
            # Group by year and calculate annual returns
            returns_df["Year"] = pd.to_datetime(returns_df["Date"]).dt.year
            annual_returns = returns_df.groupby("Year").agg({
                "Top": lambda x: np.prod(1 + x) - 1,
                "Universe Average": lambda x: np.prod(1 + x) - 1,
                "Benchmark": lambda x: np.prod(1 + x) - 1,
                "Bottom": lambda x: np.prod(1 + x) - 1
            })
            
            # Calculate average annual returns
            avg_annual_returns = pd.DataFrame({
                "Strategy": ["Top", "Universe Average", "Benchmark", "Bottom"],
                "Average Annual Return (%)": [
                    annual_returns["Top"].mean() * 100,
                    annual_returns["Universe Average"].mean() * 100,
                    annual_returns["Benchmark"].mean() * 100,
                    annual_returns["Bottom"].mean() * 100
                ]
            })
            
            # Compute cumulative returns
            cum_top = np.cumprod(1 + top_arr) - 1
            cum_bottom = np.cumprod(1 + bottom_arr) - 1
            cum_avg = np.cumprod(1 + avg_arr) - 1
            cum_bench = np.cumprod(1 + bench_arr) - 1
            
            # Add cumulative returns to the results dataframe
            avg_annual_returns["Cumulative Return (%)"] = [
                cum_top[-1] * 100,
                cum_avg[-1] * 100,
                cum_bench[-1] * 100,
                cum_bottom[-1] * 100
            ]
            
            # Display results in the results column
            with col_results:
                # Create tabs for different visualizations
                result_tab1, result_tab2 = st.tabs(["Annual Returns", "Cumulative Performance"])
                
                with result_tab1:
                    # Plot annual returns as a bar chart
                    fig_annual = go.Figure()
                    for strategy in ["Top", "Universe Average", "Benchmark", "Bottom"]:
                        fig_annual.add_trace(go.Bar(
                            x=annual_returns.index,
                            y=annual_returns[strategy] * 100,
                            name=strategy
                        ))
                    
                    fig_annual.update_layout(
                        title="Annual Returns by Strategy",
                        xaxis_title="Year",
                        yaxis_title="Annual Return (%)",
                        template="plotly_white",
                        barmode="group"
                    )
                    st.plotly_chart(fig_annual, use_container_width=True)
                
                with result_tab2:
                    # Plot cumulative performance as a line graph
                    cumulative_df = pd.DataFrame({
                        "Date": dates_arr,
                        "Top": cum_top,
                        "Universe Average": cum_avg,
                        "Benchmark": cum_bench,
                        "Bottom": cum_bottom
                    })
                    
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(x=cumulative_df["Date"], y=cumulative_df["Top"]*100,
                                        mode="lines+markers", name="Top"))
                    fig_cum.add_trace(go.Scatter(x=cumulative_df["Date"], y=cumulative_df["Universe Average"]*100,
                                        mode="lines+markers", name="Universe Average"))
                    fig_cum.add_trace(go.Scatter(x=cumulative_df["Date"], y=cumulative_df["Benchmark"]*100,
                                        mode="lines+markers", name="Benchmark"))
                    fig_cum.add_trace(go.Scatter(x=cumulative_df["Date"], y=cumulative_df["Bottom"]*100,
                                        mode="lines+markers", name="Bottom"))
                    fig_cum.update_layout(title="Cumulative Performance Over Time",
                                    xaxis_title="Date",
                                    yaxis_title="Cumulative Return (%)",
                                    template="plotly_white")
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                # Display summary table with both annual and cumulative returns
                st.subheader("Performance Summary")
                st.dataframe(avg_annual_returns, hide_index=True)





