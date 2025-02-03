import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load the transformed CSV data.
    The CSV is expected to have the following columns:
      - Metro
      - Date
      - Price
      - SizeRank  (required for the backtest to limit to the top 100 most populous metros;
                   USA should have SizeRank = 0)
    """
    df = pd.read_csv("transformed_price_data.csv", parse_dates=["Date"])
    return df

df = load_data()

# ---------------------------------------------------------
# Helper Function: Get Last Price On or Before a Target Date
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

# ---------------------------------------------------------
# PART 1: Summary – 1-Year Momentum Dashboard
# ---------------------------------------------------------
st.title("Real Estate Momentum Strategy Dashboard")

st.markdown("""
This dashboard showcases a momentum strategy based on the **last 1‑year momentum** of metros.
Below, only metros with available 1‑year data are ranked by their recent momentum.
If the `"SizeRank"` column is available, only the top 100 most populous metros are included.
""")

# Compute the 1‑year momentum for each metro.
def compute_recent_momentum(data):
    results = []
    for metro in data["Metro"].unique():
        metro_df = data[data["Metro"] == metro].sort_values("Date")
        # Use the metro's latest available date as "today"
        latest_date = metro_df["Date"].max()
        one_year_ago = latest_date - relativedelta(years=1)
        price_one_year_ago = get_last_price_on_or_before(metro_df, one_year_ago)
        latest_price = get_last_price_on_or_before(metro_df, latest_date)
        if price_one_year_ago is None or latest_price is None:
            continue
        momentum = (latest_price / price_one_year_ago) - 1
        results.append({
            "Metro": metro,
            "Recent Momentum (%)": momentum * 100,
            "Latest Date": latest_date,
            "Price One Year Ago": price_one_year_ago,
            "Latest Price": latest_price
        })
    return pd.DataFrame(results)

# Limit to top 100 most populous metros if SizeRank is available.
if "SizeRank" in df.columns:
    summary_data = df[df["SizeRank"] <= 100]
else:
    summary_data = df

recent_momentum_df = compute_recent_momentum(summary_data)
recent_momentum_df = recent_momentum_df.sort_values("Recent Momentum (%)", ascending=False)

# Display top 5 and bottom 5 metros by recent 1‑year momentum.
top_performers = recent_momentum_df.head(5)
bottom_performers = recent_momentum_df.sort_values("Recent Momentum (%)", ascending=True).head(5)

st.header("Top and Bottom Metros by 1‑Year Momentum")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Highest 1‑Year Momentum")
    st.table(top_performers[["Metro", "Recent Momentum (%)"]].reset_index(drop=True))
with col2:
    st.subheader("Lowest 1‑Year Momentum")
    st.table(bottom_performers[["Metro", "Recent Momentum (%)"]].reset_index(drop=True))

# ---------------------------------------------------------
# Metro Drilldown Section
# ---------------------------------------------------------
st.header("Metro Drilldown")
metro_list = sorted(df["Metro"].unique())
selected_metro = st.selectbox("Select a metro to analyze", options=metro_list)
metro_data = df[df["Metro"] == selected_metro].sort_values("Date")

# Plot the price trend using Plotly
fig = px.line(metro_data, x="Date", y="Price", 
              title=f"Price Trend for {selected_metro}",
              labels={"Price": "Price", "Date": "Date"})
st.plotly_chart(fig, use_container_width=True)

# Compute and display the selected metro's 1‑year momentum.
selected_latest_date = metro_data["Date"].max()
selected_one_year_ago = selected_latest_date - relativedelta(years=1)
price_one_year_ago = get_last_price_on_or_before(metro_data, selected_one_year_ago)
price_latest = get_last_price_on_or_before(metro_data, selected_latest_date)
if price_one_year_ago is not None and price_latest is not None:
    metro_momentum = (price_latest / price_one_year_ago) - 1
    st.markdown(f"**1‑Year Momentum for {selected_metro}:** {metro_momentum * 100:.2f}%")
else:
    st.markdown("Not enough data to compute 1‑Year Momentum for this metro.")

st.header("Download Price Data")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="transformed_price_data.csv",
    mime="text/csv",
)
st.markdown("Download the complete dataset.")

# ---------------------------------------------------------
# PART 1.5: Metro Comparison Section
# ---------------------------------------------------------
st.header("Metro Comparison")

# Initialize session state list for comparison if it doesn't exist.
if "compare_metros" not in st.session_state:
    st.session_state.compare_metros = []

# Provide a selectbox for metros not already in the list.
available_metros = sorted([m for m in df["Metro"].unique() if m not in st.session_state.compare_metros])
selected_to_add = st.selectbox("Select a metro to add for comparison", options=available_metros)

# Add Metro button (up to 5 metros allowed).
if st.button("Add Metro"):
    if len(st.session_state.compare_metros) < 5:
        st.session_state.compare_metros.append(selected_to_add)
    else:
        st.error("You can only add up to 5 metros for comparison.")

# Display the list of metros selected for comparison.
st.write("Metros selected for comparison:", st.session_state.compare_metros)

# If there are selected metros, show a multi-metro price comparison chart.
if st.session_state.compare_metros:
    fig_compare = go.Figure()
    for metro in st.session_state.compare_metros:
        metro_df = df[df["Metro"] == metro].sort_values("Date")
        fig_compare.add_trace(go.Scatter(
            x=metro_df["Date"],
            y=metro_df["Price"],
            mode="lines",
            name=metro
        ))
    fig_compare.update_layout(
        title="Metro Price Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# ---------------------------------------------------------
# PART 2: Historical Backtest of Momentum Strategy
# ---------------------------------------------------------
st.title("Historical Backtest of Momentum Strategy")

st.markdown("""
In this section we “backtest” a strategy where, for each historical month we:
- Compute each metro’s past 1‑year momentum (from **T – 1 year** to **T**),
- Limit to the top 100 most populous metros,
- Rank the metros by momentum,
- Select the **Top 10** (highest momentum) and **Bottom 10** (lowest momentum),
- Compute the forward 1‑year return (from **T** to **T + 1 year**) for each group,
- Also compute the USA forward return (for the metro with `SizeRank == 0`).

The table below will show the average 1‑year forward return (in %) for the Top 10, Bottom 10, and USA for each backtest month.
""")

if "SizeRank" not in df.columns:
    st.error("The Historical Backtest requires a 'SizeRank' column to limit to the top 100 most populous metros (USA should have SizeRank = 0).")
else:
    # Limit to top 100 most populous metros (excluding USA for the ranking groups)
    metros_data = df[(df["Metro"] != "United States") & (df["SizeRank"] <= 100)]
    
    # Also extract the USA data (SizeRank == 0)
    usa_data = df[df["SizeRank"] == 0]
    
    # Define the valid range for the backtest start date.
    global_valid_start = df["Date"].min() + relativedelta(years=1)
    global_valid_end = df["Date"].max() - relativedelta(years=1)
    st.markdown(f"Valid backtest start dates range from **{global_valid_start.date()}** to **{global_valid_end.date()}**")
    
    # Let the user select the backtest period.
    period_selection = st.date_input("Select Backtest Period (start and end)", 
                                     value=(global_valid_start.date(), global_valid_end.date()))
    if isinstance(period_selection, tuple) and len(period_selection) == 2:
        period_start = pd.to_datetime(period_selection[0])
        period_end = pd.to_datetime(period_selection[1])
    else:
        period_start = pd.to_datetime(period_selection)
        period_end = pd.to_datetime(period_selection)
    
    # Add a button to run the backtest.
    if st.button("Run Backtest"):
        backtest_results = {}
        current_date = period_start.replace(day=1)
        while current_date <= period_end:
            T = current_date                      # Backtest start date for this iteration.
            T_minus = T - relativedelta(years=1)    # For momentum calculation
            T_plus  = T + relativedelta(years=1)     # For forward return
            
            backtest_rows = []
            for metro in metros_data["Metro"].unique():
                metro_df = metros_data[metros_data["Metro"] == metro].sort_values("Date")
                price_T_minus = get_last_price_on_or_before(metro_df, T_minus)
                price_T = get_last_price_on_or_before(metro_df, T)
                price_T_plus = get_last_price_on_or_before(metro_df, T_plus)
                
                # Exclude metros with any missing data.
                if price_T_minus is None or price_T is None or price_T_plus is None:
                    continue
                
                momentum = (price_T / price_T_minus) - 1         # Past 1‑year return
                forward_return = (price_T_plus / price_T) - 1       # Next 1‑year return
                
                backtest_rows.append({
                    "Metro": metro,
                    "Momentum": momentum,
                    "Forward Return": forward_return
                })
            
            # Create a DataFrame for this month.
            month_df = pd.DataFrame(backtest_rows)
            
            # Compute USA forward return for this month.
            usa_forward_return = np.nan
            if not usa_data.empty:
                usa_df = usa_data.sort_values("Date")
                price_T_minus_usa = get_last_price_on_or_before(usa_df, T_minus)
                price_T_usa = get_last_price_on_or_before(usa_df, T)
                price_T_plus_usa = get_last_price_on_or_before(usa_df, T_plus)
                if (price_T_minus_usa is not None) and (price_T_usa is not None) and (price_T_plus_usa is not None):
                    usa_forward_return = (price_T_plus_usa / price_T_usa) - 1
            
            if month_df.empty or len(month_df) < 10:
                # Not enough data; record as NA.
                backtest_results[T.strftime("%Y-%m")] = {"Top 10": np.nan, "Bottom 10": np.nan, "USA": np.nan}
            else:
                # Rank metros by momentum (highest first).
                month_df = month_df.sort_values("Momentum", ascending=False).reset_index(drop=True)
                top_10 = month_df.head(10)
                bottom_10 = month_df.tail(10)
                top_10_avg = top_10["Forward Return"].mean() * 100    # as percentage
                bottom_10_avg = bottom_10["Forward Return"].mean() * 100
                backtest_results[T.strftime("%Y-%m")] = {
                    "Top 10": top_10_avg,
                    "Bottom 10": bottom_10_avg,
                    "USA": usa_forward_return * 100 if not np.isnan(usa_forward_return) else np.nan
                }
            
            current_date += relativedelta(months=1)
        
        # Create a DataFrame from the backtest_results dictionary.
        backtest_table = pd.DataFrame(backtest_results)
        
        # Ensure columns (months) are sorted chronologically.
        sorted_cols = sorted(backtest_table.columns, key=lambda x: pd.to_datetime(x, format="%Y-%m"))
        backtest_table = backtest_table[sorted_cols]
        
        st.subheader("Historical Backtest Results")
        st.markdown("Average 1‑Year Forward Return (%) by Backtest Month")
        st.dataframe(backtest_table)
        
        # ---------------------------------------------------------
        # Cumulative Performance Plot
        # ---------------------------------------------------------
        st.subheader("Cumulative Performance")
        st.markdown("The chart below shows the cumulative performance (compounded returns) for the Top 10, Bottom 10, and USA strategies.")
        
        # Calculate cumulative performance from monthly forward returns.
        # Convert monthly returns (%) into multipliers, then compound.
        cumulative = (1 + backtest_table/100).cumprod(axis=1) - 1
        cumulative = cumulative * 100  # convert back to percent
        
        # Create a Plotly line chart.
        fig_cum = go.Figure()
        for strategy in cumulative.index:
            fig_cum.add_trace(go.Scatter(
                x=cumulative.columns,
                y=cumulative.loc[strategy],
                mode='lines+markers',
                name=strategy
            ))
        fig_cum.update_layout(
            title="Cumulative Performance of Strategies",
            xaxis_title="Backtest Month",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig_cum, use_container_width=True)
