import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy.optimize import minimize

# File IDs from the link
link_input = '1A9aTYrAWyEmIBHAw56XkM-H8ArF_kmuX'
download_url = f"https://drive.google.com/uc?id={link_input}"
input = pd.read_csv(download_url)

choose_from = ['Basic Materials', 'Energy', 'Financials', 'Industrials', 'Information Technology', 'Consumer Staples', 'Utilities', 'Healthcare', 'Consumer Discretionary']

# Convert the 'DateTime' column to datetime
input['DateTime'] = pd.to_datetime(input['DateTime'], format='%d.%m.%Y %H:%M')

# Sidebar date selectors
st.sidebar.title("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2019-01-07'))
end_date = st.sidebar.date_input("End Date", value=input['DateTime'].max())

# Filter the data based on the selected date range
input_filtered = input[(input['DateTime'] >= pd.to_datetime(start_date)) & (input['DateTime'] <= pd.to_datetime(end_date))]

# Check if the end date is before the start date
if start_date > end_date:
    st.error("End Date must be after Start Date.")
else:
    input_filtered = input_filtered.dropna()

    # Load predictions for each sector
    predictions_dict = {}
    #predictions_path = os.path.join(os.path.dirname(__file__), "predictions")
    for sector in choose_from:
        #predictions_dict[sector] = pd.read_csv(f"{predictions_path}/{sector}_predictions.csv")
        predictions_dict[sector] = pd.read_csv(f"predictions_sectors/{sector}_predictions.csv")

    valid_dates = predictions_dict[choose_from[0]]['DateTime']  # Get the 'DateTime' values from the first sector
    input_filtered = input_filtered[input_filtered['DateTime'].isin(valid_dates)]

    # Sidebar sector selection
    selected_sectors = []
    with st.sidebar:
        st.title("Select Sectors to Display")
        for sector in choose_from:
            if st.checkbox(f"Show {sector}", value=True):
                selected_sectors.append(sector)

    # Plot sector performance
    fig = go.Figure()
    for sector in selected_sectors:
        # Get the prediction data for the sector
        prediction = predictions_dict[sector]
        
        # Adjust prediction length
        prediction = prediction.head(len(input_filtered))
        action = ["Buy" if val == 1 else "Sell" for val in prediction['Predicted_Cluster']]
        
        # Normalize sector data to start from the same point
        sector_values = input_filtered[f'U.S. {sector}']
        normalized_values = sector_values / sector_values.iloc[0]  # Normalize to start at 1
        
        # Add sector data to the plot
        fig.add_trace(go.Scatter(
            x=input_filtered['DateTime'], 
            y=normalized_values,  # Use normalized values
            mode='lines', 
            name=sector,
            hovertemplate="%{customdata}<br>" + "%{x|%Y-%m-%d}",
            customdata=action
        ))

    # Customize layout for sector chart
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_dark'
    )

    st.header("U.S. Sector ETFs - Buy/ Sell Signals")
    st.text("Hover over a specific day in the graph to see the model's buy/sell recommendation. \n Please do not choose an earlier date than the 7th of January 2019.")
    st.plotly_chart(fig)

    # Calculate daily returns
    for sector in choose_from:
        input[f'U.S. {sector}'] = input[f'U.S. {sector}'].pct_change()
    
    input = input.dropna()
    timeline_df = pd.to_datetime(predictions_dict['Basic Materials']['DateTime'])
    
    # **Actual Daily and Cumulative Returns Calculation**
    actual_daily_returns = []
    for i in range(len(timeline_df)):
        daily = 0
        for sector in selected_sectors:
            sec_daily_return = predictions_dict[sector].iloc[i, 2]  # Assuming the third column has daily return
            daily += sec_daily_return
        actual_daily_returns.append(daily / len(selected_sectors))
    
    # Calculate cumulative returns for the sectors
    actual_sectors_cumulative = []
    cumulative_product = 1
    for daily_return in actual_daily_returns:
        cumulative_product *= (1 + daily_return)
        actual_sectors_cumulative.append(cumulative_product - 1)

    sharpe_ratio_actual = ((1 + actual_sectors_cumulative[-1]) ** (1/5.8) - 1 - 0.011) / (np.std(actual_daily_returns) * np.sqrt(252))


    # MIN VAR portfolio
    daily_returns = []
    was_out_of_market = True

    # Go through all trading days
    for i in range(len(timeline_df)-1):
        daily = 0
        number_bullish = 0
        sectors_bullish = ['DateTime']

        # This is the transaction cost incurred evry time we go in the market
        if was_out_of_market == True:
            daily = -0.001

        # Go through every sector for a given trading day
        for sector in choose_from:

            # If a sector is predicted bullish, include it in that day's portfolio
            cluster = predictions_dict[sector]['Predicted_Cluster'].iloc[i]
            if cluster == 1:
                number_bullish += 1
                sectors_bullish.append(f'U.S. {sector}')


        # Define the date range for filtering
        end_date = timeline_df.iloc[i]
        start_date = end_date - pd.DateOffset(months=1)

        # Select the relevant columns and filter rows by date range
        daily_df = input[sectors_bullish]
        daily_df = daily_df[(daily_df['DateTime'] >= start_date) & (daily_df['DateTime'] <= end_date)]
        returns_df = daily_df.drop(columns='DateTime')

        if returns_df.empty:
            daily_returns.append(0.011/365)
            was_out_of_market = True

        elif len(sectors_bullish) <=0:
            daily_returns.append(0.011/365)
            was_out_of_market = True

        else:
            was_out_of_market = False
            
            # Calculate covariance matrix
            cov_matrix = returns_df.cov().values
            
            # Number of assets
            num_assets = len(sectors_bullish) - 1  # Excluding 'DateTime'
            
            # Define the objective function (portfolio variance)
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints: weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds for weights: (0,1) for each asset
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1 / num_assets] * num_assets)
            
            # Optimize
            result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Get the optimized weights and store them
            weights = result.x

            next_day = input[input['DateTime'] > end_date].iloc[0]        
            next_day = next_day[[col for col in sectors_bullish if col != 'DateTime']]

            daily += np.dot(weights, next_day)
            daily_returns.append(daily)


    mvp_cumulative_returns = [0]
    cumulative_product = 1
    for daily_return in daily_returns:
        cumulative_product *= (1 + daily_return)
        mvp_cumulative_returns.append(cumulative_product - 1)

    sharpe_ratio_minvar = ((1 + mvp_cumulative_returns[-1]) ** (1/5.8) - 1 - 0.011) / (np.std(daily_returns) * np.sqrt(252))
  



    
    
    # **Equally Weighted Portfolio Calculation**
    daily_returns = []
    was_out_of_market = True

    for i in range(len(timeline_df)-1):
        daily = 0
        number_bullish = 0

        if was_out_of_market:
            daily = -0.001

        for industry in selected_sectors:
            cluster = predictions_dict[industry]['Predicted_Cluster'].iloc[i]
            if cluster == 1:
                number_bullish += 1
                daily += predictions_dict[industry]['Actual_Daily_Returns'].iloc[i+1]

        if daily == 0:
            was_out_of_market = True
            daily_returns.append(0.011/365)
        elif number_bullish <= 0:
            was_out_of_market = True
            daily_returns.append(0.011/365)
        else:
            was_out_of_market = False
            daily_returns.append(daily / number_bullish)

    equallyweighted_sectors_cumulative = [0]
    cumulative_product = 1
    for daily_return in daily_returns:
        cumulative_product *= (1 + daily_return)
        equallyweighted_sectors_cumulative.append(cumulative_product - 1)

    sharpe_ratio_equallyweighted = ((1 + equallyweighted_sectors_cumulative[-1]) ** (1/5.8) - 1 - 0.011) / (np.std(daily_returns) * np.sqrt(252))


    # ADD SP500
    sp500 = input[['DateTime','SPDR S&P 500 ETF Trust']]
    valid_dates = predictions_dict[choose_from[0]]['DateTime']
    sp500 = sp500[sp500['DateTime'].isin(valid_dates)]
    sp500_returns = sp500['SPDR S&P 500 ETF Trust'].pct_change()
    sp500_cumulative = (1 + sp500_returns).cumprod() - 1
    
    # **Plot Both Cumulative Returns on the Same Graph**
    fig_cumulative = go.Figure()

    # Add the actual sector cumulative return
    
    fig_cumulative.add_trace(go.Scatter(
        x=input_filtered['DateTime'], 
        y=actual_sectors_cumulative[:len(input_filtered)], 
        mode='lines', 
        name="Market Cumulative Return",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            + "Market: %{y:.2f}"
            + "<extra></extra>"
        )
    ))

    # Add the minimum variance portfolio cumulative return
    fig_cumulative.add_trace(go.Scatter(
        x=input_filtered['DateTime'], 
        y=mvp_cumulative_returns[:len(input_filtered)], 
        mode='lines', 
        name="Minimum Variance Portfolio",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            + "MVP: %{y:.2f}"
            + "<extra></extra>"
        )
    ))

    # Add the equally weighted portfolio cumulative return
    fig_cumulative.add_trace(go.Scatter(
        x=input_filtered['DateTime'], 
        y=equallyweighted_sectors_cumulative[:len(input_filtered)], 
        mode='lines', 
        name="Equally Weighted Cumulative Return",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            + "Equally Weighted Portfolio: %{y:.2f}"
            + "<extra></extra>"
        )
    ))

    fig_cumulative.add_trace(go.Scatter(
        x=input_filtered['DateTime'], 
        y=sp500_cumulative[:len(input_filtered)], 
        mode='lines', 
        name="S&P500 Cumulative Return",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            + "S&P500: %{y:.2f}"
            + "<extra></extra>"
        )
    ))

    # Customize layout for cumulative returns chart
    fig_cumulative.update_layout(
        title={
            'text': f'Sharpe Ratios <br> Market: {sharpe_ratio_actual:.2f}<br> Minimum Variance Portfolio: {sharpe_ratio_minvar:.2f} <br> Equally Weighted Portfolio: {sharpe_ratio_equallyweighted:.2f}',
            'x': 0.5,  # Center the title
            'y': 0.95,  # Adjust vertical placement of title
            'xanchor': 'center',
            'yanchor': 'top',
        },
        margin=dict(t=100),  # Increase top margin to add space
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        template='plotly_dark'
    )


    st.header("Cumulative Returns of Selected Sectors")
    st.text("Choosing fewer sectors can negatively impact model performance, reducing diversification.")
    st.plotly_chart(fig_cumulative)
