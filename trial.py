import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

# File IDs from the link
link_input = '1-CrON3CcQAQck-xX8OzZlWeyeThdfZHc'
download_url = f"https://drive.google.com/uc?id={link_input}"
input = pd.read_csv(download_url)

choose_from = ['MSCI World', 'MSCI World Momentum', 'MSCI World Quality', 'MSCI World High Dividend Yield', 'MSCI World Volatility', 
               'MSCI World Equal Weight', 'MSCI World Small Cap', 'MSCI World Prime Value', 'MSCI World Risk Weighted']

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
    predictions_path = os.path.join(os.path.dirname(__file__), "predictions")
    for sector in choose_from:
        predictions_dict[sector] = pd.read_csv(f"{predictions_path}/{sector}_predictions.csv")

    # Sidebar sector selection
    selected_sectors = []
    with st.sidebar:
        st.title("Select Sectors to Display")
        for sector in choose_from:
            if st.checkbox(f"Show {sector}", value=True):
                selected_sectors.append(sector)

    # Plot sector performance
    fig = go.Figure()
    # Normalize and plot each sector
    for sector in selected_sectors:
        # Get the prediction data for the sector
        prediction = predictions_dict[sector]
        
        # Adjust prediction length
        prediction = prediction.head(len(input_filtered))
        action = ["Buy" if val == 1 else "Sell" for val in prediction['Predicted_Cluster']]
        
        # Normalize sector data to start from the same point
        sector_values = input_filtered[f'{sector}']
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
        input[f'{sector}'] = input[f'{sector}'].pct_change()
    
    input = input.dropna()
    timeline_df = pd.to_datetime(predictions_dict['MSCI World']['DateTime'])
    
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

        if number_bullish <= 0:
            was_out_of_market = True
            daily_returns.append(0.011/365)
        elif daily == 0:
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

    # **Plot Both Cumulative Returns on the Same Graph**
    fig_cumulative = go.Figure()

    # Add the actual sector cumulative return
    fig_cumulative.add_trace(go.Scatter(
        x=input_filtered['DateTime'], 
        y=actual_sectors_cumulative[:len(input_filtered)], 
        mode='lines', 
        name="Actual Cumulative Return",
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            + "Actual: %{y:.2f}"
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

    # Customize layout for cumulative returns chart
    fig_cumulative.update_layout(
        title=f'Sharpe Ratios \n Market: {sharpe_ratio_actual:.2f}, Equally Weighted Portfolio: {sharpe_ratio_equallyweighted:.2f}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        template='plotly_dark'
    )

    st.header("Cumulative Returns of Selected Sectors")
    st.text("Choosing fewer than 7 sectors will severely impact model performance, reducing diversification.")
    st.plotly_chart(fig_cumulative)
