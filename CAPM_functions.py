# capm_functions.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


# Function to plot interactive plotly chart
def interactive_plot(df):
    fig = px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)
    fig.update_layout(width=450, margin=dict(l=20, r=20, t=50, b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


# Function to normalize the prices based on the initial prices
def normalize(df_2):
    df = df_2.copy()
    for i in df.columns[1:]:
        df[i] = df[i] / df[i].iloc[0]
    return df


# Function to calculate daily return (percentage)
def daily_return(df):
    """
    Calculate daily returns for all stocks and S&P 500
    """
    df_daily_return = df.copy()

    for i in df.columns[1:]:
        for j in range(1, len(df)):
            df_daily_return[i].iloc[j] = ((df[i].iloc[j] - df[i].iloc[j - 1]) / df[i].iloc[j - 1]) * 100
        df_daily_return[i].iloc[0] = 0

    return df_daily_return


# Function to calculate daily returns (decimal format - better for CAPM calculations)
def daily_return_decimal(df):
    """
    Calculate daily returns in decimal format (0.01 = 1%)
    """
    df_return = df.copy()

    for column in df.columns[1:]:
        df_return[column] = df[column].pct_change()

    # Fill NaN values with 0 for the first row
    df_return = df_return.fillna(0)
    return df_return


# Function to calculate beta for a stock
def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta using covariance and variance
    beta = Cov(stock, market) / Var(market)
    """
    # Remove any rows with NaN values
    valid_data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()

    if len(valid_data) < 2:
        return np.nan

    covariance = np.cov(valid_data['stock'], valid_data['market'])[0, 1]
    market_variance = np.var(valid_data['market'])

    if market_variance == 0:
        return np.nan

    beta = covariance / market_variance
    return beta


# Function to calculate CAPM expected return
def calculate_expected_return(risk_free_rate, beta, market_return):
    """
    Calculate expected return using CAPM formula:
    E(R) = Rf + β * (Rm - Rf)
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)


# Function to calculate alpha (excess return)
def calculate_alpha(actual_return, expected_return):
    """
    Calculate alpha (α) - excess return
    α = Actual Return - Expected Return
    """
    return actual_return - expected_return


# Function to perform regression analysis and get detailed statistics
def regression_analysis(stock_returns, market_returns):
    """
    Perform linear regression and return detailed statistics
    """
    valid_data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()

    if len(valid_data) < 2:
        return {
            'beta': np.nan,
            'alpha': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_error': np.nan
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid_data['market'], valid_data['stock']
    )

    return {
        'beta': slope,
        'alpha': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_error': std_err
    }


# Function to calculate all CAPM metrics for multiple stocks
def calculate_all_capm_metrics(df, risk_free_rate=0.02):
    """
    Calculate CAPM metrics for all stocks in the dataframe
    Returns a dictionary with results for each stock
    """
    results = {}

    # Check if S&P 500 data is available
    if 'sp500' not in df.columns:
        return results

    # Calculate decimal returns (better for statistical calculations)
    returns_df = daily_return_decimal(df)

    # Calculate annualized market return
    market_returns = returns_df['sp500']
    market_annual_return = (1 + market_returns.mean()) ** 252 - 1  # Annualize

    for column in df.columns[1:]:
        if column != 'sp500':  # Skip the market column itself
            stock_returns = returns_df[column]

            # Calculate beta
            beta = calculate_beta(stock_returns, market_returns)

            # Calculate annualized stock return
            stock_annual_return = (1 + stock_returns.mean()) ** 252 - 1

            # Calculate expected return using CAPM
            expected_return = calculate_expected_return(risk_free_rate, beta, market_annual_return)

            # Calculate alpha
            alpha = calculate_alpha(stock_annual_return, expected_return)

            # Perform regression analysis for more details
            regression_stats = regression_analysis(stock_returns, market_returns)

            results[column] = {
                'beta': beta,
                'alpha': alpha,
                'actual_annual_return': stock_annual_return,
                'expected_annual_return': expected_return,
                'market_annual_return': market_annual_return,
                'regression_stats': regression_stats
            }

    return results


# Function to create scatter plot of returns with regression lines
def plot_returns_scatter(df):
    """
    Create scatter plot of stock returns vs market returns with regression lines
    """
    returns_df = daily_return_decimal(df)

    if 'sp500' not in returns_df.columns:
        return go.Figure()

    market_returns = returns_df['sp500']

    fig = go.Figure()

    for column in returns_df.columns[1:]:
        if column != 'sp500':
            stock_returns = returns_df[column]

            # Calculate regression statistics
            stats = regression_analysis(stock_returns, market_returns)

            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=market_returns,
                y=stock_returns,
                mode='markers',
                name=f'{column} (β={stats["beta"]:.2f})',
                opacity=0.6
            ))

            # Add regression line
            if not np.isnan(stats['beta']):
                x_range = np.linspace(market_returns.min(), market_returns.max(), 100)
                y_range = stats['alpha'] + stats['beta'] * x_range

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=f'{column} Regression',
                    line=dict(dash='dash'),
                    showlegend=False
                ))

    fig.update_layout(
        title='Stock Returns vs Market Returns',
        xaxis_title='S&P 500 Daily Returns',
        yaxis_title='Stock Daily Returns',
        height=500,
        width=600
    )

    return fig


# Function to display CAPM results in a nice format
def display_capm_results(results):
    """
    Convert CAPM results to a pandas DataFrame for easy display
    """
    result_data = []

    for stock, metrics in results.items():
        result_data.append({
            'Stock': stock,
            'Beta (β)': f"{metrics['beta']:.4f}",
            'Alpha (α)': f"{metrics['alpha']:.4f}",
            'Actual Annual Return': f"{metrics['actual_annual_return']:.4f}",
            'Expected Return (CAPM)': f"{metrics['expected_annual_return']:.4f}",
            'Market Annual Return': f"{metrics['market_annual_return']:.4f}",
            'R-squared': f"{metrics['regression_stats']['r_squared']:.4f}" if not np.isnan(
                metrics['regression_stats']['r_squared']) else 'N/A'
        })

    return pd.DataFrame(result_data)