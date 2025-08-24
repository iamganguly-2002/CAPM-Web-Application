# import libraries

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import CAPM_functions as cf
import plotly.express as px# Changed to import with alias
import numpy as np
import portfolio_optimizer
st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upward_trend",
                   layout="wide")

st.title("Capital Asset Pricing Model")

# getting input from user
col1, col2 = st.columns([1, 1])
with col1:
    stocks_list = st.multiselect("Choose Stocks (2+ for optimization)",
                                 ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'),
                                 ['TSLA', 'AAPL', 'AMZN', 'GOOGL'])
with col2:
    year = st.number_input("Number of years", 1, 10, 5)
    risk_free_rate = st.number_input("Risk-free rate (%)", 0.0, 10.0, 2.5) / 100

# Downloading data for S&P 500 (using ^GSPC ticker)
end = datetime.date.today()
start = datetime.date(datetime.date.today().year - year,
                      datetime.date.today().month,
                      datetime.date.today().day)

# downloading data for SP500
try:
    SP500 = yf.download('^GSPC', start=start, end=end)
    SP500 = SP500[['Close']]  # Keep only close price
    SP500.columns = ['sp500']
except Exception as e:
    st.error(f"Error downloading S&P 500 data: {e}")
    SP500 = pd.DataFrame()

# Downloading data for selected stocks
stocks_df = pd.DataFrame()

for stock in stocks_list:
    try:
        data = yf.download(stock, start=start, end=end)
        stocks_df[stock] = data['Close']
    except Exception as e:
        st.error(f"Error downloading {stock} data: {e}")

# Merge data if we have both stocks and SP500 data
if not stocks_df.empty and not SP500.empty:
    stocks_df = stocks_df.reset_index()
    SP500 = SP500.reset_index()
    combined_df = pd.merge(stocks_df, SP500, on='Date', how='inner')
else:
    combined_df = pd.DataFrame()
    st.warning("Could not download data for all selected stocks. Please check your selections.")

if not combined_df.empty:
    # Display data
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### DataFrame head")
        st.dataframe(combined_df.head(), use_container_width=True)

    with col2:
        st.markdown("### DataFrame tail")
        st.dataframe(combined_df.tail(), use_container_width=True)

    # Interactive plots
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Price of All stocks")
        st.plotly_chart(cf.interactive_plot(combined_df))
    with col2:
        st.markdown("### Price of All stocks (After Normalization)")
        st.plotly_chart(cf.interactive_plot(cf.normalize(combined_df)))

    # Calculate returns
    stocks_daily_return = cf.daily_return(combined_df)
    stocks_daily_return_decimal = cf.daily_return_decimal(combined_df)

    # CAPM Analysis Section
    st.markdown("---")
    st.header("CAPM Analysis")

    # Calculate all CAPM metrics
    capm_results = cf.calculate_all_capm_metrics(combined_df, risk_free_rate)

    if capm_results:
        # Display CAPM results in a table
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### CAPM Metrics")
            results_df = cf.display_capm_results(capm_results)
            st.dataframe(results_df, use_container_width=True)

        with col2:
            st.markdown("### Returns vs Market Scatter Plot")
            scatter_fig = cf.plot_returns_scatter(combined_df)
            st.plotly_chart(scatter_fig)

        # Display detailed regression statistics
        st.markdown("### Detailed Regression Statistics")
        for stock, metrics in capm_results.items():
            with st.expander(f"Regression Details for {stock}"):
                st.write(f"**Beta (β):** {metrics['beta']:.4f}")
                st.write(f"**Alpha (α):** {metrics['alpha']:.4f}")
                st.write(f"**R-squared:** {metrics['regression_stats']['r_squared']:.4f}")
                st.write(f"**P-value:** {metrics['regression_stats']['p_value']:.6f}")
                st.write(f"**Standard Error:** {metrics['regression_stats']['std_error']:.6f}")
                st.write(f"**Actual Annual Return:** {metrics['actual_annual_return'] * 100:.2f}%")
                st.write(f"**Expected Annual Return (CAPM):** {metrics['expected_annual_return'] * 100:.2f}%")
                st.write(f"**Market Annual Return:** {metrics['market_annual_return'] * 100:.2f}%")

        # Interpretation
        st.markdown("### Interpretation")
        st.info("""
        - **Beta (β)**: Measures the stock's volatility relative to the market. 
          - β > 1: More volatile than market
          - β = 1: Same volatility as market  
          - β < 1: Less volatile than market
          - β < 0: Moves opposite to market

        - **Alpha (α)**: Excess return over expected CAPM return.
          - α > 0: Outperformed expectations
          - α < 0: Underperformed expectations

        - **R-squared**: How much of stock's movement is explained by market movement (0-1 scale)
        """)

    else:
        st.warning("No CAPM results available. Please check if S&P 500 data was downloaded correctly.")

    # Display daily returns data
    st.markdown("### Daily Returns Data")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Percentage Returns")
        st.dataframe(stocks_daily_return.head(), use_container_width=True)
    with col2:
        st.markdown("#### Decimal Returns (for calculations)")
        st.dataframe(stocks_daily_return_decimal.head(), use_container_width=True)

    # PORTFOLIO OPTIMIZATION SECTION
    st.markdown("---")
    st.header("📊 Portfolio Optimization")

    if len(stocks_list) >= 2:
        try:
            # Import portfolio optimizer
            from portfolio_optimizer import PortfolioOptimizer

            # Create optimizer instance
            optimizer = PortfolioOptimizer(stocks_daily_return_decimal)

            st.markdown("### Efficient Frontier Analysis")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Efficient Frontier Plot
                st.markdown("#### Efficient Frontier")
                frontier_fig = optimizer.plot_efficient_frontier(risk_free_rate)
                st.plotly_chart(frontier_fig, use_container_width=True)

            with col2:
                st.markdown("#### Optimal Portfolios")

                try:
                    # Calculate optimal portfolios
                    sharpe_portfolio = optimizer.optimize_portfolio(risk_free_rate, 'sharpe')
                    min_var_portfolio = optimizer.optimize_portfolio(risk_free_rate, 'min_variance')

                    # Display Sharpe-optimal portfolio
                    st.markdown("**🎯 Max Sharpe Ratio Portfolio**")
                    sharpe_data = []
                    for i, ticker in enumerate(sharpe_portfolio['tickers']):
                        sharpe_data.append({
                            'Stock': ticker,
                            'Weight': f"{sharpe_portfolio['weights'][i] * 100:.1f}%"
                        })
                    sharpe_df = pd.DataFrame(sharpe_data)
                    st.dataframe(sharpe_df, use_container_width=True)
                    st.metric("Expected Return", f"{sharpe_portfolio['return'] * 100:.2f}%")
                    st.metric("Volatility (Risk)", f"{sharpe_portfolio['volatility'] * 100:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe_portfolio['sharpe_ratio']:.3f}")

                    st.markdown("---")

                    # Display Min-Variance portfolio
                    st.markdown("**🛡️ Minimum Variance Portfolio**")
                    min_var_data = []
                    for i, ticker in enumerate(min_var_portfolio['tickers']):
                        min_var_data.append({
                            'Stock': ticker,
                            'Weight': f"{min_var_portfolio['weights'][i] * 100:.1f}%"
                        })
                    min_var_df = pd.DataFrame(min_var_data)
                    st.dataframe(min_var_df, use_container_width=True)
                    st.metric("Expected Return", f"{min_var_portfolio['return'] * 100:.2f}%")
                    st.metric("Volatility (Risk)", f"{min_var_portfolio['volatility'] * 100:.2f}%")
                    st.metric("Sharpe Ratio", f"{min_var_portfolio['sharpe_ratio']:.3f}")

                except Exception as e:
                    st.error(f"Portfolio optimization calculation failed: {str(e)}")

            # Correlation Matrix
            st.markdown("### 📈 Correlation Matrix")
            correlation_matrix = stocks_daily_return_decimal[stocks_list].corr()

            fig = px.imshow(correlation_matrix,
                            labels=dict(x="Stocks", y="Stocks", color="Correlation"),
                            x=stocks_list,
                            y=stocks_list,
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            title="Stock Returns Correlation Matrix")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Portfolio Optimization Explanation
            with st.expander("ℹ️ About Portfolio Optimization"):
                st.markdown("""
                **Modern Portfolio Theory (MPT)** by Harry Markowitz suggests that investors can construct portfolios 
                to maximize expected return for a given level of risk.

                **Key Concepts:**
                - **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a defined level of risk
                - **Max Sharpe Portfolio**: Best risk-adjusted returns (return per unit of risk)
                - **Minimum Variance Portfolio**: Lowest possible risk configuration
                - **Diversification**: Combining assets with low correlation to reduce overall portfolio risk

                **How to use this:**
                1. Look at the Efficient Frontier to understand risk-return tradeoffs
                2. Choose between Max Sharpe (better returns) or Min Variance (lower risk)
                3. Check correlation matrix to understand diversification benefits
                """)

        except ImportError:
            st.warning("Portfolio optimization features require the portfolio_optimizer.py file in the same directory.")
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")
    else:
        st.info("🔍 Select at least 2 stocks to enable portfolio optimization features.")

else:
    st.warning("No data available for analysis. Please select stocks and ensure data download was successful.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data from Yahoo Finance | Modern Portfolio Theory Implementation")