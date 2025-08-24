# portfolio_optimizer.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, returns_df):
        self.returns_df = returns_df
        self.tickers = [col for col in returns_df.columns if col != 'Date' and col != 'sp500']
        self.mean_returns = self._calculate_mean_returns()
        self.cov_matrix = self._calculate_cov_matrix()

    def _calculate_mean_returns(self):
        """Calculate annualized mean returns"""
        daily_returns = self.returns_df[self.tickers]
        annual_returns = daily_returns.mean() * 252
        return annual_returns

    def _calculate_cov_matrix(self):
        """Calculate annualized covariance matrix"""
        daily_returns = self.returns_df[self.tickers]
        daily_cov = daily_returns.cov()
        annual_cov = daily_cov * 252
        return annual_cov

    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility"""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_volatility

    def negative_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """Negative Sharpe ratio for minimization"""
        port_return, port_vol = self.portfolio_performance(weights)
        if port_vol == 0:
            return -1000  # Large negative value for zero volatility
        sharpe_ratio = (port_return - risk_free_rate) / port_vol
        return -sharpe_ratio

    def portfolio_variance(self, weights):
        """Portfolio variance for minimization"""
        port_return, port_vol = self.portfolio_performance(weights)
        return port_vol ** 2

    def optimize_portfolio(self, risk_free_rate=0.02, optimization_type='sharpe'):
        """Optimize portfolio based on specified objective"""
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]

        if optimization_type == 'sharpe':
            objective = self.negative_sharpe_ratio
            args = (risk_free_rate,)
        elif optimization_type == 'min_variance':
            objective = self.portfolio_variance
            args = ()
        else:
            raise ValueError("Optimization type must be 'sharpe' or 'min_variance'")

        result = minimize(objective, initial_weights, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError("Portfolio optimization failed")

        optimal_weights = result.x
        optimal_return, optimal_volatility = self.portfolio_performance(optimal_weights)
        sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility if optimal_volatility > 0 else 0

        return {
            'weights': optimal_weights,
            'return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe_ratio': sharpe_ratio,
            'tickers': self.tickers
        }

    def efficient_frontier(self, risk_free_rate=0.02, num_points=100):
        """Generate efficient frontier"""
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), num_points)
        efficient_portfolios = []

        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - target_return}
            ]
            bounds = tuple((0, 1) for _ in range(len(self.tickers)))
            initial_weights = len(self.tickers) * [1. / len(self.tickers)]

            result = minimize(self.portfolio_variance, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = result.x
                return_val, volatility = self.portfolio_performance(weights)
                sharpe_ratio = (return_val - risk_free_rate) / volatility if volatility > 0 else 0
                efficient_portfolios.append({
                    'return': return_val,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': weights
                })

        return pd.DataFrame(efficient_portfolios)

    def plot_efficient_frontier(self, risk_free_rate=0.02, num_points=100):
        """Plot efficient frontier with optimal portfolios"""
        # Get efficient frontier
        frontier_df = self.efficient_frontier(risk_free_rate, num_points)

        # Get optimal portfolios
        try:
            sharpe_portfolio = self.optimize_portfolio(risk_free_rate, 'sharpe')
            min_var_portfolio = self.optimize_portfolio(risk_free_rate, 'min_variance')
        except:
            # Fallback if optimization fails
            sharpe_portfolio = min_var_portfolio = None

        # Create plot
        fig = go.Figure()

        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'],
            y=frontier_df['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))

        # Individual stocks
        for i, ticker in enumerate(self.tickers):
            stock_return = self.mean_returns.iloc[i]
            stock_volatility = np.sqrt(self.cov_matrix.iloc[i, i])
            fig.add_trace(go.Scatter(
                x=[stock_volatility],
                y=[stock_return],
                mode='markers+text',
                name=ticker,
                text=[ticker],
                textposition="top center",
                marker=dict(size=12, symbol='circle')
            ))

        # Optimal portfolios if available
        if sharpe_portfolio:
            fig.add_trace(go.Scatter(
                x=[sharpe_portfolio['volatility']],
                y=[sharpe_portfolio['return']],
                mode='markers+text',
                name='Max Sharpe Ratio',
                text=['Max Sharpe'],
                textposition="top center",
                marker=dict(size=15, symbol='star', color='gold')
            ))

        if min_var_portfolio:
            fig.add_trace(go.Scatter(
                x=[min_var_portfolio['volatility']],
                y=[min_var_portfolio['return']],
                mode='markers+text',
                name='Min Variance',
                text=['Min Variance'],
                textposition="top center",
                marker=dict(size=15, symbol='diamond', color='green')
            ))

        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Annualized Volatility',
            yaxis_title='Annualized Return',
            height=600,
            width=800,
            showlegend=True
        )

        return fig


def calculate_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate portfolio statistics given weights"""
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }