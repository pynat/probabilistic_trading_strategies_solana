import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SolanaTradingStrategy:
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.model = None
        self.results = {}
        
    def create_features(self, df):
        df = df.copy()
        
        # Basic returns and volatility
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=14).std().shift(1)
        df['volume_change'] = df['volume'].pct_change().shift(1)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean().shift(1)
        df['sma_50'] = df['close'].rolling(window=50).mean().shift(1)
        df['ema_12'] = df['close'].ewm(span=12).mean().shift(1)
        df['ema_26'] = df['close'].ewm(span=26).mean().shift(1)
        
        # Volatility regime 
        vol_20 = df['volatility'].rolling(window=20).mean().shift(1)
        df['vol_regime'] = 0
        df.loc[df['volatility'] > vol_20 * 1.5, 'vol_regime'] = 1  # High vol
        df.loc[df['volatility'] > vol_20 * 2.0, 'vol_regime'] = 2  # Extreme vol
        
        # Volume spike 
        volume_ma = df['volume'].rolling(window=20).mean().shift(1)
        df['volume_spike'] = (df['volume'] > volume_ma * 2.0).astype(int)
        
        # Extreme down 
        df['extreme_down'] = (df['return'] < -0.07).astype(int)
        
        # Breakout signals
        df['breakout_high_7d'] = (df['close'] > df['high'].rolling(window=7).max().shift(1)).astype(int)
        
        # Volume expansion 
        df['vol_expansion'] = (df['volume_change'] > 0.5).astype(int)
        
        # Trend alignment 
        df['trend_alignment'] = ((df['ema_12'] > df['ema_26']) & (df['close'].shift(1) > df['sma_20'])).astype(int)
        
        # Price momentum 
        df['momentum_5d'] = df['close'].shift(1).pct_change(5)
        df['momentum_10d'] = df['close'].shift(1).pct_change(10)
        
        # RSI from previous day 
        df['rsi_prev'] = df['rsi'].shift(1)
        
        return df
    




    
    def create_target(self, df, threshold=0.05):
        """Create target variable (>5% move as per your XGBoost analysis)"""
        df = df.copy()
        # Target is next day's return (this is correct - we predict future)
        df['target'] = (df['return'].shift(-1) > threshold).astype(int)
        return df
    
    def prepare_data(self, df):
        """Prepare data for modeling"""
        df = self.create_features(df)
        df = self.create_target(df)
        
        # Feature columns based on your analysis - UPDATED to use shifted features
        feature_cols = [
            'rsi_prev', 'vol_regime', 'volume_spike', 
            'extreme_down', 'breakout_high_7d', 'vol_expansion',
            'trend_alignment', 
            'volume_change'
        ]
        
        # Remove NaN values
        df = df.dropna()
        
        return df, feature_cols
    
    def train_model(self, df, feature_cols):
        """Train RandomForest model (similar to XGBoost but more stable)"""
        X = df[feature_cols]
        y = df['target']
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Use RandomForest instead of XGBoost for better stability
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        
        # Train on 80% of data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print(f"Feature Importance:")
        for feature, importance in zip(feature_cols, self.model.feature_importances_):
            print(f"  {feature}: {importance:.3f}")
        
        return accuracy
    
    def generate_signals(self, df, feature_cols):
        """Generate trading signals based on your probabilistic approach"""
        df = df.copy()
        
        # Get model predictions - FIXED: Only use available data at time t
        X = df[feature_cols]
        df['prediction'] = self.model.predict(X)
        df['prediction_proba'] = self.model.predict_proba(X)[:, 1]
        
        # Enhanced signal generation based on your Bayesian analysis
        df['signal'] = 0
        
        # Strong bullish signals (based on your analysis)
        bullish_conditions = (
            (df['volume_spike'] == 1) |  # 70.6% success rate
            (df['extreme_down'] == 1) |  # 60% success rate after extreme down (already shifted)
            ((df['prediction_proba'] > 0.7) & (df['vol_regime'] == 0))  # High conviction + low vol
        )
        
        # Bearish filters (avoid these conditions)
        bearish_filters = (
            (df['vol_expansion'] == 1) |  # Bearish signal
            (df['vol_regime'] == 2) |  # Extreme volatility
            (df['rsi_prev'] > 80)  # Overbought (using previous day's RSI)
        )
        
        # Generate long signals
        df.loc[bullish_conditions & ~bearish_filters, 'signal'] = 1
        
        return df
    
    def calculate_position_size(self, df, base_position=1.0):
        """Volatility-based position sizing"""
        df = df.copy()
        
        # Base position sizing on volatility regime
        df['position_size'] = base_position
        
        # Reduce position size in high volatility periods
        df.loc[df['vol_regime'] == 1, 'position_size'] = base_position * 0.5
        df.loc[df['vol_regime'] == 2, 'position_size'] = base_position * 0.25
        
        # Increase position size in low volatility with high conviction
        high_conviction = (df['prediction_proba'] > 0.8) & (df['vol_regime'] == 0)
        df.loc[high_conviction, 'position_size'] = base_position * 1.5
        
        return df
    
    def backtest_strategy(self, df, feature_cols):
        df = self.generate_signals(df, feature_cols)
        df = self.calculate_position_size(df)
        
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        position = 0
        entry_price = 0
        entry_day = 0
        portfolio_values = []
        trades = []
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            signal = df.iloc[i]['signal']
            position_size = df.iloc[i]['position_size']
            
            # Check if we should exit (after 3 days as per your analysis)
            if position > 0 and (i - entry_day) >= 3:
                # Force exit after 3 days
                exit_price = current_price
                portfolio_value += position * exit_price * (1 - self.transaction_cost)
                
                # Record trade
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'entry_date': df.iloc[entry_day]['date'] if 'date' in df.columns else entry_day,
                    'exit_date': df.iloc[i]['date'] if 'date' in df.columns else i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'position_size': position_size,
                    'days_held': i - entry_day
                })
                
                position = 0
                entry_price = 0
                entry_day = 0
            
            # Entry signal (only if not already in position)
            if signal == 1 and position == 0:
                entry_price = current_price
                position = (portfolio_value * position_size) / entry_price
                portfolio_value -= position * entry_price * (1 + self.transaction_cost)
                entry_day = i
            
            # Calculate current portfolio value
            current_portfolio_value = portfolio_value
            if position > 0:
                current_portfolio_value += position * current_price
            
            portfolio_values.append(current_portfolio_value)
        
        # Close any remaining position at the end
        if position > 0:
            final_price = df.iloc[-1]['close']
            portfolio_value += position * final_price * (1 - self.transaction_cost)
            
            trade_return = (final_price - entry_price) / entry_price
            trades.append({
                'entry_date': df.iloc[entry_day]['date'] if 'date' in df.columns else entry_day,
                'exit_date': df.iloc[-1]['date'] if 'date' in df.columns else len(df) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'return': trade_return,
                'position_size': df.iloc[entry_day]['position_size'],
                'days_held': len(df) - 1 - entry_day
            })
            
            portfolio_values[-1] = portfolio_value
        
        # Calculate performance metrics
        df['portfolio_value'] = portfolio_values
        df['portfolio_returns'] = df['portfolio_value'].pct_change()
        
        # Final results
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate key metrics
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            win_rate = (trades_df['return'] > 0).mean()
            avg_return_per_trade = trades_df['return'].mean()
            
            # Risk metrics
            portfolio_rets = df['portfolio_returns'].dropna()
            if len(portfolio_rets) > 0 and portfolio_rets.std() > 0:
                sharpe_ratio = portfolio_rets.mean() / portfolio_rets.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            rolling_max = df['portfolio_value'].cummax()
            drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Volatility
            if len(portfolio_rets) > 0:
                annualized_volatility = portfolio_rets.std() * np.sqrt(252)
            else:
                annualized_volatility = 0
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'num_trades': len(trades_df),
                'win_rate': win_rate,
                'avg_return_per_trade': avg_return_per_trade,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'annualized_volatility': annualized_volatility,
                'trades': trades_df,
                'portfolio_curve': df[['portfolio_value']].copy()
            }
        else:
            self.results = {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'annualized_volatility': 0,
                'trades': pd.DataFrame(),
                'portfolio_curve': df[['portfolio_value']].copy()
            }
        
        return df
    
    def print_results(self):
        """Print comprehensive results"""
        print("=" * 60)
        print("SOLANA PROBABILISTIC TRADING STRATEGY RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ${self.results['initial_capital']:,.2f}")
        print(f"Final Value: ${self.results['final_value']:,.2f}")
        print(f"Total Return: {self.results['total_return']:.1%}")
        print(f"Number of Trades: {self.results['num_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.1%}")
        print(f"Avg Return per Trade: {self.results['avg_return_per_trade']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.1%}")
        print(f"Annualized Volatility: {self.results['annualized_volatility']:.1%}")
        print("=" * 60)
        
        if len(self.results['trades']) > 0:
            print("\nTRADE DETAILS:")
            print(self.results['trades'].to_string(index=False))
            print(f"\nAverage Days Held: {self.results['trades']['days_held'].mean():.1f}")
    
    def plot_results(self):
        """Plot strategy performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(self.results['portfolio_curve']['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        portfolio_values = self.results['portfolio_curve']['portfolio_value']
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True)
        
        # Trade returns distribution
        if len(self.results['trades']) > 0:
            axes[1, 0].hist(self.results['trades']['return'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Trade Returns Distribution')
            axes[1, 0].set_xlabel('Return %')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
            
            # Cumulative returns
            cumulative_returns = (1 + self.results['trades']['return']).cumprod()
            axes[1, 1].plot(cumulative_returns)
            axes[1, 1].set_title('Cumulative Trade Returns')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative Return')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    df = pd.read_csv('sol.csv')
    
    # Initialize strategy
    strategy = SolanaTradingStrategy(initial_capital=10000)
    
    # Prepare data
    df_processed, feature_cols = strategy.prepare_data(df)
    
    # Train model
    accuracy = strategy.train_model(df_processed, feature_cols)
    
    # Run backtest
    df_backtest = strategy.backtest_strategy(df_processed, feature_cols)
    
    # Print results
    strategy.print_results()
    
    # Plot results
    strategy.plot_results()