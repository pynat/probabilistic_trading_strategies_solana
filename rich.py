import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class VolatilityRegimeBacktester:
    
    def __init__(self, base_position_size=0.02, risk_free_rate=0.02):
        self.base_position_size = base_position_size
        self.risk_free_rate = risk_free_rate
        self.mean_vol = 0.0421  # 4.21% aus deiner Analyse
        self.high_vol_threshold = 0.0619  # 6.19% aus deiner Analyse
        
        # Position limits pro Regime (basierend auf deiner Analyse)
        self.position_limits = {
            0: 0.02,  # LOW_VOL: Full size (81.8% Erfolgsrate)
            1: 0.015, # MID_VOL: Reduced size (57.1% Erfolgsrate)
            2: 0.01   # HIGH_VOL: Small size (68.8% Erfolgsrate)
        }
        
        self.trades = []
        
    def load_data(self, csv_path):
        print(f"Loading data from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Prüfe auf benötigte Columns
            required_columns = [
                'close', 'volatility', 'return', 
                'breakout_high_7d', 'vol_expansion', 'volume_spike'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            # Prüfe auf Index (falls Datum)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            print(f"✅ Data loaded successfully: {len(df)} rows")
            print(f"Date range: {df.index[0]} to {df.index[-1]}" if hasattr(df.index, 'date') else f"Row range: 0 to {len(df)-1}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def calculate_volatility_regime(self, df):
        """
        Berechne Volatilitäts-Regime (nutzt deine bereits berechnete Volatilität)
        """
        df = df.copy()
        
        # Regime Classification basierend auf deiner Analyse
        df['vol_regime'] = np.where(
            df['volatility'] < self.mean_vol * 0.8, 0,  # LOW_VOL: <3.37%
            np.where(
                df['volatility'] < self.high_vol_threshold, 1,  # MID_VOL: 3.37%-6.19%
                2  # HIGH_VOL: >6.19%
            )
        )
        
        return df
    
    def calculate_position_size(self, current_vol, regime):
        """
        Berechne volatilitäts-adjustierte Positionsgröße
        """
        if pd.isna(current_vol) or current_vol <= 0:
            return 0
        
        # Volatilitäts-Anpassung: base_size * (mean_vol / current_vol)
        vol_adjusted_size = self.base_position_size * (self.mean_vol / current_vol)
        
        # Regime-spezifische Limits
        max_size = self.position_limits.get(regime, 0.01)
        
        # Clamp zwischen 0.5% und regime-spezifischem Maximum
        return np.clip(vol_adjusted_size, 0.005, max_size)
    
    def get_signal(self, row):
        """
        Bestimme Trading-Signal basierend auf Regime und Indikatoren
        """
        regime = row['vol_regime']
        
        if pd.isna(regime):
            return None, None
        
        regime = int(regime)
        
        # LOW_VOL Regime: Breakout High 7D (beste Performance: 81.8%)
        if regime == 0:
            if row.get('breakout_high_7d', 0) == 1:
                return 'LONG', 'LOW_VOL_BREAKOUT'
        
        # MID_VOL Regime: Volume Expansion (moderate Performance: 57.1%)
        elif regime == 1:
            if row.get('vol_expansion', 0) == 1:
                return 'LONG', 'MID_VOL_EXPANSION'
        
        # HIGH_VOL Regime: Volume Spike (gute Performance: 68.8%)
        elif regime == 2:
            if row.get('volume_spike', 0) == 1:
                return 'LONG', 'HIGH_VOL_SPIKE'
        
        return None, None
    
    def run_backtest(self, df, initial_capital=10000):
        """
        Führe vollständigen Backtest durch
        """
        print(f"\n=== VOLATILITY REGIME BACKTEST ===")
        print(f"Initial Capital: ${initial_capital:,.0f}")
        print(f"Base Position Size: {self.base_position_size:.1%}")
        print("-" * 50)
        
        # Regime berechnen
        df = self.calculate_volatility_regime(df)
        
        # Regime-Verteilung anzeigen
        regime_counts = df['vol_regime'].value_counts().sort_index()
        print(f"REGIME DISTRIBUTION:")
        for regime, count in regime_counts.items():
            regime_name = ['LOW_VOL', 'MID_VOL', 'HIGH_VOL'][int(regime)]
            pct = count / len(df) * 100
            avg_vol = df[df['vol_regime'] == regime]['volatility'].mean()
            print(f"  Regime {int(regime)} ({regime_name}): {count} periods ({pct:.1f}%) - Avg Vol: {avg_vol:.1%}")
        
        # Backtest-Spalten initialisieren
        df['position'] = 0.0
        df['signal_name'] = ''
        df['portfolio_value'] = float(initial_capital)
        df['trade_returns'] = 0.0
        
        # Reset trades
        self.trades = []
        
        # Trading-Schleife
        signals_found = 0
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            
            # Signal prüfen
            signal_type, strategy_name = self.get_signal(current_row)
            
            if signal_type:
                signals_found += 1
                current_vol = current_row['volatility']
                regime = int(current_row['vol_regime'])
                position_size = self.calculate_position_size(current_vol, regime)
                
                # Position setzen
                df.at[df.index[i], 'position'] = position_size
                df.at[df.index[i], 'signal_name'] = strategy_name
                
                # Trade-Info speichern
                trade_info = {
                    'date': df.index[i],
                    'strategy': strategy_name,
                    'regime': regime,
                    'position_size': position_size,
                    'volatility': current_vol,
                    'entry_price': current_row['close']
                }
                self.trades.append(trade_info)
        
        print(f"\nSignals found: {signals_found}")
        
        # Returns berechnen
        for i in range(1, len(df)):
            prev_position = df.iloc[i-1]['position']
            if prev_position != 0:
                market_return = df.iloc[i]['return']
                trade_return = market_return * prev_position
                df.at[df.index[i], 'trade_returns'] = trade_return
                df.at[df.index[i], 'portfolio_value'] = df.iloc[i-1]['portfolio_value'] * (1 + trade_return)
            else:
                df.at[df.index[i], 'portfolio_value'] = df.iloc[i-1]['portfolio_value']
        
        # Performance-Metriken berechnen
        self.calculate_performance_metrics(df, initial_capital)
        
        return df
    
    def calculate_performance_metrics(self, df, initial_capital):
        """
        Berechne und zeige Performance-Metriken
        """
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_capital - 1) * 100
        
        print(f"\n=== PERFORMANCE RESULTS ===")
        print(f"Final Portfolio Value: ${final_value:,.0f}")
        print(f"Total Return: {total_return:.1f}%")
        print(f"Number of Trades: {len(self.trades)}")
        
        # Trade-spezifische Metriken
        if len(self.trades) > 0:
            trade_returns = df[df['trade_returns'] != 0]['trade_returns']
            
            if len(trade_returns) > 0:
                winning_trades = (trade_returns > 0).sum()
                losing_trades = (trade_returns < 0).sum()
                win_rate = (winning_trades / len(trade_returns)) * 100
                avg_return = trade_returns.mean() * 100
                
                print(f"Winning Trades: {winning_trades}")
                print(f"Losing Trades: {losing_trades}")
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Average Return per Trade: {avg_return:.2f}%")
                
                # Risk-Metriken
                returns_std = trade_returns.std()
                if returns_std > 0:
                    annualized_vol = returns_std * np.sqrt(252) * 100  # 252 trading days
                    print(f"Strategy Volatility: {annualized_vol:.1f}%")
                
                # Drawdown
                running_max = df['portfolio_value'].cummax()
                drawdown = (df['portfolio_value'] / running_max - 1) * 100
                max_drawdown = drawdown.min()
                print(f"Maximum Drawdown: {max_drawdown:.1f}%")
                
                # Sharpe Ratio
                if returns_std > 0:
                    excess_return = total_return - (self.risk_free_rate * 100)
                    sharpe = excess_return / (returns_std * np.sqrt(252) * 100)
                    print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Regime-Breakdown
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            print(f"\n=== TRADES BY REGIME ===")
            for regime in sorted(trades_df['regime'].unique()):
                regime_trades = trades_df[trades_df['regime'] == regime]
                regime_name = ['LOW_VOL', 'MID_VOL', 'HIGH_VOL'][int(regime)]
                avg_size = regime_trades['position_size'].mean()
                print(f"Regime {regime} ({regime_name}): {len(regime_trades)} trades, "
                      f"Avg Size: {avg_size:.1%}")
            
            # Strategy-Breakdown
            print(f"\n=== TRADES BY STRATEGY ===")
            strategy_counts = trades_df['strategy'].value_counts()
            for strategy, count in strategy_counts.items():
                avg_size = trades_df[trades_df['strategy'] == strategy]['position_size'].mean()
                print(f"{strategy}: {count} trades, Avg Size: {avg_size:.1%}")
    
    def get_trade_details(self):
        """
        Gebe detaillierte Trade-Informationen zurück
        """
        if self.trades:
            return pd.DataFrame(self.trades)
        return pd.DataFrame()


# === USAGE FUNCTIONS ===
def run_backtest_from_csv(csv_path, initial_capital=10000, base_position_size=0.02):
    """
    Führe Backtest direkt von CSV-Datei aus
    """
    # Backtester initialisieren
    backtester = VolatilityRegimeBacktester(base_position_size=base_position_size)
    
    # Daten laden
    df = backtester.load_data(csv_path)
    if df is None:
        return None, None
    
    # Backtest ausführen
    result_df = backtester.run_backtest(df, initial_capital)
    
    # Trade-Details
    trade_details = backtester.get_trade_details()
    
    return result_df, trade_details


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Beispiel-Verwendung
    print("=== BACKTEST EXAMPLE ===")
    print("Verwendung:")
    print("1. result_df, trades = run_backtest_from_csv('your_data.csv')")
    print("2. Oder manuell:")
    print("   backtester = VolatilityRegimeBacktester()")
    print("   df = backtester.load_data('your_data.csv')")
    print("   result_df = backtester.run_backtest(df)")
    
    # Beispiel für erwartete CSV-Struktur
    print(f"\n=== EXPECTED CSV STRUCTURE ===")
    print("Required columns:")
    print("- date (optional, for index)")
    print("- close (Schlusskurse)")
    print("- volatility (bereits berechnet)")  
    print("- return (bereits berechnet)")
    print("- breakout_high_7d (dein Indikator)")
    print("- vol_expansion (dein Indikator)")
    print("- volume_spike (dein Indikator)")
    
    result_df, trades = run_backtest_from_csv('sol_ready.csv', initial_capital=10000)
    print(trades.head()) if trades is not None else None