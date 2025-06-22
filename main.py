# === Import Required Libraries ===
import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt

# === Helper Visualization Function ===

# Plots historical and predicted volatility for a given stock
def plot_volatility(df, future_vol, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['volatility'], label='Historical 30-day Volatility', color='blue')
    plt.axhline(future_vol, color='red', linestyle='--', label='Predicted Future Volatility')
    plt.title(f"Volatility for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Volatility Prediction Pipeline ===

# Predicts future volatility for each stock in the portfolio using an LSTM model
def predict_volatility(portfolio):
    predictions = {}
    for symbol in portfolio:
        df = get_historical_data(symbol)
        if df is None or df.empty:
            print(f"⚠ Skipping {symbol}, no data.")
            continue

        features, future_vol, _ = preprocess_data(df)
        if len(features) < 60 or len(future_vol) < 60:
            print(f"⚠ Not enough data for {symbol}, skipping.")
            continue

        X, y = create_sequences(features, future_vol)
        if len(X) < 1:
            print(f"⚠ Not enough sequences for {symbol}, skipping.")
            continue

        model = build_lstm_model(X.shape[1:])
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # Predict next period's volatility
        last_seq = np.expand_dims(X[-1], axis=0)
        pred_vol = model.predict(last_seq, verbose=0)[0][0]
        predictions[symbol] = pred_vol

        # Recompute volatility for plotting
        df['returns'] = df['Mid'].pct_change()
        df['volatility'] = df['returns'].rolling(30).std()
        plot_volatility(df, pred_vol, symbol)

    return predictions

# === Utility and Input Functions ===

# Get stock sector using Yahoo Finance API
def get_stock_sector(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return info.get("sector", "Unknown")

# Validates a user's input against allowed options
def validate_input(prompt, valid_options):
    choice = input(prompt).strip()
    while choice not in valid_options:
        choice = input(f"Invalid choice. {prompt}").strip()
    return choice

# Allows user to build a portfolio interactively
def build_portfolio():
    portfolio = []
    sector_distribution = defaultdict(int)

    while True:
        keywords = input("\nEnter company name or ticker (or 'done'): ").strip()
        if keywords.lower() == 'done':
            break
        ticker = yf.Ticker(keywords)
        try:
            info = ticker.info
            symbol = info["symbol"]
            name = info["shortName"]
            print(f"Found: {symbol} - {name}")
            confirm = validate_input("Add this stock? (yes/no): ", ["yes", "no"])
            if confirm == "yes":
                sector = get_stock_sector(symbol)
                portfolio.append(symbol)
                sector_distribution[sector] += 1
                print(f"✅ Added {symbol} ({sector})")
        except Exception:
            print("❌ Could not find stock. Please try again.")
    return portfolio, sector_distribution

# Collects user-defined risk level (1 to 5)
def get_risk_level():
    risk_level = input("\nEnter your risk level (1-5): ").strip()
    while not risk_level.isdigit() or not (1 <= int(risk_level) <= 5):
        risk_level = input("Invalid input. Please enter 1-5: ").strip()
    return int(risk_level)

# Displays final portfolio breakdown to user
def display_final_portfolio(portfolio, sector_distribution, risk_level):
    print("\n📈 Final Portfolio:")
    print(f"Risk Level: {risk_level}/5")
    for i, stock in enumerate(portfolio, 1):
        print(f"{i}. {stock} ({get_stock_sector(stock)})")

# === Data Processing and Modeling ===

# Downloads historical stock data and computes mid price
def get_historical_data(symbol, years=4):
    df = yf.download(symbol, period=f"{years}y", interval="1d", progress=False)
    if df.empty:
        print(f"⚠ No data for {symbol}")
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df['Mid'] = (df['High'] + df['Low']) / 2.0
    return df

# Calculates beta relative to the market
def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

# Calculates beta and volatility for a given stock
def calculate_risk_metrics(df, market_df):
    df['returns'] = df['Mid'].pct_change()
    market_df['returns'] = market_df['Mid'].pct_change()
    df.dropna(inplace=True)
    market_df.dropna(inplace=True)
    beta = calculate_beta(df['returns'], market_df['returns'])
    volatility = df['returns'].std()
    return beta, volatility

# Prepares features and future volatility targets for training
def preprocess_data(df, window=30, future_window=30, smoothing_window_size=2500):
    df['returns'] = df['Mid'].pct_change()
    df['volatility'] = df['returns'].rolling(window).std()
    df['volume'] = df['Volume'].pct_change()
    df['future_vol'] = df['volatility'].shift(-future_window).rolling(future_window).mean()
    df.dropna(inplace=True)

    features = df[['Mid', 'returns', 'volatility', 'volume']].values
    scaler = MinMaxScaler()
    
    # Normalize in chunks to avoid memory overload
    for di in range(0, len(features), smoothing_window_size):
        upper = di + smoothing_window_size
        fit_data = features[di:] if upper > len(features) else features[di:upper]
        scaler.fit(fit_data)
        features[di:upper] = scaler.transform(fit_data)

    return features, df['future_vol'].values, scaler

# Converts data into sequential format for LSTM input
def create_sequences(data, targets, seq_length=30):
    X, y = [], []
    for i in range(seq_length, len(data) - 1):
        X.append(data[i - seq_length:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

# Builds and compiles an LSTM neural network model
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# === Portfolio Backtesting and Recommendation ===

# Backtests portfolio performance vs S&P 500 and NASDAQ
def backtest_portfolio(allocations, portfolio):
    tickers = portfolio + ['^GSPC', '^IXIC']
    raw_data = yf.download(tickers, period="4y", interval="1d", progress=False, group_by="ticker", auto_adjust=True)

    # Extract adjusted closing prices
    adj_data = {}
    for sym in tickers:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if (sym, 'Close') in raw_data.columns:
                    adj_data[sym] = raw_data[sym]['Close']
                elif 'Adj Close' in raw_data[sym]:
                    adj_data[sym] = raw_data[sym]['Adj Close']
            else:
                if sym in raw_data.columns:
                    adj_data[sym] = raw_data[sym]
        except Exception as e:
            print(f"⚠ Failed to load data for {sym}: {e}")

    if not adj_data:
        print("⚠ No valid price data found.")
        return

    df = pd.DataFrame(adj_data).dropna()
    if df.empty:
        print("⚠ Final price data is empty.")
        return

    norm = df / df.iloc[0]  # Normalize to 1
    valid_symbols = [s for s in portfolio if s in norm.columns]
    weights = np.array([allocations.get(s, 0)/100 for s in valid_symbols])

    if not valid_symbols:
        print("⚠ No valid portfolio stocks found.")
        return

    portfolio_value = (norm[valid_symbols] * weights).sum(axis=1)

    # Plot performance
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label="Your Portfolio", linewidth=2)
    if '^GSPC' in norm.columns:
        plt.plot(norm['^GSPC'], label="S&P 500", linestyle='--')
    if '^IXIC' in norm.columns:
        plt.plot(norm['^IXIC'], label="NASDAQ", linestyle='--')
    plt.title("Backtest: Portfolio vs S&P 500 & NASDAQ (4 Years)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Portfolio Allocation Logic ===

# Computes a combined risk score from predicted volatility and beta
def compute_risk_scores(predictions, betas, user_risk_level):
    alpha = user_risk_level / 5.0  # weight for volatility
    scores = {}
    for symbol in predictions:
        vol = predictions[symbol]
        beta = betas.get(symbol, 1.0)
        score = alpha * vol + (1 - alpha) * beta
        scores[symbol] = score
    return scores

# Recommends stocks based on risk preference
def recommend_stocks(risk_scores, user_risk_level, top_n=5):
    sorted_stocks = sorted(risk_scores.items(), key=lambda x: x[1])
    if user_risk_level <= 2:
        return sorted_stocks[:top_n]  # Conservative: lowest risk
    elif user_risk_level >= 4:
        return sorted_stocks[-top_n:]  # Aggressive: highest risk
    else:
        # Balanced: middle of the pack
        mid = len(sorted_stocks) // 2
        return sorted_stocks[mid - top_n//2 : mid + top_n//2]

# Calculates portfolio allocations based on adjusted risk scores
def calculate_allocations(risk_scores, user_risk_level):
    alpha = user_risk_level / 5.0
    adjusted_scores = {}
    for symbol, score in risk_scores.items():
        try:
            inverse_score = 1.0 / score if score > 0 else 0.0001
        except ZeroDivisionError:
            inverse_score = 0.0001
        adjusted = (1 - alpha) * inverse_score + alpha * score
        adjusted_scores[symbol] = adjusted

    total = sum(adjusted_scores.values())
    allocations = {symbol: (adj / total) * 100 for symbol, adj in adjusted_scores.items()}
    return allocations

# Runs the full risk-based recommendation pipeline
def run_risk_recommendation(predictions, portfolio, risk_level):
    betas = {}
    market_df = get_historical_data('SPY')
    for symbol in portfolio:
        df = get_historical_data(symbol)
        if df is not None and market_df is not None:
            try:
                beta, _ = calculate_risk_metrics(df.copy(), market_df.copy())
                betas[symbol] = beta
            except:
                continue

    risk_scores = compute_risk_scores(predictions, betas, risk_level)
    recommendations = recommend_stocks(risk_scores, risk_level)

    print("\n🌟 Personalized Investment Suggestions:")
    for symbol, score in recommendations:
        beta_val = betas.get(symbol, 'N/A')
        beta_str = f"{beta_val:.2f}" if isinstance(beta_val, (float, int)) else str(beta_val)
        print(f"{symbol}: Risk Score = {score:.4f}, Beta = {beta_str}, Volatility = {predictions[symbol]:.4f}")

    allocations = calculate_allocations(risk_scores, risk_level)

    print("\n📊 Suggested Portfolio Allocation:")
    for symbol, percent in allocations.items():
        print(f"{symbol}: {percent:.2f}%")

    # Display pie chart of allocation
    labels = list(allocations.keys())
    sizes = list(allocations.values())
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title("Portfolio Allocation by Risk-Adjusted Weights")
    plt.axis('equal')
    plt.show()

    backtest_portfolio(allocations, portfolio)

# === Entry Point ===

# Main program loop
def main():
    portfolio, sector_distribution = build_portfolio()
    risk_level = get_risk_level()
    display_final_portfolio(portfolio, sector_distribution, risk_level)
    predictions = predict_volatility(portfolio)
    print("\n📊 Final Predicted Volatility:")
    for stock, vol in predictions.items():
        print(f"{stock}: {vol:.4f}")
    run_risk_recommendation(predictions, portfolio, risk_level)

# Run script if executed directly
if __name__ == "__main__":
    main()
