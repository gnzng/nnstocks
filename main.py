# Minimal, pandas-free stock price prediction script
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


class StockPricePredictor(nn.Module):
    def __init__(self, num_stocks):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # input: [stock_id, date]

    def forward(self, stock_ids, dates):
        if dates.dim() == 1:
            dates = dates.unsqueeze(1)
        x = torch.cat([stock_ids.float().unsqueeze(1), dates], dim=1)
        return self.fc(x)


def fetch_data(symbols, start, end):
    all_X, all_y = [], []
    stock_to_idx = {s: i for i, s in enumerate(symbols)}
    for symbol in symbols:
        data = yf.download(symbol, start=start, end=end, auto_adjust=False)
        data.reset_index(inplace=True)  # Reset index to make Date a column
        data["Date"] = pd.to_datetime(data["Date"])  # Convert Date column to datetime

        # I already know which stock it is here so lets move away from stupid pandas
        # Convert pandas datetime to numpy float timestamps efficiently
        start_date = data["Date"].iloc[0]
        dates_np = (data["Date"] - start_date).dt.days.to_numpy()
        closes_np = data["Close"].to_numpy().flatten()
        print(f"Downloaded data for {symbol}:")
        print(data.head())
        if "Date" not in data.columns or "Close" not in data.columns:
            print(f"Missing required columns (Date or Close) in data for {symbol}")
            continue
        try:
            all_X.append(dates_np)
            all_y.append(closes_np)
        except Exception as e:
            print(f" Error: {e}")
    if not all_X:
        raise ValueError("No valid data found.")
    return np.array(all_X), np.array(all_y), stock_to_idx


def main():
    symbols = ["AAPL", "MSFT", "GOOG"]
    start = "2010-01-01"
    end = "2020-01-01"
    epochs = 10
    batch_size = 64

    X, y, stock_to_idx = fetch_data(symbols, start, end)

    # Reshape data: flatten each stock's dates and prices into a single array
    num_stocks, num_days = X.shape
    stock_ids = np.repeat(np.arange(num_stocks), num_days)
    dates = X.flatten()
    prices = y.flatten()

    scaler = StandardScaler()
    dates_scaled = scaler.fit_transform(dates.reshape(-1, 1)).flatten()

    X_all = np.column_stack([stock_ids, dates_scaled])
    y_all = prices

    model = StockPricePredictor(num_stocks=len(stock_to_idx))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses = []
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_all), batch_size):
            batch_X = X_all[i : i + batch_size]
            batch_y = y_all[i : i + batch_size]
            stock_ids = torch.tensor(batch_X[:, 0], dtype=torch.long)
            dates = torch.tensor(batch_X[:, 1], dtype=torch.float32)
            batch_y = torch.tensor(batch_y, dtype=torch.float32)
            optimizer.zero_grad()
            preds = model(stock_ids, dates).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        train_losses.append(loss.item())
        print(
            f"Epoch {epoch+1}: Train Loss={loss.item():.4f}"
        )

    # Plot predictions vs real prices for all tickers
    plt.figure(figsize=(12, 6))
    for symbol in symbols:
        idx = stock_to_idx[symbol]
        dates = X[idx]
        prices_actual = y[idx]
        dates_norm = scaler.transform(dates.reshape(-1, 1)).flatten()
        stock_ids_tensor = torch.tensor([idx] * len(dates_norm), dtype=torch.long)
        dates_tensor = torch.tensor(dates_norm, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            preds = model(stock_ids_tensor, dates_tensor).squeeze(-1).numpy()
        plt.plot(dates, prices_actual, label=f"{symbol} Actual")
        plt.plot(dates, preds, "--", label=f"{symbol} Predicted")
    plt.xlabel(f"Days since {start}")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices for All Tickers")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
