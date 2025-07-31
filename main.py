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
    def __init__(self, num_stocks, embedding_dim=8, hidden_dim=32):
        super().__init__()
        self.stock_embed = nn.Embedding(num_stocks, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, stock_ids, dates):
        stock_embeds = self.stock_embed(stock_ids)
        if dates.dim() == 1:
            dates = dates.unsqueeze(1)
        elif dates.dim() == 2 and dates.shape[1] != 1:
            dates = dates[:, 0].unsqueeze(1)
        x = torch.cat([stock_embeds, dates], dim=1)
        x = F.relu(self.fc1(x))
        price = self.fc2(x)
        return price


def fetch_data(symbols, start, end):
    all_X, all_y = [], []
    stock_to_idx = {s: i for i, s in enumerate(symbols)}
    for symbol in symbols:
        data = yf.download(symbol, start=start, end=end, auto_adjust=False)
        data.reset_index(inplace=True)  # Reset index to make Date a column
        data["Date"] = pd.to_datetime(data["Date"])  # Convert Date column to datetime

        # I already know which stock it is here so lets move away from stupid pandas
        # Convert pandas datetime to numpy float timestamps efficiently
        dates_np = data["Date"].dt.strftime("%Y%m%d").astype(int).to_numpy()
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

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    model = StockPricePredictor(num_stocks=len(stock_to_idx))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
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
        with torch.no_grad():
            stock_ids = torch.tensor(X_val[:, 0], dtype=torch.long)
            dates = torch.tensor(X_val[:, 1], dtype=torch.float32)
            val_y_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_preds = model(stock_ids, dates)
            val_loss = criterion(val_preds, val_y_tensor)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(
            f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}"
        )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    # plt.legend()
    # plt.show()

    # Example prediction
    stock_id = torch.tensor([stock_to_idx["AAPL"]], dtype=torch.long)
    date_to_lookup = 20101212
    date_norm = scaler.transform(np.array([[date_to_lookup]]))
    date = torch.tensor(date_norm.reshape(-1, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(stock_id, date)
    print(f"Predicted Price for AAPL on {date_to_lookup}: ${pred.item():.2f}")


if __name__ == "__main__":
    main()
