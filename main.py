import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class StockPricePredictor(nn.Module):
    def __init__(self, num_stocks, hidden_size=64):
        super().__init__()
        # Embedding for categorical stock IDs
        self.stock_embedding = nn.Embedding(num_stocks, 16)
        # Network to process embeddings + date
        self.fc1 = nn.Linear(16 + 1, hidden_size)  # stock embedding + date
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, stock_ids, dates):
        # Get embeddings for stock IDs
        stock_embeds = self.stock_embedding(stock_ids)

        if dates.dim() == 1:
            dates = dates.unsqueeze(1)

        # Concatenate embeddings with dates
        x = torch.cat([stock_embeds, dates], dim=1)

        # Forward through network
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def fetch_data(symbols, start, end):
    all_data = []

    for i, symbol in enumerate(symbols):
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if len(data) == 0:
            continue

        data = data.reset_index()

        # Convert to days since start
        start_date = data["Date"].iloc[0]
        days = (data["Date"] - start_date).dt.days.values
        prices = data["Close"].values

        # Create array with stock_id, date, price
        stock_data = np.column_stack(
            [
                np.full(len(days), i),  # stock_id
                days,  # days since start
                prices,  # closing prices
            ]
        )
        all_data.append(stock_data)

    if not all_data:
        raise ValueError("No valid data found.")

    # Combine all data
    combined = np.vstack(all_data)
    return combined[:, 0].astype(int), combined[:, 1], combined[:, 2]


def main():
    symbols = ["AAPL", "MSFT", "GOOG"]
    start = "2010-01-01"
    end = "2020-01-01"
    epochs = 10
    batch_size = 64

    # Fetch data
    stock_ids, dates, prices = fetch_data(symbols, start, end)

    # Normalize features and targets
    date_scaler = StandardScaler()
    price_scaler = StandardScaler()

    dates_scaled = date_scaler.fit_transform(dates.reshape(-1, 1)).flatten()
    prices_scaled = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Convert to tensors
    stock_ids_tensor = torch.tensor(stock_ids, dtype=torch.long)
    dates_tensor = torch.tensor(dates_scaled, dtype=torch.float32)
    prices_tensor = torch.tensor(prices_scaled, dtype=torch.float32)

    # Create model
    model = StockPricePredictor(num_stocks=len(symbols), hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training
    train_losses = []
    model.train()

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(stock_ids_tensor))
        stock_ids_shuffled = stock_ids_tensor[perm]
        dates_shuffled = dates_tensor[perm]
        prices_shuffled = prices_tensor[perm]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(stock_ids_tensor), batch_size):
            batch_stocks = stock_ids_shuffled[i : i + batch_size]
            batch_dates = dates_shuffled[i : i + batch_size]
            batch_prices = prices_shuffled[i : i + batch_size]

            optimizer.zero_grad()
            preds = model(batch_stocks, batch_dates).squeeze(-1)
            loss = criterion(preds, batch_prices)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

    # Plotting
    model.eval()
    plt.figure(figsize=(15, 5))

    for i, symbol in enumerate(symbols):
        plt.subplot(1, 3, i + 1)

        # Get data for this stock
        mask = stock_ids == i
        stock_dates = dates[mask]
        stock_prices = prices[mask]
        stock_dates_scaled = dates_scaled[mask]

        # Make predictions
        with torch.no_grad():
            stock_tensor = torch.tensor([i] * sum(mask), dtype=torch.long)
            dates_tensor = torch.tensor(stock_dates_scaled, dtype=torch.float32)
            preds_scaled = model(stock_tensor, dates_tensor).squeeze(-1).numpy()

        # Inverse transform predictions
        preds = price_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # Plot
        plt.plot(stock_dates, stock_prices, label="Actual", alpha=0.7)
        plt.plot(stock_dates, preds, label="Predicted", alpha=0.7)
        plt.xlabel("Days since start")
        plt.ylabel("Price ($)")
        plt.title(f"{symbol} - Actual vs Predicted")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
