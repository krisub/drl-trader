# Using FNSPID on Hugging Face.

from datasets import load_dataset
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import csv

TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WMT",
    "DIS",
    "GOOGL",
    "NVDA",
    "AMZN",
]

START_YEAR = 2012
END_YEAR = 2023
INTERMEDIATE_FILE = "filtered_news_raw.csv"
FINAL_FILE = "fnspid_embeddings.csv"


def fetch_and_filter_stream():
    ticker_set = set(TICKERS)
    dataset = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

    with open(INTERMEDIATE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "ticker", "title"])

    buffer = []
    chunk_size = 1000
    total_saved = 0

    for i, row in tqdm(enumerate(dataset), desc="Scanning Stream", mininterval=1.0):
        stock = row.get("Stock_symbol")
        date_str = row.get("Date")
        title = row.get("Article_title")

        if not stock or not date_str or not title:
            continue

        if str(stock).upper() not in ticker_set:
            continue

        # FNSPID format: "YYYY-MM-DD HH:MM:SS UTC"
        try:
            year = int(str(date_str)[:4])
            if year < START_YEAR or year > END_YEAR:
                continue
        except ValueError:
            continue

        buffer.append([date_str, str(stock).upper(), title])

        if len(buffer) >= chunk_size:
            with open(INTERMEDIATE_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(buffer)
            total_saved += len(buffer)
            buffer = []

    if buffer:
        with open(INTERMEDIATE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(buffer)
        total_saved += len(buffer)

    print(f"\nSaved {total_saved} rows to {INTERMEDIATE_FILE}")


def generate_embeddings():
    if not os.path.exists(INTERMEDIATE_FILE):
        print("Error: Intermediate file not found.")
        return

    print("Loading FinBERT...")
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertModel.from_pretrained("yiyanghkust/finbert-tone")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    df_raw = pd.read_csv(INTERMEDIATE_FILE)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True).dt.date
    df_raw = df_raw.sort_values(["date", "ticker"])

    headlines = df_raw["title"].astype(str).tolist()
    embeddings_list = []

    batch_size = 32

    print(f"Embedding {len(headlines)} headlines...")
    for i in tqdm(range(0, len(headlines), batch_size), desc="Embedding"):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(cls_embeddings)

    all_embeddings = np.vstack(embeddings_list)
    df_emb = pd.DataFrame(all_embeddings)
    df_meta = df_raw[["date", "ticker"]].reset_index(drop=True)
    df_combined = pd.concat([df_meta, df_emb], axis=1)
    daily_news = df_combined.groupby(["date", "ticker"]).mean().reset_index()

    daily_news.to_csv(FINAL_FILE, index=False)
    print(f"Embeddings saved to {FINAL_FILE}")


# Run once (H100 GPU), saved to CSV.
if __name__ == "__main__":
    fetch_and_filter_stream()
    generate_embeddings()
