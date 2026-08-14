"""OHLCV 증분 캐시 (parquet).

전종목 400일치를 매번 새로 받으면 느리므로, 리포지토리에 캐시를 두고
새 영업일 데이터만 덧붙인다. 첫 실행만 오래 걸리고 이후엔 수 초.
"""
from __future__ import annotations

import os

import pandas as pd

from . import config as C

LONG_COLS = ["date", "ticker", "open", "high", "low", "close", "volume", "value"]


def _path(name: str) -> str:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    return os.path.join(C.CACHE_DIR, f"{name}.parquet")


def load(name: str) -> pd.DataFrame:
    p = _path(name)
    if not os.path.exists(p):
        return pd.DataFrame(columns=LONG_COLS)
    try:
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:  # 손상 시 캐시 무시하고 재수집
        print(f"[cache] {name} 읽기 실패 → 무시하고 재수집: {e}")
        return pd.DataFrame(columns=LONG_COLS)


def save(name: str, df: pd.DataFrame, keep_days: int = C.HISTORY_DAYS) -> None:
    if df.empty:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    # 최근 keep_days 영업일만 보관 (리포 용량 관리)
    keep_dates = sorted(df["date"].unique())[-keep_days:]
    df = df[df["date"].isin(keep_dates)]
    df.sort_values(["ticker", "date"]).to_parquet(_path(name), index=False,
                                                  compression="zstd")
    print(f"[cache] {name} 저장: {len(df):,}행 / {df['ticker'].nunique():,}종목")


def cached_dates(name: str) -> set:
    df = load(name)
    if df.empty:
        return set()
    return set(pd.to_datetime(df["date"]).dt.strftime("%Y%m%d"))


def to_frames(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """long format → {ticker: OHLCV DataFrame(index=date)}"""
    out = {}
    long_df = long_df.sort_values("date")
    for ticker, g in long_df.groupby("ticker", sort=False):
        d = g.set_index("date")[["open", "high", "low", "close", "volume"]].astype("float64")
        d = d[~d.index.duplicated(keep="last")]
        out[str(ticker)] = d
    return out
