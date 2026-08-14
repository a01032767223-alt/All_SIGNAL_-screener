"""국내주식(KOSPI/KOSDAQ) 데이터 — pykrx(KRX 공식 데이터) 기반.

핵심: 전종목 하루치를 한 번의 호출로 받는다(get_market_ohlcv_by_ticker).
따라서 400영업일 = 400콜. 첫 실행만 오래 걸리고 이후엔 캐시에 없는 날짜만 받는다.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd

from .. import cache
from .. import config as C

CACHE_NAME = "kr_ohlcv"


def _stock():
    from pykrx import stock  # 지연 import (테스트 환경에서 미설치여도 무방)
    return stock


def _ohlcv_by_ticker(stock, date_str: str) -> pd.DataFrame | None:
    for fn in ("get_market_ohlcv_by_ticker", "get_market_ohlcv"):
        f = getattr(stock, fn, None)
        if f is None:
            continue
        try:
            df = f(date_str, market="ALL")
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


def _business_days(n_days: int) -> list[str]:
    """최근 n_days 달력일 중 주말 제외 (휴장일은 조회 결과가 비어 자연히 걸러짐)."""
    today = datetime.now()
    out = []
    for i in range(int(n_days * 1.45)):
        d = today - timedelta(days=i)
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        if len(out) >= n_days:
            break
    return sorted(out)


def fetch_ohlcv(history_days: int = C.HISTORY_DAYS, verbose: bool = True) -> pd.DataFrame:
    """전종목 일봉 long-format DataFrame 반환 (캐시 증분 갱신)."""
    stock = _stock()
    old = cache.load(CACHE_NAME)
    have = cache.cached_dates(CACHE_NAME)
    want = _business_days(history_days)
    todo = [d for d in want if d not in have]

    if verbose:
        print(f"[kr] 캐시 {len(have)}일 보유 · {len(todo)}일 신규 수집")

    rows = []
    for i, d in enumerate(todo, 1):
        df = _ohlcv_by_ticker(stock, d)
        if df is None or df.empty:
            continue  # 휴장일
        df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low",
                                "종가": "close", "거래량": "volume", "거래대금": "value"})
        need = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            continue
        if "value" not in df.columns:
            df["value"] = df["close"] * df["volume"]
        sub = df[need + ["value"]].copy()
        sub = sub[sub["close"] > 0]
        sub["ticker"] = sub.index.astype(str)
        sub["date"] = pd.to_datetime(d)
        rows.append(sub.reset_index(drop=True))
        if verbose and i % 25 == 0:
            print(f"  ... {i}/{len(todo)}")
        time.sleep(0.12)  # KRX 예의상 간격

    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cache.LONG_COLS)
    allx = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if allx.empty:
        raise RuntimeError("국내주식 데이터를 한 건도 받지 못했습니다 (KRX 응답 확인 필요)")
    cache.save(CACHE_NAME, allx)
    return allx


def fetch_meta(date_str: str | None = None) -> pd.DataFrame:
    """종목명·시가총액 등 메타 (1~2콜)."""
    stock = _stock()
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    names = pd.DataFrame()
    for back in range(0, 10):
        d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=back)).strftime("%Y%m%d")
        try:
            cap = stock.get_market_cap_by_ticker(d, market="ALL")
            if cap is not None and not cap.empty:
                names = cap.rename(columns={"시가총액": "marketcap", "거래대금": "turnover"})
                names.index = names.index.astype(str)
                break
        except Exception:
            continue
    if names.empty:
        return pd.DataFrame(columns=["name", "marketcap", "turnover", "market"])

    # 종목명 — 한 번의 호출로 전체 확보
    try:
        chg = stock.get_market_price_change_by_ticker(d, d, market="ALL")
        name_map = chg["종목명"].astype(str).to_dict() if "종목명" in chg.columns else {}
        name_map = {str(k): v for k, v in name_map.items()}
    except Exception:
        name_map = {}
    names["name"] = [name_map.get(t, t) for t in names.index]

    # 시장 구분
    market_map = {}
    for mk in ("KOSPI", "KOSDAQ"):
        try:
            for t in stock.get_market_ticker_list(d, market=mk):
                market_map[str(t)] = mk
        except Exception:
            pass
    names["market"] = [market_map.get(t, "") for t in names.index]

    keep = [c for c in ("name", "marketcap", "turnover", "market") if c in names.columns]
    return names[keep]


def excluded_tickers(date_str: str | None = None) -> set:
    """ETF·ETN·ELW 등 지표 해석이 다른 종목 제외."""
    stock = _stock()
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    out = set()
    for fn in ("get_etf_ticker_list", "get_etn_ticker_list", "get_elw_ticker_list"):
        f = getattr(stock, fn, None)
        if f is None:
            continue
        try:
            out |= {str(t) for t in f(date_str)}
        except Exception:
            pass
    return out


def apply_universe_filter(meta: pd.DataFrame, excluded: set) -> pd.DataFrame:
    """시총·거래대금·종목명 패턴 기본 필터."""
    df = meta.copy()
    before = len(df)
    if "marketcap" in df.columns:
        df = df[df["marketcap"] >= C.KR_MIN_MARKETCAP]
    if "turnover" in df.columns:
        df = df[df["turnover"] >= C.KR_MIN_TURNOVER]
    df = df[~df.index.isin(excluded)]
    if "name" in df.columns:
        pat = "|".join(C.KR_EXCLUDE_PATTERNS)
        df = df[~df["name"].astype(str).str.contains(pat, na=False)]
        df = df[~df["name"].astype(str).str.endswith("우")]
    print(f"[kr] 유니버스 필터: {before:,} → {len(df):,}종목")
    return df
