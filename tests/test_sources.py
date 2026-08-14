"""데이터 소스 로직 검증 (네트워크 호출 없이 응답을 가짜로 주입).

실행: python tests/test_sources.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from screener.sources import upbit


def test_upbit_warning_flag():
    """유의/주의 종목 판정.

    caution은 항목별 bool dict라서 '딕셔너리가 비어있지 않은가'로 보면 안 된다.
    전부 False인 dict를 위험으로 판정하면 멀쩡한 종목이 유니버스에서 사라진다.
    """
    samples = [
        ({"market": "KRW-BTC", "korean_name": "비트코인",
          "market_event": {"warning": False, "caution": {}}}, False),
        ({"market": "KRW-AAA", "korean_name": "정상",
          "market_event": {"warning": False,
                           "caution": {"PRICE_FLUCTUATIONS": False,
                                       "TRADING_VOLUME_SOARING": False}}}, False),
        ({"market": "KRW-BBB", "korean_name": "주의",
          "market_event": {"warning": False,
                           "caution": {"PRICE_FLUCTUATIONS": True}}}, True),
        ({"market": "KRW-CCC", "korean_name": "유의",
          "market_event": {"warning": True, "caution": {}}}, True),
        ({"market": "KRW-DDD", "korean_name": "구형API",
          "market_warning": "CAUTION"}, True),
    ]
    payload = [s[0] for s in samples] + [
        {"market": "BTC-ETH", "korean_name": "KRW아님", "market_event": {}}]

    orig = upbit._get
    upbit._get = lambda path, params=None, retries=4: payload
    try:
        df = upbit.list_markets()
    finally:
        upbit._get = orig

    assert "BTC-ETH" not in df.index, "KRW 마켓만 남아야 한다"
    for raw, expect in samples:
        assert bool(df.loc[raw["market"], "warning"]) is expect, raw["market"]


def test_upbit_candle_parsing():
    """캔들 응답이 오름차순 OHLCV DataFrame으로 정규화되는지."""
    raw = [
        {"candle_date_time_kst": "2026-08-14T12:00:00", "opening_price": 110,
         "high_price": 115, "low_price": 108, "trade_price": 112,
         "candle_acc_trade_volume": 5.0},
        {"candle_date_time_kst": "2026-08-14T08:00:00", "opening_price": 100,
         "high_price": 112, "low_price": 99, "trade_price": 110,
         "candle_acc_trade_volume": 3.0},
    ]
    orig = upbit._get
    upbit._get = lambda path, params=None, retries=4: raw
    try:
        df = upbit.candles("KRW-BTC", "4h", 200)
    finally:
        upbit._get = orig

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing, "업비트는 최신순으로 주므로 뒤집어야 한다"
    assert df["close"].iloc[-1] == 112
    assert (df["high"] >= df["low"]).all()


def test_kr_universe_filter():
    """시총·거래대금·우선주/스팩 제외가 실제로 걸러내는지."""
    from screener import config as C
    from screener.sources import kr_stock

    meta = pd.DataFrame(
        {"name": ["정상종목", "소형주", "저거래", "삼성전자우", "케이스팩1호", "ETF상품"],
         "marketcap": [1e12, 1e10, 1e12, 1e12, 1e12, 1e12],
         "turnover": [1e9, 1e9, 1e7, 1e9, 1e9, 1e9],
         "market": ["KOSPI"] * 6},
        index=["000001", "000002", "000003", "000004", "000005", "000006"])

    out = kr_stock.apply_universe_filter(meta, excluded={"000006"})
    assert list(out.index) == ["000001"], list(out.index)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print("\n실패" if fails else "\n전체 통과", f"({fails} failed)")
    raise SystemExit(1 if fails else 0)
