"""스크리너 엔트리포인트.

사용법:
  python -m screener.run --market kr          # 국내주식
  python -m screener.run --market coin        # 업비트 코인
  python -m screener.run --market coin --notify
  python -m screener.run --market demo        # 합성 데이터로 파이프라인 점검
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config as C
from . import rules as R
from . import score as S

KST = timezone(timedelta(hours=9))
OUT_DIR = os.path.join("docs", "data")
HIST_DIR = os.path.join(OUT_DIR, "history")

MARKET_LABEL = {"kr": "국내주식", "coin": "코인(업비트)"}


# ─────────────────────────────────────────────────────────
def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[out] {path} ({os.path.getsize(path)/1024:.0f} KB)")


def _load_prev(market: str) -> dict:
    p = os.path.join(OUT_DIR, f"{market}_latest.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _diff_new(prev: dict, items: list[dict], grades=("S", "A")) -> list[dict]:
    """직전 실행 대비 새로 상위 등급에 진입한 종목 (알림 스팸 방지)."""
    old = {i["symbol"] for i in prev.get("items", []) if i.get("grade") in grades}
    return [i for i in items if i["grade"] in grades and i["symbol"] not in old]


# ─────────────────────────────────────────────────────────
def screen_kr() -> dict:
    from .sources import kr_stock

    frames, meta = kr_stock.load()

    items, errors, last_date = [], 0, None
    for ticker, df in frames.items():
        try:
            if len(df) < C.MIN_BARS:
                continue
            res = S.evaluate({"1d": df, "1w": S.resample_weekly(df)}, "kr")
            if last_date is None or df.index[-1] > last_date:
                last_date = df.index[-1]
            if res is None or res["score"] < C.MIN_OUTPUT_SCORE:
                continue
            info = meta.loc[ticker]
            res.update({
                "symbol": ticker,
                "name": str(info.get("name", ticker)),
                "market": str(info.get("market", "")),
                "turnover": float(info.get("turnover", 0.0) or 0.0),
                "marketcap": float(info.get("marketcap", 0.0) or 0.0),
                "link": f"https://m.stock.naver.com/domestic/stock/{ticker}/total",
            })
            items.append(res)
        except Exception:
            errors += 1
            if errors <= 3:
                traceback.print_exc()

    data_date = str(last_date)[:10] if last_date is not None else \
        datetime.now(KST).strftime("%Y-%m-%d")
    print(f"[kr] 평가 {len(frames):,}종목 → 후보 {len(items):,} (오류 {errors})")
    return _payload("kr", items, len(frames), data_date)


def screen_coin() -> dict:
    from .sources import upbit

    uni = upbit.universe()
    items, errors = [], 0
    for market, row in uni.iterrows():
        try:
            frames = {}
            for tf in ("4h", "1d", "1w"):
                df = upbit.candles(market, tf, 200)
                if not df.empty:
                    frames[tf] = df
            if "1d" not in frames or len(frames["1d"]) < C.COIN_MIN_LISTED_DAYS:
                continue
            res = S.evaluate(frames, "coin")
            if res is None or res["score"] < C.MIN_OUTPUT_SCORE:
                continue
            res.update({
                "symbol": market,
                "name": str(row["name"]),
                "market": "업비트 KRW",
                "turnover": float(row["acc_trade_price_24h"]),
                "marketcap": 0.0,
                "link": f"https://upbit.com/exchange?code=CRIX.UPBIT.{market}",
            })
            items.append(res)
        except Exception:
            errors += 1
            if errors <= 3:
                traceback.print_exc()

    print(f"[coin] 평가 {len(uni)}종목 → 후보 {len(items)} (오류 {errors})")
    return _payload("coin", items, len(uni), datetime.now(KST).strftime("%Y-%m-%d"))


def screen_demo() -> dict:
    """네트워크 없이 파이프라인·대시보드를 점검하기 위한 합성 데이터."""
    import numpy as np

    rng = np.random.default_rng(7)
    items = []
    for i in range(24):
        n = 260
        drift = rng.normal(0.0012 if i % 3 == 0 else -0.0003, 0.001)
        steps = rng.normal(drift, 0.022, n)
        if i % 3 == 0:                       # 최근 상승 + 거래량 증가 종목
            steps[-12:] += 0.012
        close = 10000 * np.exp(np.cumsum(steps))
        high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
        open_ = np.r_[close[0], close[:-1]]
        vol = rng.lognormal(11, 0.4, n)
        if i % 3 == 0:
            vol[-3:] *= rng.uniform(1.8, 3.2)
        # 주말에 실행하면 bdate_range(end=토/일, periods=n)이 n-1개만 반환하는
        # pandas 특성이 있어(끝점이 영업일이 아니면 카운트가 어긋남), 여유 있게
        # 만든 뒤 뒤에서 n개만 잘라 항상 길이를 맞춘다.
        idx = pd.bdate_range(end=datetime.now(), periods=n + 5)[-n:]
        df = pd.DataFrame({"open": open_, "high": high, "low": low,
                           "close": close, "volume": vol}, index=idx)
        res = S.evaluate({"1d": df, "1w": S.resample_weekly(df)}, "kr")
        if res is None or res["score"] < C.MIN_OUTPUT_SCORE:
            continue
        res.update({"symbol": f"00000{i:02d}", "name": f"샘플종목{i:02d}",
                    "market": "KOSPI" if i % 2 else "KOSDAQ",
                    "turnover": float(vol[-1] * close[-1]), "marketcap": 5e11,
                    "link": "#"})
        items.append(res)
    return _payload("kr", items, 24, datetime.now(KST).strftime("%Y-%m-%d"))


# ─────────────────────────────────────────────────────────
def _payload(market: str, items: list[dict], scanned: int, data_date: str) -> dict:
    items.sort(key=lambda x: x["score"], reverse=True)
    counts = {}
    for it in items:
        counts[it["grade"]] = counts.get(it["grade"], 0) + 1

    total_found = len(items)
    truncated = max(0, total_found - C.MAX_OUTPUT_ITEMS)
    if truncated:
        # 조용히 잘라내지 않고 몇 건을 뺐는지 남깁니다
        print(f"[out] 후보 {total_found:,}건 중 상위 {C.MAX_OUTPUT_ITEMS}건만 저장 "
              f"({truncated:,}건 제외, 최저 점수 {items[C.MAX_OUTPUT_ITEMS - 1]['score']:.1f})")
        items = items[:C.MAX_OUTPUT_ITEMS]

    return {
        "market": market,
        "market_label": MARKET_LABEL.get(market, market),
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "data_date": data_date,
        "scanned": scanned,
        "count": len(items),
        "total_found": total_found,
        "truncated": truncated,
        "grade_counts": counts,
        "weights": C.WEIGHTS,
        "indicator_labels": C.INDICATOR_LABELS,
        "grade_cuts": {g: c for g, c in C.GRADE_CUTS},
        # 종목마다 같은 내용이라 최상단에 한 번만 담습니다
        "grade_info": C.GRADE_INFO,
        "indicator_help": R.BEGINNER_HELP,
        "conditions": R.CORE_CONDITION_TEXT,
        "items": items,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["kr", "coin", "demo"])
    ap.add_argument("--notify", action="store_true", help="텔레그램/이메일 발송")
    ap.add_argument("--daily-summary", action="store_true",
                    help="신규 진입뿐 아니라 전체 요약을 강제로 발송")
    args = ap.parse_args()

    fn = {"kr": screen_kr, "coin": screen_coin, "demo": screen_demo}[args.market]
    payload = fn()
    market_key = payload["market"]

    prev = _load_prev(market_key)
    new_items = _diff_new(prev, payload["items"])
    payload["new_entries"] = [i["symbol"] for i in new_items]

    _write_json(os.path.join(OUT_DIR, f"{market_key}_latest.json"), payload)
    # 히스토리 스냅샷 (나중에 적중률 검증용) — 상위 60종목만 경량 보관
    slim = {k: v for k, v in payload.items() if k != "items"}
    slim["items"] = [{k: it[k] for k in ("symbol", "name", "score", "grade", "price",
                                         "change_pct", "risk")}
                     for it in payload["items"][:60]]
    _write_json(os.path.join(HIST_DIR, f"{market_key}_{payload['data_date']}.json"), slim)

    # 인덱스(대시보드가 최근 갱신 시각을 알 수 있게)
    index_path = os.path.join(OUT_DIR, "index.json")
    idx = {}
    if os.path.exists(index_path):
        try:
            idx = json.load(open(index_path, encoding="utf-8"))
        except Exception:
            idx = {}
    idx[market_key] = {"generated_at": payload["generated_at"],
                       "data_date": payload["data_date"],
                       "count": payload["count"],
                       "grade_counts": payload["grade_counts"]}
    _write_json(index_path, idx)

    if args.notify:
        from . import notify
        notify.dispatch(payload, new_items, force_summary=args.daily_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
