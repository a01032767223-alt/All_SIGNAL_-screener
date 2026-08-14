"""지표 → 0~100 점수 변환 규칙.

각 함수는 (score, detail, ok) 를 돌려준다.
  score : 0~100 연속값 (순위를 만들기 위해 이분법을 쓰지 않는다)
  detail: 대시보드에 그대로 보여줄 사람말 설명
  ok    : 사용자가 정의한 '핵심 매수 조건' 충족 여부 (체크리스트용)
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C


def _f(x, default=float("nan")) -> float:
    """NaN/None 안전 변환."""
    try:
        v = float(x)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """구간 선형보간 — 지표값을 점수로 부드럽게 매핑."""
    return float(np.interp(x, xs, ys))


# ─────────────────────────────────────────────────────────
# 1. 이동평균선 (25%)  — 20MA > 60MA + 가격이 20MA 위
# ─────────────────────────────────────────────────────────
def score_ma(cur, prev5) -> tuple[float, str, bool]:
    close = _f(cur["close"])
    ma5, ma20, ma60 = _f(cur.get("ma5")), _f(cur.get("ma20")), _f(cur.get("ma60"))
    if any(math.isnan(v) for v in (close, ma20, ma60)):
        return 0.0, "이동평균 계산에 필요한 데이터 부족", False

    pts, notes = 0.0, []

    # 20MA > 60MA (중기 추세 전환)
    gap = (ma20 / ma60 - 1) * 100 if ma60 else 0.0
    if ma20 > ma60:
        pts += _interp(gap, [0, 3], [30, 40])
        notes.append(f"20MA>60MA (+{gap:.1f}%)")
    else:
        pts += _interp(gap, [-5, 0], [0, 15])
        notes.append(f"20MA<60MA ({gap:.1f}%)")

    # 종가 > 20MA
    dev = (close / ma20 - 1) * 100 if ma20 else 0.0
    if close > ma20:
        pts += 30
        notes.append(f"종가 20MA 위 (이격 {dev:+.1f}%)")
    else:
        pts += _interp(dev, [-5, 0], [0, 12])
        notes.append(f"종가 20MA 아래 ({dev:+.1f}%)")

    # 20MA 기울기 (5봉)
    ma20_prev = _f(prev5.get("ma20")) if prev5 is not None else float("nan")
    slope = (ma20 / ma20_prev - 1) * 100 if ma20_prev and not math.isnan(ma20_prev) else 0.0
    pts += _interp(slope, [-1, 0, 1.5], [0, 8, 20])
    notes.append(f"20MA 5일 기울기 {slope:+.2f}%")

    # 정배열
    if not math.isnan(ma5) and ma5 > ma20 > ma60:
        pts += 10
        notes.append("정배열(5>20>60)")

    # 과열 감점 — 20MA 대비 15% 이상 벌어지면 추격매수 구간
    if dev > 15:
        penalty = min(25.0, (dev - 15) * 1.5)
        pts -= penalty
        notes.append(f"과열 감점 −{penalty:.0f}")

    ok = (ma20 > ma60) and (close > ma20)
    return _clamp(pts), " · ".join(notes), ok


# ─────────────────────────────────────────────────────────
# 2. 거래량 (20%) — 상승 시 거래량 ≥ 평균의 1.5배
# ─────────────────────────────────────────────────────────
def score_volume(cur) -> tuple[float, str, bool]:
    vr = _f(cur.get("vol_ratio"))
    ret = _f(cur.get("ret_pct"), 0.0)
    if math.isnan(vr):
        return 0.0, "거래량 평균 산출 불가", False

    base = _interp(vr, [0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 4.0],
                       [5,   18,  30,  65,  85,  100, 100])
    if ret < -0.5:
        base *= 0.40
        note = f"거래량 {vr:.1f}배지만 하락 마감({ret:+.1f}%) → 매도 물량 가능"
    elif ret < 0.5:
        base *= 0.75
        note = f"거래량 {vr:.1f}배, 보합({ret:+.1f}%)"
    else:
        note = f"상승({ret:+.1f}%) + 거래량 20일 평균의 {vr:.1f}배"

    ok = (vr >= 1.5) and (ret > 0)
    return _clamp(base), note, ok


# ─────────────────────────────────────────────────────────
# 3. 가격구조 (20%) — 저항 돌파 또는 주요 지지선 반등
# ─────────────────────────────────────────────────────────
def score_structure(cur) -> tuple[float, str, bool]:
    close = _f(cur["close"])
    prior_high, low_n, high_n = _f(cur.get("prior_high")), _f(cur.get("low_n")), _f(cur.get("high_n"))
    ma20, ma60 = _f(cur.get("ma20")), _f(cur.get("ma60"))
    ret = _f(cur.get("ret_pct"), 0.0)
    if math.isnan(prior_high) or math.isnan(low_n) or math.isnan(high_n) or high_n <= low_n:
        return 0.0, "가격 구조 판단에 필요한 데이터 부족", False

    # 60봉 박스 내 위치 (0=바닥, 1=천장)
    pos = (close - low_n) / (high_n - low_n)

    # (a) 저항 돌파
    if close > prior_high:
        over = (close / prior_high - 1) * 100
        return _clamp(_interp(over, [0, 3], [92, 100])), \
               f"60봉 최고가 돌파 (+{over:.1f}%)", True
    if close >= prior_high * 0.98:
        return 82.0, f"저항({prior_high:,.0f}) 돌파 임박 — 박스 상단 {pos*100:.0f}% 지점", True

    # (b) 주요 지지선 반등
    supports = [("20MA", ma20), ("60MA", ma60), ("직전 저점", _f(cur.get("prior_low")))]
    for name, lvl in supports:
        if math.isnan(lvl) or lvl <= 0:
            continue
        near = abs(close / lvl - 1) * 100
        if near <= 3 and ret > 0:
            return _clamp(_interp(near, [0, 3], [80, 68])), \
                   f"{name}({lvl:,.0f}) 지지 반등 ({ret:+.1f}%)", True

    # (c) 그 외 — 박스 내 위치로 채점
    sc = _interp(pos, [0.0, 0.3, 0.6, 0.85, 1.0], [15, 28, 55, 72, 80])
    if ret <= 0:
        sc *= 0.85
    return _clamp(sc), f"박스 내 {pos*100:.0f}% 지점 (저항 {prior_high:,.0f})", False


# ─────────────────────────────────────────────────────────
# 4. RSI (10%) — 30~50에서 상승 전환
# ─────────────────────────────────────────────────────────
def score_rsi(cur, prev) -> tuple[float, str, bool]:
    r = _f(cur.get("rsi"))
    rp = _f(prev.get("rsi")) if prev is not None else float("nan")
    if math.isnan(r):
        return 0.0, "RSI 데이터 부족", False
    rising = (not math.isnan(rp)) and r > rp
    arrow = "상승 중" if rising else "하락·횡보"

    if 30 <= r <= 50:
        sc = 100.0 if rising else 45.0
    elif 50 < r <= 60:
        sc = 80.0 if rising else 50.0
    elif 60 < r <= 70:
        sc = 55.0 if rising else 38.0
    elif r > 70:
        sc = _clamp(_interp(r, [70, 85], [40, 10]))
    else:  # r < 30
        sc = 35.0 if rising else 12.0

    ok = (30 <= r <= 50) and rising
    return _clamp(sc), f"RSI {r:.1f} ({arrow})", ok


# ─────────────────────────────────────────────────────────
# 5. MACD (8%) — 골든크로스 + 히스토그램 증가
# ─────────────────────────────────────────────────────────
def score_macd(cur, prev, recent) -> tuple[float, str, bool]:
    m, s, h = _f(cur.get("macd")), _f(cur.get("macd_signal")), _f(cur.get("macd_hist"))
    hp = _f(prev.get("macd_hist")) if prev is not None else float("nan")
    if any(math.isnan(v) for v in (m, s, h)):
        return 0.0, "MACD 데이터 부족", False

    hist_up = (not math.isnan(hp)) and h > hp
    # 최근 3봉 내 골든크로스 여부
    gc = False
    if recent is not None and len(recent) >= 2:
        diff = (recent["macd"] - recent["macd_signal"]).tail(4).tolist()
        for i in range(1, len(diff)):
            if diff[i - 1] <= 0 < diff[i]:
                gc = True

    if gc and hist_up:
        sc, note = 100.0, "골든크로스 3봉 이내 + 히스토그램 증가"
    elif gc:
        sc, note = 88.0, "골든크로스 3봉 이내"
    elif m > s and hist_up:
        sc, note = 82.0, "시그널 위 + 히스토그램 증가"
    elif m > s:
        sc, note = 66.0, "시그널 위 (모멘텀 둔화)"
    elif hist_up:
        sc, note = 45.0, "시그널 아래지만 히스토그램 축소(반등 조짐)"
    else:
        sc, note = 15.0, "데드크로스 구간"

    if m > 0:
        sc = _clamp(sc + 5)
        note += " · MACD 0선 위"

    ok = gc and hist_up
    return _clamp(sc), note, ok


# ─────────────────────────────────────────────────────────
# 6. 볼린저밴드 (7%) — 하단 반등 또는 수축 후 상단 돌파
# ─────────────────────────────────────────────────────────
def score_bb(cur) -> tuple[float, str, bool]:
    close, up, lo = _f(cur["close"]), _f(cur.get("bb_upper")), _f(cur.get("bb_lower"))
    pct_b, wpct = _f(cur.get("bb_pct_b")), _f(cur.get("bb_width_pct"))
    ret = _f(cur.get("ret_pct"), 0.0)
    if any(math.isnan(v) for v in (up, lo, pct_b)):
        return 0.0, "볼린저밴드 데이터 부족", False

    squeeze = (not math.isnan(wpct)) and wpct <= 0.20
    sq_txt = "밴드 수축 후 " if squeeze else ""

    if close > up:
        sc = 100.0 if squeeze else 74.0
        note, ok = f"{sq_txt}상단 돌파 (%B {pct_b:.2f})", squeeze
    elif squeeze and pct_b >= 0.55:
        sc, note, ok = 86.0, f"밴드 수축 + 상단 방향 (%B {pct_b:.2f})", True
    elif pct_b <= 0.15 and ret > 0:
        sc, note, ok = 80.0, f"하단 반등 (%B {pct_b:.2f}, {ret:+.1f}%)", True
    elif 0.5 <= pct_b <= 0.9:
        sc, note, ok = 64.0, f"중단선 위 진행 (%B {pct_b:.2f})", False
    else:
        sc = _interp(pct_b, [0.0, 0.3, 0.5], [25, 38, 52])
        note, ok = f"%B {pct_b:.2f}", False
    return _clamp(sc), note, ok


# ─────────────────────────────────────────────────────────
# 7. ADX (5%) — ADX > 20~25 이면서 +DI > -DI
# ─────────────────────────────────────────────────────────
def score_adx(cur) -> tuple[float, str, bool]:
    a, p, m = _f(cur.get("adx")), _f(cur.get("plus_di")), _f(cur.get("minus_di"))
    if any(math.isnan(v) for v in (a, p, m)):
        return 0.0, "ADX 데이터 부족", False
    bull = p > m
    if not bull:
        sc = _clamp(_interp(a, [15, 30], [30, 8]))
        return sc, f"ADX {a:.0f} · −DI 우위(하락 추세)", False
    if a >= 25:
        sc = 100.0 if a <= 45 else _clamp(_interp(a, [45, 60], [100, 78]))
    elif a >= 20:
        sc = _interp(a, [20, 25], [70, 95])
    else:
        sc = _interp(a, [10, 20], [25, 65])
    ok = a >= 20 and bull
    return _clamp(sc), f"ADX {a:.0f} · +DI {p:.0f} > −DI {m:.0f}", ok


# ─────────────────────────────────────────────────────────
# 8. OBV (5%) — 매집 흐름 확인
# ─────────────────────────────────────────────────────────
def score_obv(cur) -> tuple[float, str, bool]:
    o, sl, mx = _f(cur.get("obv")), _f(cur.get("obv_slope")), _f(cur.get("obv_max60"))
    if math.isnan(sl):
        return 0.0, "OBV 데이터 부족", False
    at_high = (not math.isnan(mx)) and (not math.isnan(o)) and o >= mx * 0.999
    if sl > 0 and at_high:
        return 100.0, "OBV 상승 + 60봉 신고 (매집)", True
    if sl > 0:
        return _clamp(_interp(sl, [0, 15], [70, 92])), f"OBV 20봉 상승 ({sl:+.1f}%)", True
    if sl > -5:
        return 42.0, f"OBV 횡보 ({sl:+.1f}%)", False
    return 15.0, f"OBV 하락 ({sl:+.1f}%)", False


# ─────────────────────────────────────────────────────────
# 통합
# ─────────────────────────────────────────────────────────
CORE_CONDITION_TEXT = {
    "ma": "20MA > 60MA + 가격이 20MA 위",
    "volume": "상승 시 거래량 ≥ 평균의 1.5배",
    "structure": "저항 돌파 또는 주요 지지선 반등",
    "rsi": "30~50에서 상승 전환",
    "macd": "골든크로스 + 히스토그램 증가",
    "bb": "하단 반등 또는 수축 후 상단 돌파",
    "adx": "ADX > 20~25 + +DI > −DI",
    "obv": "OBV 상승 추세 (매집)",
}


def score_all(edf) -> dict:
    """지표가 붙은 DataFrame(enrich 결과) → 지표별 점수 dict."""
    cur = edf.iloc[-1]
    prev = edf.iloc[-2] if len(edf) >= 2 else None
    prev5 = edf.iloc[-6] if len(edf) >= 6 else None
    recent = edf.tail(6)

    raw = {
        "ma":        score_ma(cur, prev5),
        "volume":    score_volume(cur),
        "structure": score_structure(cur),
        "rsi":       score_rsi(cur, prev),
        "macd":      score_macd(cur, prev, recent),
        "bb":        score_bb(cur),
        "adx":       score_adx(cur),
        "obv":       score_obv(cur),
    }

    parts, total = {}, 0.0
    for key, (sc, note, ok) in raw.items():
        w = C.WEIGHTS[key]
        total += sc * w / 100.0
        parts[key] = {
            "label": C.INDICATOR_LABELS[key],
            "score": round(sc, 1),
            "weight": w,
            "contrib": round(sc * w / 100.0, 2),
            "detail": note,
            "ok": bool(ok),
            "condition": CORE_CONDITION_TEXT[key],
        }
    return {"total": round(_clamp(total), 2), "parts": parts}
