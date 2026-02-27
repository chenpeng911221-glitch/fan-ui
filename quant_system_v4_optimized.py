import os
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Config:
    """全局配置。"""

    api_url: str = os.getenv(
        "OKX_API_URL", "http://43.134.54.218:8088/api/v5/market/history-candles"
    )
    symbol: str = os.getenv("SYMBOL", "ETH-USDT-SWAP")
    interval: str = os.getenv("INTERVAL", "15m")
    total_limit: int = int(os.getenv("TOTAL_LIMIT", "8000"))

    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.015"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
    atr_stop_multiplier: float = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))

    train_ratio: float = float(os.getenv("TRAIN_RATIO", "0.70"))
    min_train_samples: int = int(os.getenv("MIN_TRAIN_SAMPLES", "300"))

    proba_long: float = float(os.getenv("PROBA_LONG", "0.62"))
    proba_short: float = float(os.getenv("PROBA_SHORT", "0.38"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.55"))

    n_features_select: int = int(os.getenv("N_FEATURES_SELECT", "25"))

    base_position_size: float = float(os.getenv("BASE_POSITION_SIZE", "0.05"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.15"))

    commission_rate: float = float(os.getenv("COMMISSION_RATE", "0.0006"))
    slippage_rate: float = float(os.getenv("SLIPPAGE_RATE", "0.0002"))

    ensemble_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.ensemble_weights is None:
            object.__setattr__(
                self,
                "ensemble_weights",
                {"xgb": 0.45, "rf": 0.30, "lr": 0.25},
            )


class ExitReason(Enum):
    STOP_LOSS = "止损"
    TAKE_PROFIT = "止盈"
    TIME_STOP = "时间止损"


def calculate_max_drawdown(pnls: List[float]) -> Tuple[float, int, int]:
    """计算最大回撤（基于累计收益曲线）。"""
    if not pnls:
        return 0.0, 0, 0

    cumulative = np.cumsum(np.array(pnls, dtype=float))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative

    valley_idx = int(np.argmax(drawdowns))
    peak_idx = int(np.argmax(cumulative[: valley_idx + 1])) if valley_idx > 0 else 0
    max_dd = float(drawdowns[valley_idx])
    return -max_dd, peak_idx, valley_idx


def calculate_recovery_factor(total_pnl: float, max_drawdown: float) -> float:
    if max_drawdown >= 0:
        return 0.0
    return float(total_pnl / abs(max_drawdown))


def fetch_data(config: Config) -> pd.DataFrame:
    """分页拉取 K 线数据。"""
    limit_per_page = 300
    klines: List[List[str]] = []
    next_end = None

    print(f"获取 {config.symbol}({config.interval}) K线数据...")

    while len(klines) < config.total_limit:
        params = {
            "instId": config.symbol,
            "bar": config.interval,
            "limit": limit_per_page,
        }
        if next_end:
            params["after"] = next_end

        try:
            r = requests.get(config.api_url, params=params, timeout=15)
            r.raise_for_status()
            data = sorted(r.json().get("data", []), key=lambda x: int(x[0]))
        except Exception as exc:  # noqa: BLE001
            print(f"获取数据错误: {exc}")
            break

        if not data:
            break

        klines.extend(data)
        next_end = data[0][0]

        if len(data) < limit_per_page:
            break
        time.sleep(0.25)

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        {k[0]: k for k in klines}.values(),
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "volCcy",
            "volCcyQuote",
            "confirm",
        ],
    ).sort_values("timestamp")

    df[["open", "high", "low", "close", "vol"]] = df[
        ["open", "high", "low", "close", "vol"]
    ].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")

    print(f"成功获取 {len(df)} 条K线数据（{config.interval}）")
    return df[["timestamp", "open", "high", "low", "close", "vol"]].reset_index(drop=True)


class AdvancedFeatureEngine:
    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        tr = np.zeros_like(closes)
        tr[1:] = np.maximum.reduce(
            [
                highs[1:] - lows[1:],
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ]
        )
        atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
        default = np.nanmean(tr[tr > 0]) if np.any(tr > 0) else 0.01
        return np.nan_to_num(atr, nan=default)

    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(closes, prepend=closes[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = pd.Series(gains).rolling(period, min_periods=1).mean().to_numpy()
        avg_loss = pd.Series(losses).rolling(period, min_periods=1).mean().to_numpy()
        rs = (avg_gain + 1e-10) / (avg_loss + 1e-10)
        return (100 - (100 / (1 + rs))) / 100.0

    @staticmethod
    def calculate_macd(closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().to_numpy()
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().to_numpy()
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9, adjust=False).mean().to_numpy()
        return macd, signal

    @staticmethod
    def build_advanced_features(df: pd.DataFrame) -> np.ndarray:
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        vols = df["vol"].to_numpy(dtype=float)

        n = len(closes)
        eps = 1e-12
        feats: List[float] = []

        # 动量
        for p in (3, 5, 10):
            feats.append((closes[-1] - closes[-p]) / (closes[-p] + eps) if n > p else 0.0)
        feats.append((closes[-1] - closes[-5]) / (closes[-5] + eps) if n > 5 else 0.0)

        # 波动
        rets = np.diff(closes) / (closes[:-1] + eps) if n > 1 else np.array([0.0])
        base_vol = float(np.std(rets))
        feats.extend([
            base_vol,
            float(np.std(rets[-20:])) if len(rets) >= 20 else base_vol,
            float(np.sqrt(np.mean(np.log((highs + eps) / (lows + eps)) ** 2))) if n > 1 else 0.0,
        ])

        # 量能
        vol_ma20 = float(np.mean(vols[-20:])) if n >= 20 else float(np.mean(vols))
        feats.append(vols[-1] / (vol_ma20 + eps))
        feats.append((vols[-1] - np.mean(vols[-5:])) / (np.mean(vols[-5:]) + eps) if n > 5 else 0.0)
        feats.append((vols[-1] - 2 * vols[-2] + vols[-3]) / (np.mean(vols) + eps) if n > 3 else 0.0)

        # 趋势
        sma5 = np.mean(closes[-5:]) if n >= 5 else closes[-1]
        sma10 = np.mean(closes[-10:]) if n >= 10 else closes[-1]
        feats.extend([
            (closes[-1] - sma5) / (sma5 + eps),
            (sma5 - sma10) / (sma10 + eps),
        ])
        if n > 20:
            high20, low20 = np.max(highs[-20:]), np.min(lows[-20:])
            feats.append((closes[-1] - low20) / (high20 - low20 + eps))
        else:
            feats.append(0.5)

        # RSI + MACD
        rsi14 = AdvancedFeatureEngine.calculate_rsi(closes, 14)
        rsi7 = AdvancedFeatureEngine.calculate_rsi(closes, 7)
        feats.append(float(rsi14[-1]))
        feats.append(float(rsi7[-1]))

        macd, signal = AdvancedFeatureEngine.calculate_macd(closes)
        scale = np.max(np.abs(closes)) + eps
        feats.append(float(macd[-1] / scale))
        feats.append(float((macd[-1] - signal[-1]) / scale))

        # 布林
        sma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        std20 = np.std(closes[-20:]) if n >= 20 else 0.01
        upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
        feats.append(float(np.clip((closes[-1] - lower) / (upper - lower + eps), 0, 1)))
        feats.append(float((upper - lower) / (sma20 + eps)))

        # K线形态
        body = abs(closes[-1] - opens[-1])
        full_range = max(highs[-1] - lows[-1], eps)
        feats.extend(
            [
                body / full_range,
                (highs[-1] - max(opens[-1], closes[-1])) / full_range,
                (min(opens[-1], closes[-1]) - lows[-1]) / full_range,
                1.0 if closes[-1] > opens[-1] else 0.0,
            ]
        )

        # 均值回归
        if n > 20:
            mean20 = np.mean(closes[-20:])
            feats.append((closes[-1] - mean20) / (mean20 + eps))
            feats.append((closes[-1] - mean20) / (std20 + eps) / 3.0)
        else:
            feats.extend([0.0, 0.0])

        # ATR
        atr14 = AdvancedFeatureEngine.calculate_atr(highs, lows, closes, 14)[-1]
        atr7 = AdvancedFeatureEngine.calculate_atr(highs, lows, closes, 7)[-1]
        feats.append(float(atr14 / (closes[-1] + eps)))
        feats.append(float(atr7 / (closes[-1] + eps)))

        # 微观结构
        feats.append(float((highs[-1] / (lows[-1] + eps) - 1) * 100))
        feats.append(float((closes[-1] - lows[-1]) / (highs[-1] - lows[-1] + eps)))

        if len(rets) > 5 and len(vols) > 6:
            vdiff = np.diff(vols)
            min_len = min(len(rets), len(vdiff))
            corr = np.corrcoef(rets[-min_len:], vdiff[-min_len:])[0, 1]
            feats.append(float(np.nan_to_num(corr)))
        else:
            feats.append(0.0)

        return np.array(feats, dtype=np.float32)


def train_ensemble_model(config: Config, X_train: np.ndarray, y_train: np.ndarray):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    selector = SelectKBest(score_func=f_classif, k=min(config.n_features_select, X_scaled.shape[1]))
    X_sel = selector.fit_transform(X_scaled, y_train)

    models: Dict[str, Any] = {}

    try:
        from xgboost import XGBClassifier

        n_pos = max(np.sum(y_train == 1), 1)
        n_neg = max(np.sum(y_train == 0), 1)
        models["xgb"] = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.02,
            min_child_weight=8,
            gamma=0.2,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=n_neg / n_pos,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        models["xgb"].fit(X_sel, y_train)
        print("✓ XGBoost训练完成")
    except ImportError:
        print("⚠ XGBoost不可用，自动降级到 RF+LR")

    models["rf"] = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    models["rf"].fit(X_sel, y_train)

    models["lr"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
        random_state=42,
    )
    models["lr"].fit(X_sel, y_train)

    return models, scaler, selector


def predict_ensemble(config: Config, models: Dict[str, Any], scaler: Any, selector: Any, X: np.ndarray):
    X_sel = selector.transform(scaler.transform(X))

    probs = np.zeros((len(X_sel), 2), dtype=float)
    effective_weight_sum = 0.0

    for name, model in models.items():
        w = config.ensemble_weights.get(name, 1.0)
        probs += model.predict_proba(X_sel) * w
        effective_weight_sum += w

    probs /= max(effective_weight_sum, 1e-12)
    p_up = probs[:, 1]
    confidence = np.abs(p_up - 0.5) * 2
    return p_up, confidence


class AdaptiveRiskManager:
    @staticmethod
    def calculate_position_size(config: Config, signal_quality: float, confidence: float, recent_pnls: Optional[List[float]] = None) -> float:
        if confidence < config.min_confidence:
            return 0.01

        q = np.clip(signal_quality / 100.0, 0.0, 1.0)
        c = np.clip(confidence, 0.0, 1.0)

        if recent_pnls:
            recent = recent_pnls[-10:]
            win_rate_factor = 0.5 + (sum(1 for p in recent if p > 0) / len(recent)) * 0.5
        else:
            win_rate_factor = 0.8

        size = config.base_position_size * q * c * win_rate_factor
        return float(np.clip(size, 0.01, config.max_position_size))

    @staticmethod
    def calculate_dynamic_stops(config: Config, entry_price: float, atr_ratio: float, direction: int, recent_volatility: float = 0.01) -> Tuple[float, float]:
        atr_stop = atr_ratio * config.atr_stop_multiplier
        vol_factor = np.clip(1 + (recent_volatility - 0.01) * 2, 0.8, 1.5)

        adjusted_stop = atr_stop * vol_factor
        adjusted_tp = adjusted_stop * 1.5

        if direction == 1:
            return entry_price * (1 - adjusted_stop), entry_price * (1 + adjusted_tp)
        return entry_price * (1 + adjusted_stop), entry_price * (1 - adjusted_tp)


def walk_forward_validate(X: np.ndarray, y: np.ndarray, min_train: int = 300, step: int = 100) -> Dict[str, float]:
    """简单 walk-forward 验证，减少一次性切分的偶然性。"""
    if len(X) < min_train + step:
        return {"wf_accuracy": float("nan"), "folds": 0}

    preds_all: List[int] = []
    y_all: List[int] = []

    cfg = Config()
    for start in range(min_train, len(X) - step + 1, step):
        X_train, y_train = X[:start], y[:start]
        X_test, y_test = X[start : start + step], y[start : start + step]

        models, scaler, selector = train_ensemble_model(cfg, X_train, y_train)
        p_up, _ = predict_ensemble(cfg, models, scaler, selector, X_test)
        pred = (p_up >= 0.5).astype(int)

        preds_all.extend(pred.tolist())
        y_all.extend(y_test.tolist())

    acc = accuracy_score(y_all, preds_all) if y_all else float("nan")
    return {"wf_accuracy": float(acc), "folds": len(y_all)}


def advanced_backtest(df: pd.DataFrame, config: Config, window_size: int = 20, horizon: int = 5) -> Optional[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("开始高级特征工程回测")
    print("=" * 80)

    closes = df["close"].to_numpy(dtype=float)
    X_all, y_all, valid_indices, future_klines = [], [], [], []
    eps = 1e-12

    for i in range(0, len(closes) - window_size - horizon + 1):
        future_start = i + window_size
        future_end = future_start + horizon

        window_df = df.iloc[i : i + window_size]
        future_df = df.iloc[future_start:future_end]

        X_all.append(AdvancedFeatureEngine.build_advanced_features(window_df))

        start_price = closes[i + window_size - 1]
        end_price = closes[future_end - 1]
        y_all.append(1 if (end_price - start_price) / (start_price + eps) > 0 else 0)

        valid_indices.append(i)
        future_klines.append(future_df)

    if len(X_all) < config.min_train_samples:
        print(f"样本不足 ({len(X_all)} < {config.min_train_samples})")
        return None

    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=int)

    wf_res = walk_forward_validate(X_all, y_all, min_train=max(config.min_train_samples, 300), step=100)
    if wf_res["folds"]:
        print(f"✓ Walk-forward 准确率: {wf_res['wf_accuracy']:.2%}")

    split = int(len(X_all) * config.train_ratio)
    X_train, y_train = X_all[:split], y_all[:split]
    X_test = X_all[split:]

    models, scaler, selector = train_ensemble_model(config, X_train, y_train)
    p_up, confidences = predict_ensemble(config, models, scaler, selector, X_test)

    signals = np.full(len(X_test), -1, dtype=int)
    signals[p_up >= config.proba_long] = 1
    signals[p_up <= config.proba_short] = 0

    risk = AdaptiveRiskManager()

    stats = {"total_pnl": 0.0, "wins": 0, "losses": 0, "pnls": [], "entries": 0, "signals": 0}

    for j, s in enumerate(signals):
        if s == -1:
            continue
        stats["signals"] += 1

        if confidences[j] < config.min_confidence:
            continue

        i = valid_indices[split + j]
        entry_price = float(df.iloc[i + window_size - 1]["close"])

        position_size = risk.calculate_position_size(config, 75, float(confidences[j]), stats["pnls"][-10:])
        if position_size < 0.01:
            continue

        stats["entries"] += 1

        recent_window = df.iloc[max(0, i + window_size - 20) : i + window_size]
        atr_ratio = float(AdvancedFeatureEngine.calculate_atr(
            recent_window["high"].to_numpy(),
            recent_window["low"].to_numpy(),
            recent_window["close"].to_numpy(),
            config.atr_period,
        )[-1] / (entry_price + eps))

        stop_loss, take_profit = risk.calculate_dynamic_stops(config, entry_price, atr_ratio, int(s))

        future_df = future_klines[split + j]
        pnl = None

        for k in range(len(future_df)):
            bar = future_df.iloc[k]
            h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
            exit_price = None

            if s == 1:
                if l <= stop_loss:
                    exit_price = stop_loss
                elif h >= take_profit:
                    exit_price = take_profit
            else:
                if h >= stop_loss:
                    exit_price = stop_loss
                elif l <= take_profit:
                    exit_price = take_profit

            if exit_price is None and k >= 15:
                exit_price = c

            if exit_price is not None:
                raw = ((exit_price - entry_price) / (entry_price + eps)) if s == 1 else ((entry_price - exit_price) / (entry_price + eps))
                pnl = (raw - config.commission_rate * 2 - config.slippage_rate) * 100 * position_size
                break

        if pnl is None:
            exit_price = float(future_df.iloc[-1]["close"])
            raw = ((exit_price - entry_price) / (entry_price + eps)) if s == 1 else ((entry_price - exit_price) / (entry_price + eps))
            pnl = (raw - config.commission_rate * 2 - config.slippage_rate) * 100 * position_size

        stats["total_pnl"] += pnl
        stats["pnls"].append(pnl)
        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    if stats["entries"] == 0:
        return None

    pnls = np.array(stats["pnls"], dtype=float)
    win_rate = stats["wins"] / stats["entries"]
    avg_win = float(np.mean(pnls[pnls > 0])) if stats["wins"] else 0.0
    avg_loss = float(np.mean(pnls[pnls <= 0])) if stats["losses"] else 0.0
    pl_ratio = abs(avg_win / (avg_loss + 1e-10))
    max_drawdown, _, _ = calculate_max_drawdown(stats["pnls"])

    return {
        "total_pnl": float(stats["total_pnl"]),
        "win_rate": float(win_rate),
        "signal_accuracy": float(stats["wins"] / max(stats["signals"], 1)),
        "pl_ratio": float(pl_ratio),
        "entries": int(stats["entries"]),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": float(max_drawdown),
        "recovery_factor": float(calculate_recovery_factor(stats["total_pnl"], max_drawdown)),
    }


def main() -> int:
    print("\n" + "🚀 " * 30)
    print("企业级量化交易系统 v4.0 - 优化版")
    print("🚀 " * 30 + "\n")

    config = Config()
    df = fetch_data(config)
    if df.empty:
        print("获取数据失败")
        return 1

    result = advanced_backtest(df, config, window_size=20, horizon=5)
    if not result:
        print("回测失败")
        return 1

    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    print(f"总收益: {result['total_pnl']:.2f}%")
    print(f"胜率: {result['win_rate']:.2%}")
    print(f"信号准确率: {result['signal_accuracy']:.2%}")
    print(f"盈亏比: {result['pl_ratio']:.2f}")
    print(f"交易次数: {result['entries']}")
    print(f"最大回撤: {result['max_drawdown']:.2f}%")
    print(f"恢复因子: {result['recovery_factor']:.2f}")
    print("=" * 80 + "\n")

    print("📊 与原始版本的对比：")
    print("  原始版本: 收益11%, 胜率约55%, 回撤-12%")
    print(
        f"  改进版本: 收益{result['total_pnl']:.1f}%, 胜率{result['win_rate']:.1%}, 回撤{result['max_drawdown']:.1f}%"
    )
    print(f"  提升幅度: {(result['total_pnl'] - 11) / 11 * 100:.1f}% ↑")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
