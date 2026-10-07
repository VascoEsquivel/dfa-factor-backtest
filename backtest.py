"""
Backtest the factor-tilted portfolio against the S&P 500.

At each month-end t the portfolio holds the top quintile of composite_score
(composite_quintile = 1, built in sql/factor_queries.sql), equal-weighted, and
earns those stocks' total returns from t to t+1. Ranks use only information
available at t; the forward return is attached afterwards with a one-month
shift, so a ranking never sees the return it is graded on.

Compared against:
  * S&P 500 total return (^SP500TR): the apples-to-apples benchmark, since the
    portfolio's adj_close returns include dividends (falls back to ^GSPC)
  * S&P 500 price index (^GSPC): the headline index, dividends excluded
  * the equal-weighted universe: all ~100 stocks, no factor tilt; separates
    "what factor selection added" from "what equal weighting did"
  * the equal-weight S&P 500 ETF (RSP), whose membership is point-in-time:
    the gap between it and the equal-weighted universe estimates how much
    survivorship bias the fixed universe carries

Also reports single-factor portfolios (attribution), turnover, and, if the Ken
French data library is reachable, a Fama-French 3-factor + momentum regression
showing which factor exposures the portfolio actually carried.

Usage (venv active):
    python backtest.py               # report + output/ files
    python backtest.py --docs        # also refresh the README chart in docs/
    python backtest.py --months 12   # only the last 12 rebalances
"""

import argparse
import io
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # render to file; no display needed
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import requests                            # noqa: E402

from db import ROOT, get_engine            # noqa: E402

OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"
CHART_NAME = "backtest_results.png"

BENCH_TR, BENCH_PRICE, BENCH_EW, RISK_FREE = "^SP500TR", "^GSPC", "RSP", "^IRX"
MIN_MOMENTUM_COVERAGE = 0.5               # a month is rebalanced once momentum covers half the names
COST_BPS = 10                             # illustrative one-way trading cost for the cost estimate

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FILES = {"F-F_Research_Data_Factors_CSV.zip": ["Mkt-RF", "SMB", "HML", "RF"],
            "F-F_Momentum_Factor_CSV.zip": ["Mom"]}

# Chart palette: light surface, categorical slots 1-3, recessive chrome.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES_COLORS = {"Factor portfolio": "#2a78d6", "S&P 500": "#eb6834", "Equal-weight universe": "#1baf7a"}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_data(engine):
    scores = pd.read_sql("""
        SELECT ticker, month_end, momentum_12m, size_market_cap, value_pe,
               momentum_z, size_z, value_z, composite_score, factors_used,
               composite_quintile, fundamentals_as_of
        FROM factor_scores
        ORDER BY month_end, ticker""", engine, parse_dates=["month_end", "fundamentals_as_of"])
    monthly = pd.read_sql("""
        SELECT mp.ticker, mp.month_end, mp.price_date, mp.close, mp.adj_close, t.kind
        FROM monthly_prices mp
        JOIN tickers t ON t.ticker = mp.ticker""", engine, parse_dates=["month_end", "price_date"])
    names = pd.read_sql("SELECT ticker, name, sector FROM tickers", engine).set_index("ticker")
    if scores.empty:
        sys.exit("factor_scores is empty -- run python fetch_data.py first.")
    return scores, monthly, names


def forward_returns(monthly: pd.DataFrame):
    """Month-end-indexed frames. Row t holds the return over (t, t+1]: it is
    realized AFTER the ranking date t, which is what makes the test out-of-sample."""
    stocks = (monthly[monthly["kind"] == "stock"]
              .pivot(index="month_end", columns="ticker", values="adj_close").sort_index())
    months = stocks.index
    steps = pd.PeriodIndex(months, freq="M").asi8
    if len(steps) > 1 and not (np.diff(steps) == 1).all():
        sys.exit("monthly_prices has a gap in its month-ends -- re-run fetch_data.py.")
    stock_fwd = stocks.shift(-1) / stocks - 1

    # monthly_prices takes each series' last close in the month; for a benchmark
    # that is a day or more before the market's month-end, say so.
    if "price_date" in monthly:
        idx = monthly[monthly["kind"] != "stock"]
        stale = idx[idx["price_date"] != idx["month_end"]]
        if not stale.empty:
            print(f"warning: {len(stale)} benchmark/T-bill month-end levels are from an earlier day "
                  f"({', '.join(sorted(stale['ticker'].unique()))}); their monthly returns are slightly off")

    # adj_close: RSP's dividends count; for the indexes and ^IRX it equals close.
    levels = (monthly[monthly["kind"] != "stock"]
              .pivot(index="month_end", columns="ticker", values="adj_close").reindex(months))
    bench_fwd = pd.DataFrame(index=months)
    for sym in (BENCH_TR, BENCH_PRICE, BENCH_EW):
        if sym in levels:
            bench_fwd[sym] = levels[sym].shift(-1) / levels[sym] - 1
    if RISK_FREE in levels and levels[RISK_FREE].notna().any():
        # ^IRX is the 13-week T-bill yield in percent; the yield at t is what a
        # bill bought at t earns over the next month (approximately yield / 12).
        rf = levels[RISK_FREE].ffill() / 100 / 12
    else:
        print("warning: no T-bill data (^IRX); Sharpe ratios use a 0% risk-free rate")
        rf = pd.Series(0.0, index=months)
    return stock_fwd, bench_fwd, rf.rename("rf")


def next_month_end(t) -> pd.Timestamp:
    """Calendar end of the month after t. (t + MonthEnd(1) would be wrong when t
    is a last trading day before the calendar end, e.g. Fri 30 May -> 31 May.)"""
    return (pd.Timestamp(t).to_period("M") + 1).to_timestamp(how="end").normalize()


def top_quintile(scores: pd.Series) -> list[str]:
    """Top ceil(n/5) names by score, ties broken by ticker. Matches NTILE(5)
    ORDER BY score DESC, ticker in factor_queries.sql (NTILE gives the leftover
    rows to the first buckets, so the top bucket holds ceil(n/5))."""
    s = scores.dropna()
    k = math.ceil(len(s) / 5)
    return sorted(s.index, key=lambda t: (-s[t], t))[:k]


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
def choose_benchmark(dates: list, bench_fwd: pd.DataFrame) -> str:
    """^SP500TR if it has a return for every rebalance month, else ^GSPC.

    Decided on the rebalance months themselves, so a hole in the total-return
    series switches the benchmark (with a warning) instead of silently deleting
    months, and a hole outside the test window changes nothing.
    """
    if BENCH_TR in bench_fwd and bench_fwd.loc[dates, BENCH_TR].notna().all():
        return BENCH_TR
    if BENCH_PRICE not in bench_fwd or bench_fwd.loc[dates, BENCH_PRICE].isna().any():
        sys.exit("no complete S&P 500 series (^SP500TR or ^GSPC) -- re-run fetch_data.py.")
    print("warning: ^SP500TR is missing or incomplete; benchmarking against the price index "
          "^GSPC instead (no dividends, so the benchmark is understated by ~1.4%/yr)")
    return BENCH_PRICE


def performance_columns(bench_sym: str, has_rsp: bool) -> tuple[str, dict]:
    """(benchmark label, {table label: returns column}) for the performance table."""
    bench_label = "S&P 500 TR" if bench_sym == BENCH_TR else "S&P 500 price"
    columns = {"Factor portfolio": "Factor portfolio", bench_label: "Benchmark"}
    if bench_sym == BENCH_TR:                       # otherwise ^GSPC already IS the benchmark column
        columns["S&P 500 price"] = "S&P 500 price (^GSPC)"
    columns["EW universe"] = "Equal-weight universe"
    if has_rsp:
        columns["S&P 500 EW (RSP)"] = "S&P 500 equal weight (RSP)"
    return bench_label, columns


def rebalance_dates(scores: pd.DataFrame, stock_fwd: pd.DataFrame, months: int | None) -> list[pd.Timestamp]:
    """Month-ends with live momentum (>= half the names) and a next month. They
    must be consecutive: compounding, annualization and turnover all assume one
    month per step."""
    scored = scores[scores["composite_score"].notna()]
    coverage = scored.groupby("month_end")["momentum_z"].apply(lambda s: s.notna().mean())
    dates = [t for t in stock_fwd.index[:-1]                       # the last month has no t+1
             if coverage.get(t, 0) >= MIN_MOMENTUM_COVERAGE]
    if months:
        dates = dates[-months:]
    if len(dates) < 2:
        sys.exit("fewer than 2 rebalance months with live momentum -- fetch a longer window "
                 "(python fetch_data.py --backtest-months 24).")
    steps = pd.PeriodIndex(dates, freq="M").asi8
    if not (np.diff(steps) == 1).all():
        gaps = [f"{a:%Y-%m}->{b:%Y-%m}" for a, b, d in zip(dates, dates[1:], np.diff(steps)) if d != 1]
        sys.exit(f"rebalance months are not consecutive ({', '.join(gaps)}) -- re-run fetch_data.py.")
    return dates


def run_backtest(scores, stock_fwd, dates):
    """Monthly returns of the composite portfolio, the single-factor portfolios
    and the equal-weighted universe, plus composite holdings per date."""
    by_month = {t: g.set_index("ticker") for t, g in scores.groupby("month_end")}
    rows, holdings, missing = [], {}, 0
    for t in dates:
        g = by_month[t]
        names = top_quintile(g["composite_score"])
        sql_names = set(g.index[g["composite_quintile"] == 1])
        if set(names) != sql_names:
            raise AssertionError(f"{t:%Y-%m}: pandas top quintile != SQL NTILE quintile 1")
        fwd = stock_fwd.loc[t]
        missing += int(fwd.reindex(names).isna().sum())
        row = {"month_end": t,
               "Factor portfolio": fwd.reindex(names).mean(),
               "Equal-weight universe": fwd.reindex(g.index[g["composite_score"].notna()]).mean(),
               "n_holdings": len(names)}
        for col, label in (("momentum_z", "Momentum only"), ("size_z", "Size only"), ("value_z", "Value only")):
            row[label] = fwd.reindex(top_quintile(g[col])).mean()
        rows.append(row)
        holdings[t] = names
    if missing:
        print(f"warning: {missing} holding-months had no next-month price and were left out of the average")
    return pd.DataFrame(rows).set_index("month_end"), holdings


def turnover(holdings: dict, stock_fwd: pd.DataFrame) -> pd.Series:
    """One-way turnover at each rebalance after the first: half the absolute
    change between the new equal weights and last month's weights after they
    drifted with returns. (The initial buy is excluded.)"""
    out, prev_t, prev_w = {}, None, None
    for t, names in holdings.items():
        w = pd.Series(1.0 / len(names), index=names)
        if prev_w is not None:
            r = stock_fwd.loc[prev_t].reindex(prev_w.index).fillna(0.0)
            drifted = prev_w * (1 + r)
            drifted /= drifted.sum()
            idx = drifted.index.union(w.index)
            out[t] = 0.5 * (w.reindex(idx, fill_value=0) - drifted.reindex(idx, fill_value=0)).abs().sum()
        prev_t, prev_w = t, w
    return pd.Series(out, dtype=float)


def load_daily(engine, start) -> pd.DataFrame:
    """Daily adj_close (date x ticker) from `start`, forward-filled over the odd
    missing day so every series has a level on every market date."""
    daily = pd.read_sql("SELECT ticker, date, adj_close FROM prices WHERE date >= %(start)s",
                        engine, params={"start": pd.Timestamp(start).date()}, parse_dates=["date"])
    return daily.pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()


def daily_wealth(members: dict, daily: pd.DataFrame, month_ends: pd.Index) -> pd.Series:
    """Growth of $1 day by day: at each rank date t the money is split equally
    across members[t] and left to drift (buy-and-hold) until the next month-end.
    At month-ends this equals compounding the monthly returns; in between it
    shows the drawdowns a month-end series hides."""
    ends = list(month_ends)
    dates = list(members)
    wealth, pieces = 1.0, [pd.Series([1.0], index=[dates[0]])]
    for t in dates:
        nxt = ends[ends.index(t) + 1]
        names = [s for s in members[t] if s in daily.columns]
        base = daily.loc[t, names]
        names = list(base.index[base.notna() & (base > 0)])
        window = daily.loc[(daily.index > t) & (daily.index <= nxt), names]
        path = wealth * window.div(base[names]).mean(axis=1)
        pieces.append(path)
        wealth = float(path.iloc[-1])
    return pd.concat(pieces)


def index_wealth(level: pd.Series, first, last) -> pd.Series:
    """Growth of $1 for a total-return level series between two dates."""
    window = level.loc[(level.index >= first) & (level.index <= last)]
    return window / window.iloc[0]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the MONTH-END growth-of-$1 path, starting
    from the initial $1 (so a loss in the very first month counts). Intra-month
    lows are invisible to it; see path_drawdown for the daily figure."""
    wealth = np.concatenate([[1.0], np.cumprod(1 + returns.to_numpy(dtype=float))])
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def path_drawdown(wealth: pd.Series) -> float:
    """Largest peak-to-trough loss of a (daily) wealth path."""
    return float((wealth / wealth.cummax() - 1).min())


def performance(r: pd.Series, rf: pd.Series) -> dict:
    r = r.astype(float)
    n = len(r)
    cumulative = float(np.prod(1 + r) - 1)
    excess = r - rf.reindex(r.index).fillna(0.0)
    sd_excess = excess.std(ddof=1)
    sharpe_m = excess.mean() / sd_excess if sd_excess > 0 else float("nan")
    return {
        "months": n,
        "cumulative_return": cumulative,
        "annualized_return": (1 + cumulative) ** (12 / n) - 1,
        "annualized_volatility": float(r.std(ddof=1) * math.sqrt(12)),
        "sharpe_ratio": float(sharpe_m * math.sqrt(12)),
        # Lo (2002) iid standard error of the annualized Sharpe ratio
        "sharpe_std_error": float(math.sqrt((1 + 0.5 * sharpe_m ** 2) / n) * math.sqrt(12)),
        "max_drawdown": max_drawdown(r),
    }


def relative(r: pd.Series, b: pd.Series) -> dict:
    active = (r - b).astype(float)
    te = active.std(ddof=1)
    return {
        "active_return_annualized": float(active.mean() * 12),
        "tracking_error": float(te * math.sqrt(12)),
        "information_ratio": float(active.mean() / te * math.sqrt(12)) if te > 0 else float("nan"),
        "beta": float(np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1)),
        "months_outperformed": int((active > 0).sum()),
    }


# -----------------------------------------------------------------------------
# Fama-French regression
# -----------------------------------------------------------------------------
def parse_ff_csv(text: str, columns: list[str]) -> pd.DataFrame:
    """The monthly block of a Ken French CSV: the first header row (starts with
    a comma) and the YYYYMM rows under it, in decimals instead of percent."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(","))
    header = [h.strip() for h in lines[start].split(",")[1:]]
    data = []
    for line in lines[start + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if not (parts[0].isdigit() and len(parts[0]) == 6):
            break                                       # end of the monthly block
        data.append([parts[0], *map(float, parts[1:])])
    df = pd.DataFrame(data, columns=["yyyymm", *header])
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("yyyymm"), format="%Y%m"), freq="M")
    return df[columns] / 100


def fama_french_factors() -> pd.DataFrame | None:
    frames = []
    try:
        for name, cols in FF_FILES.items():
            resp = requests.get(FF_BASE + name, timeout=30,
                                headers={"User-Agent": "dfa-factor-backtest/1.0"})
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                frames.append(parse_ff_csv(z.read(z.namelist()[0]).decode("latin-1"), cols))
    except Exception as exc:                            # offline, site change ...: skip the section
        print(f"note: Fama-French factors unavailable ({exc}); skipping the regression")
        return None
    return pd.concat(frames, axis=1, join="inner")


def ols(y: np.ndarray, X: np.ndarray):
    """OLS with an intercept already in X. Returns coefficients, t-stats, R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, beta / se, r2


def factor_regressions(returns: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame | None:
    """Regress each return series on Mkt-RF, SMB, HML and Mom. Returns are
    keyed to the month they were EARNED in (the month after the ranking date)."""
    earned = returns.copy()
    earned.index = pd.PeriodIndex(earned.index, freq="M") + 1
    data = earned.join(ff, how="inner")
    if len(data) < 12:
        print(f"note: only {len(data)} months overlap the Fama-French data; skipping the regression")
        return None
    X = np.column_stack([np.ones(len(data)), data[["Mkt-RF", "SMB", "HML", "Mom"]].to_numpy()])
    rows = {}
    targets = {"Factor portfolio": data["Factor portfolio"] - data["RF"],
               "Equal-weight universe": data["Equal-weight universe"] - data["RF"],
               "Factor - EW universe": data["Factor portfolio"] - data["Equal-weight universe"]}
    for label, y in targets.items():
        b, t, r2 = ols(y.to_numpy(), X)
        rows[label] = {"alpha_annualized": b[0] * 12, "alpha_t": t[0],
                       "mkt": b[1], "mkt_t": t[1], "smb": b[2], "smb_t": t[2],
                       "hml": b[3], "hml_t": t[3], "mom": b[4], "mom_t": t[4],
                       "r2": r2, "months": len(data),
                       "first_month": str(data.index[0]), "last_month": str(data.index[-1])}
    return pd.DataFrame(rows).T


# -----------------------------------------------------------------------------
# Chart
# -----------------------------------------------------------------------------
def plot(returns: pd.DataFrame, bench_label: str, path: Path, paths: dict | None = None) -> None:
    """Growth of $1 (top) and drawdown from peak (bottom), starting at $1 on the
    first rebalance date. Uses the daily paths when given (so intra-month
    sell-offs show), otherwise the month-end path."""
    first = returns.index[0]
    month_dates = [first] + [next_month_end(d) for d in returns.index]
    columns = {"Factor portfolio": "Factor portfolio", "S&P 500": "Benchmark",
               "Equal-weight universe": "Equal-weight universe"}
    series = {}
    for label, col in columns.items():
        if paths and col in paths:
            series[label] = paths[col]
        else:
            wealth = np.concatenate([[1.0], np.cumprod(1 + returns[col].to_numpy(dtype=float))])
            series[label] = pd.Series(wealth, index=month_dates)
    last_x = max(s.index[-1] for s in series.values())

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    end_values = {}
    for label, wealth in series.items():
        drawdown = wealth / wealth.cummax() - 1
        lw = 1.4 if label == "Equal-weight universe" else 1.9
        display = ((f"S&P 500 total return ({bench_label})" if bench_label == BENCH_TR
                    else f"S&P 500 ({bench_label})") if label == "S&P 500" else label)
        ax.plot(wealth.index, wealth.to_numpy(), color=SERIES_COLORS[label], lw=lw,
                solid_joinstyle="round", label=display)
        ax_dd.plot(drawdown.index, drawdown.to_numpy() * 100, color=SERIES_COLORS[label], lw=lw * 0.7,
                   solid_joinstyle="round")
        ax.plot(wealth.index[-1], wealth.iloc[-1], "o", ms=5, color=SERIES_COLORS[label],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        end_values[label] = float(wealth.iloc[-1])

    # Direct end labels, skipped when two would collide (the legend still names every line).
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed = []
    for label, v in sorted(end_values.items(), key=lambda kv: -kv[1]):
        if all(abs(v - p) > 0.045 * y_range for p in placed):
            ax.annotate(f"${v:,.2f}", (last_x, v), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=INK, fontweight="semibold")
            placed.append(v)
    dates = month_dates

    ax.axhline(1.0, color=AXIS, lw=0.8, zorder=0)
    ax_dd.axhline(0.0, color=AXIS, lw=0.8, zorder=0)
    for a in (ax, ax_dd):
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color=GRID, lw=0.7)
        a.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(AXIS)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax_dd.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=True))
    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Growth of $1", color=INK_2)
    ax_dd.set_ylabel("Drawdown" + (" (daily)" if paths else ""), color=INK_2)
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(dates[0] - pd.Timedelta(days=10), dates[-1] + pd.Timedelta(days=55))
    leg = ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, handlelength=1.6)
    for line in leg.get_lines():
        line.set_linewidth(2.2)
    ax.set_title(f"Factor-tilted portfolio vs. the S&P 500  ·  top quintile of momentum + size + value, "
                 f"equal-weighted, rebalanced monthly\n{len(returns)} monthly returns, "
                 f"{dates[1]:%b %Y} – {dates[-1]:%b %Y}  ·  growth of $1 invested {dates[0]:%d %b %Y}",
                 loc="left", color=INK, fontsize=11, fontweight="semibold", linespacing=1.5)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def pct(x, digits=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f}%"


def sanity_checks(returns: pd.DataFrame, perf: dict, paths: dict | None = None) -> list[str]:
    problems = []
    if np.allclose(returns["Factor portfolio"], returns["Benchmark"]):
        problems.append("factor portfolio returns are identical to the benchmark")
    for label, p in perf.items():
        for key in ("sharpe_ratio", "max_drawdown", "annualized_return", "annualized_volatility"):
            if not math.isfinite(p[key]):
                problems.append(f"{label}: {key} is not finite")
    if returns[["Factor portfolio", "Benchmark"]].isna().any().any():
        problems.append("missing monthly returns in the backtest window")
    # The daily path is built independently from daily prices; it must land
    # exactly where compounding the monthly returns does.
    for col, wealth in (paths or {}).items():
        if col in returns:
            monthly_end = float(np.prod(1 + returns[col].astype(float)))
            if not math.isclose(float(wealth.iloc[-1]), monthly_end, rel_tol=1e-6):
                problems.append(f"{col}: daily path ends at {wealth.iloc[-1]:.6f}, "
                                f"monthly compounding at {monthly_end:.6f}")
    return problems


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--months", type=int, default=None, help="only the last N rebalances")
    p.add_argument("--no-ff", action="store_true", help="skip the Fama-French regression")
    p.add_argument("--docs", action="store_true", help=f"also write the chart to docs/{CHART_NAME}")
    args = p.parse_args(argv)

    engine = get_engine()
    scores, monthly, names = load_data(engine)
    stock_fwd, bench_fwd, rf = forward_returns(monthly)
    dates = rebalance_dates(scores, stock_fwd, args.months)
    bench_sym = choose_benchmark(dates, bench_fwd)
    returns, holdings = run_backtest(scores, stock_fwd, dates)

    returns["Benchmark"] = bench_fwd.loc[dates, bench_sym]
    returns["S&P 500 price (^GSPC)"] = bench_fwd.loc[dates, BENCH_PRICE]
    has_rsp = BENCH_EW in bench_fwd and bench_fwd.loc[dates, BENCH_EW].notna().all()
    if has_rsp:
        returns["S&P 500 equal weight (RSP)"] = bench_fwd.loc[dates, BENCH_EW]
    returns["rf"] = rf.reindex(dates).fillna(0.0)

    # Daily growth-of-$1 paths: the drawdowns month-end data can't see.
    daily = load_daily(engine, dates[0])
    last = stock_fwd.index[list(stock_fwd.index).index(dates[-1]) + 1]
    by_month = {t: g for t, g in scores.groupby("month_end")}
    paths = {"Factor portfolio": daily_wealth(holdings, daily, stock_fwd.index),
             "Equal-weight universe": daily_wealth(
                 {t: list(by_month[t].loc[by_month[t]["composite_score"].notna(), "ticker"]) for t in dates},
                 daily, stock_fwd.index)}
    for col, sym in (("Benchmark", bench_sym), ("S&P 500 price (^GSPC)", BENCH_PRICE),
                     ("S&P 500 equal weight (RSP)", BENCH_EW)):
        if col in returns and sym in daily:
            paths[col] = index_wealth(daily[sym], dates[0], last)

    bench_label, columns = performance_columns(bench_sym, has_rsp)
    perf = {label: performance(returns[col], returns["rf"]) for label, col in columns.items()}
    for label, col in columns.items():
        perf[label]["max_drawdown_daily"] = path_drawdown(paths[col]) if col in paths else float("nan")
    attribution = {label: performance(returns[label], returns["rf"])
                   for label in ("Momentum only", "Size only", "Value only", "Factor portfolio")}
    rel = relative(returns["Factor portfolio"], returns["Benchmark"])
    to = turnover(holdings, stock_fwd)
    # Checked before anything is written, so a bad run can't overwrite good results.
    problems = sanity_checks(returns, perf, paths)

    # ---- report -------------------------------------------------------------------
    t0, t1 = dates[0], dates[-1]
    in_window = scores[scores["month_end"].isin(dates)]
    lookahead = (in_window["fundamentals_as_of"] > in_window["month_end"]).mean()
    coverage = {f: in_window[f + "_z"].notna().mean() for f in ("momentum", "size", "value")}
    n_univ = int(in_window.groupby("month_end")["composite_score"].count().median())
    print("=" * 78)
    print("FACTOR-TILTED PORTFOLIO BACKTEST  (momentum + size + value, top quintile, equal weight)")
    print("=" * 78)
    print(f"rebalances : {len(dates)} month-ends, {t0:%Y-%m-%d} .. {t1:%Y-%m-%d} "
          f"(returns earned {next_month_end(t0):%b %Y} .. {next_month_end(t1):%b %Y})")
    print(f"universe   : ~{n_univ} scored stocks per month, holding {int(returns['n_holdings'].median())} "
          f"(top quintile); all 3 factors on {in_window['factors_used'].eq(3).mean():.0%} of stock-months")
    print(f"coverage   : momentum {coverage['momentum']:.0%}, size {coverage['size']:.0%}, "
          f"value {coverage['value']:.0%} of stock-months (value is blank when earnings <= 0)")
    print(f"look-ahead : {lookahead:.0%} of stock-months use fundamentals dated after their month-end"
          + ("" if lookahead == 0 else "  <-- NOT point-in-time (bias-audit mode?)"))
    no_fund = in_window[in_window["fundamentals_as_of"].isna()]
    if not no_fund.empty:
        gaps = no_fund.groupby("ticker")["month_end"].agg(["min", "max", "count"])
        print("no fundamentals: " + "; ".join(f"{t} {r['count']} month(s) {r['min']:%Y-%m}..{r['max']:%Y-%m}"
                                              for t, r in gaps.iterrows())
              + "  (momentum-only those months: no point-in-time annual report within 13 months)")
    print(f"risk-free  : {'13-week T-bill (^IRX)' if returns['rf'].abs().sum() > 0 else 'none (0%)'}, "
          f"avg {returns['rf'].mean() * 12:.2%}/yr\n")

    print("1) Performance")
    print(f"{'':24}" + "".join(f"{label:>18}" for label in perf))
    for key, fmt in (("cumulative_return", pct), ("annualized_return", pct),
                     ("annualized_volatility", lambda x: f"{x * 100:.1f}%"),
                     ("sharpe_ratio", lambda x: f"{x:.2f}"), ("sharpe_std_error", lambda x: f"+/-{x:.2f}"),
                     ("max_drawdown_daily", pct), ("max_drawdown", pct)):
        row_label = {"max_drawdown_daily": "max drawdown (daily)",
                     "max_drawdown": "max drawdown (month-end)"}.get(key, key.replace("_", " "))
        print(f"{row_label:24}" + "".join(f"{fmt(p.get(key, float('nan'))):>18}" for p in perf.values()))

    geo_gap = perf["Factor portfolio"]["annualized_return"] - perf[bench_label]["annualized_return"]
    print(f"\n2) Factor portfolio vs. {bench_label} ({bench_sym})")
    print(f"   active return {pct(rel['active_return_annualized'])}/yr (mean monthly difference x 12; "
          f"the gap in annualized returns above is {geo_gap * 100:+.1f} points)")
    print(f"   tracking error {rel['tracking_error']:.1%}, information ratio {rel['information_ratio']:.2f}, "
          f"beta {rel['beta']:.2f}, beat the index in {rel['months_outperformed']} of {len(dates)} months")

    print("\n3) Attribution: top quintile of each factor alone (equal weight)")
    print(f"   {'':18}{'ann. return':>12}{'volatility':>12}{'Sharpe':>8}{'max DD (m/e)':>14}")
    for label, a in attribution.items():
        print(f"   {label:18}{pct(a['annualized_return']):>12}{a['annualized_volatility']:>12.1%}"
              f"{a['sharpe_ratio']:>8.2f}{pct(a['max_drawdown']):>14}")

    avg_to = float(to.mean()) if len(to) else float("nan")
    drag = avg_to * 2 * COST_BPS / 1e4 * 12
    print(f"\n4) Turnover: {avg_to:.0%} of the portfolio replaced per month (one-way, avg); at "
          f"{COST_BPS} bps per trade that is roughly {drag:.2%}/yr of costs (not deducted above)")

    reg = None
    if not args.no_ff:
        ff = fama_french_factors()
        if ff is not None:
            reg = factor_regressions(returns, ff)
    if reg is not None:
        r0 = reg.iloc[0]
        print(f"\n5) Fama-French 3-factor + momentum regression, {int(r0['months'])} months "
              f"({r0['first_month']} .. {r0['last_month']}), t-stats in parentheses")
        print(f"   {'':24}{'alpha/yr':>15}{'Mkt-RF':>14}{'SMB':>14}{'HML':>14}{'Mom':>14}{'R2':>6}")
        for label, row in reg.iterrows():
            cells = [f"{row['alpha_annualized'] * 100:+.1f}% ({row['alpha_t']:+.1f})"]
            cells += [f"{row[k]:+.2f} ({row[k + '_t']:+.1f})" for k in ("mkt", "smb", "hml", "mom")]
            print(f"   {label:24}" + "".join(f"{c:>15}" if i == 0 else f"{c:>14}" for i, c in enumerate(cells))
                  + f"{row['r2']:>6.2f}")

    latest = scores["month_end"].max()
    current = scores[(scores["month_end"] == latest) & (scores["composite_quintile"] == 1)]
    current = current.sort_values("composite_score", ascending=False)
    print(f"\n6) Current top quintile (ranked {latest:%Y-%m-%d}, i.e. next month's holdings; not in the backtest)")
    for _, row in current.iterrows():
        z = lambda v: "  -  " if pd.isna(v) else f"{v:+.2f}"     # noqa: E731
        print(f"   {row['ticker']:6} {str(names['name'].get(row['ticker']))[:28]:28} "
              f"composite {row['composite_score']:+.2f}  mom {z(row['momentum_z'])}  "
              f"size {z(row['size_z'])}  value {z(row['value_z'])}")

    if problems:
        print("\nSANITY CHECK FAILED (no output files were written):\n  " + "\n  ".join(problems))
        sys.exit(1)

    # ---- files ----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    chart = OUTPUT_DIR / CHART_NAME
    plot(returns, bench_sym, chart, paths)
    if args.docs:
        DOCS_DIR.mkdir(exist_ok=True)
        shutil.copyfile(chart, DOCS_DIR / CHART_NAME)
    out = returns.copy()
    out.insert(0, "return_month", [f"{next_month_end(t):%Y-%m}" for t in out.index])
    out.to_csv(OUTPUT_DIR / "monthly_returns.csv", float_format="%.6f")
    pd.DataFrame([{"rank_date": t.date(), "ticker": s, "name": names["name"].get(s),
                   "sector": names["sector"].get(s)} for t, h in holdings.items() for s in h]
                 ).to_csv(OUTPUT_DIR / "holdings.csv", index=False)
    summary = {"benchmark": bench_sym, "first_rank_date": str(t0.date()), "last_rank_date": str(t1.date()),
               "rebalances": len(dates), "performance": perf, "attribution": attribution,
               "relative_to_benchmark": rel, "avg_monthly_turnover": avg_to,
               "lookahead_share": float(lookahead),
               "fama_french": None if reg is None else json.loads(reg.to_json(orient="index"))}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {chart.relative_to(ROOT)}, output/monthly_returns.csv, output/holdings.csv, "
          f"output/summary.json" + (f", docs/{CHART_NAME}" if args.docs else ""))
    print("sanity checks passed: returns differ from the benchmark; Sharpe, drawdown, "
          "returns and volatility are all finite")


if __name__ == "__main__":
    main()
