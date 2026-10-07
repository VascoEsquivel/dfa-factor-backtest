"""
Pull the data and load it into Postgres, then rebuild factor_scores.

  1. universe      ~100 S&P 500 constituents scraped from Wikipedia (a
                   hardcoded 100-stock fallback list is used if that fails),
                   sampled in proportion to GICS sector weights. By default
                   only stocks that were ALREADY in the index at the first
                   rebalance date are eligible (Wikipedia's "Date added").
  2. prices        daily OHLCV from yfinance for those stocks, plus the
                   S&P 500 (^GSPC), S&P 500 total return (^SP500TR), the
                   equal-weight S&P 500 ETF (RSP) and the 13-week T-bill
                   yield (^IRX, the risk-free rate)
  3. fundamentals  point-in-time annual reports (~4 fiscal years of net income,
                   EPS, diluted shares, book equity from yfinance's financial
                   statements, each dated fiscal year-end + 4 months) and
                   today's quote-page snapshot (market cap, P/E, P/B, EPS)
  4. factors       runs sql/factor_queries.sql

The price window is the backtest plus a 12-month momentum warm-up, ending at
the last COMPLETED month, so no partial month ever enters the data. With the
defaults (24-month backtest) that is ~3 years of daily data.

A ticker that fails to download is retried, then dropped with a warning; the
pipeline only stops if the S&P 500 itself can't be fetched or fewer than
MIN_UNIVERSE stocks survive.

Usage (from the repo root, venv active):
    python fetch_data.py
    python fetch_data.py --n-tickers 150 --backtest-months 36
    python fetch_data.py --end 2026-06-30            # stop at an earlier month-end
    python fetch_data.py --tickers AAPL,MSFT,JPM,...  # explicit universe
    python fetch_data.py --factors-only               # rebuild factor_scores, no download

Bias audit (reproduces the naive approaches the defaults avoid):
    python fetch_data.py --universe current           # today's members, incl. later additions
    python fetch_data.py --factors-only --snapshot-fundamentals   # today's EPS applied to the past
"""

import argparse
import hashlib
import io
import logging
import math
import re
import statistics
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from psycopg2.extras import execute_values

from db import SQL_DIR, connect, run_sql_file

log = logging.getLogger("fetch_data")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HTTP_HEADERS = {"User-Agent": "dfa-factor-backtest/1.0 (academic research project; python-requests)"}

BENCHMARK = "^GSPC"                       # required: S&P 500 price index
OPTIONAL_SERIES = {                       # nice to have: degrade gracefully without them
    "^SP500TR": ("S&P 500 Total Return", "index"),
    "RSP": ("Invesco S&P 500 Equal Weight ETF", "index"),
    "^IRX": ("13-Week Treasury Bill yield (%)", "rate"),
}
MIN_UNIVERSE = 25                         # below this a top quintile is ~5 names: stop
MOMENTUM_WARMUP_MONTHS = 12               # 12-1 momentum needs month-end t-12
ANNUAL_REPORT_LAG_MONTHS = 4              # a fiscal year's numbers are used from FYE + 4 months

# Used only if the Wikipedia scrape fails. 100 long-standing S&P 500 members,
# sector counts roughly proportional to the index (GICS sectors). Being
# well-known names they skew larger than a random draw would.
FALLBACK_UNIVERSE = [
    # Information Technology (15)
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft", "Information Technology"),
    ("NVDA", "Nvidia", "Information Technology"),
    ("AVGO", "Broadcom", "Information Technology"),
    ("ORCL", "Oracle Corporation", "Information Technology"),
    ("CSCO", "Cisco", "Information Technology"),
    ("ADBE", "Adobe Inc.", "Information Technology"),
    ("CRM", "Salesforce", "Information Technology"),
    ("ACN", "Accenture", "Information Technology"),
    ("TXN", "Texas Instruments", "Information Technology"),
    ("QCOM", "Qualcomm", "Information Technology"),
    ("IBM", "IBM", "Information Technology"),
    ("AMD", "Advanced Micro Devices", "Information Technology"),
    ("INTU", "Intuit", "Information Technology"),
    ("HPQ", "HP Inc.", "Information Technology"),
    # Industrials (16)
    ("GE", "GE Aerospace", "Industrials"),
    ("CAT", "Caterpillar Inc.", "Industrials"),
    ("HON", "Honeywell", "Industrials"),
    ("UNP", "Union Pacific Corporation", "Industrials"),
    ("RTX", "RTX Corporation", "Industrials"),
    ("LMT", "Lockheed Martin", "Industrials"),
    ("DE", "Deere & Company", "Industrials"),
    ("UPS", "United Parcel Service", "Industrials"),
    ("MMM", "3M", "Industrials"),
    ("GD", "General Dynamics", "Industrials"),
    ("CSX", "CSX Corporation", "Industrials"),
    ("PH", "Parker Hannifin", "Industrials"),
    ("EMR", "Emerson Electric", "Industrials"),
    ("ITW", "Illinois Tool Works", "Industrials"),
    ("FDX", "FedEx", "Industrials"),
    ("WM", "Waste Management", "Industrials"),
    # Financials (15)
    ("JPM", "JPMorgan Chase", "Financials"),
    ("BAC", "Bank of America", "Financials"),
    ("WFC", "Wells Fargo", "Financials"),
    ("GS", "Goldman Sachs", "Financials"),
    ("MS", "Morgan Stanley", "Financials"),
    ("C", "Citigroup", "Financials"),
    ("BLK", "BlackRock", "Financials"),
    ("SCHW", "Charles Schwab Corporation", "Financials"),
    ("AXP", "American Express", "Financials"),
    ("SPGI", "S&P Global", "Financials"),
    ("CB", "Chubb Limited", "Financials"),
    ("PGR", "Progressive Corporation", "Financials"),
    ("USB", "U.S. Bancorp", "Financials"),
    ("PNC", "PNC Financial Services", "Financials"),
    ("BRK-B", "Berkshire Hathaway", "Financials"),
    # Health Care (12)
    ("UNH", "UnitedHealth Group", "Health Care"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("LLY", "Eli Lilly and Company", "Health Care"),
    ("PFE", "Pfizer", "Health Care"),
    ("MRK", "Merck & Co.", "Health Care"),
    ("ABBV", "AbbVie", "Health Care"),
    ("TMO", "Thermo Fisher Scientific", "Health Care"),
    ("ABT", "Abbott Laboratories", "Health Care"),
    ("DHR", "Danaher Corporation", "Health Care"),
    ("AMGN", "Amgen", "Health Care"),
    ("BMY", "Bristol Myers Squibb", "Health Care"),
    ("MDT", "Medtronic", "Health Care"),
    # Consumer Discretionary (9)
    ("AMZN", "Amazon", "Consumer Discretionary"),
    ("TSLA", "Tesla, Inc.", "Consumer Discretionary"),
    ("HD", "Home Depot", "Consumer Discretionary"),
    ("MCD", "McDonald's", "Consumer Discretionary"),
    ("NKE", "Nike, Inc.", "Consumer Discretionary"),
    ("LOW", "Lowe's", "Consumer Discretionary"),
    ("SBUX", "Starbucks", "Consumer Discretionary"),
    ("TJX", "TJX Companies", "Consumer Discretionary"),
    ("BKNG", "Booking Holdings", "Consumer Discretionary"),
    # Consumer Staples (7)
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("KO", "Coca-Cola Company", "Consumer Staples"),
    ("PEP", "PepsiCo", "Consumer Staples"),
    ("WMT", "Walmart", "Consumer Staples"),
    ("COST", "Costco", "Consumer Staples"),
    ("CL", "Colgate-Palmolive", "Consumer Staples"),
    ("MO", "Altria", "Consumer Staples"),
    # Utilities (6)
    ("NEE", "NextEra Energy", "Utilities"),
    ("DUK", "Duke Energy", "Utilities"),
    ("SO", "Southern Company", "Utilities"),
    ("D", "Dominion Energy", "Utilities"),
    ("AEP", "American Electric Power", "Utilities"),
    ("EXC", "Exelon", "Utilities"),
    # Real Estate (6)
    ("PLD", "Prologis", "Real Estate"),
    ("AMT", "American Tower", "Real Estate"),
    ("EQIX", "Equinix", "Real Estate"),
    ("SPG", "Simon Property Group", "Real Estate"),
    ("O", "Realty Income", "Real Estate"),
    ("PSA", "Public Storage", "Real Estate"),
    # Materials (5)
    ("LIN", "Linde plc", "Materials"),
    ("SHW", "Sherwin-Williams", "Materials"),
    ("APD", "Air Products", "Materials"),
    ("ECL", "Ecolab", "Materials"),
    ("NEM", "Newmont", "Materials"),
    # Communication Services (5)
    ("GOOGL", "Alphabet Inc. (Class A)", "Communication Services"),
    ("META", "Meta Platforms", "Communication Services"),
    ("NFLX", "Netflix", "Communication Services"),
    ("DIS", "Walt Disney Company", "Communication Services"),
    ("VZ", "Verizon", "Communication Services"),
    # Energy (4)
    ("XOM", "ExxonMobil", "Energy"),
    ("CVX", "Chevron Corporation", "Energy"),
    ("COP", "ConocoPhillips", "Energy"),
    ("EOG", "EOG Resources", "Energy"),
]

PRICE_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
                 "Adj Close": "adj_close", "Volume": "volume"}


# -----------------------------------------------------------------------------
# Universe
# -----------------------------------------------------------------------------
def yahoo_symbol(symbol: str) -> str:
    """Wikipedia writes share classes with a dot (BRK.B); Yahoo uses a dash (BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def parse_inclusion_dates(table: pd.DataFrame) -> pd.Series:
    """The "Date added" column as Timestamps. The start-members filter depends on
    it, so a renamed column (it was "Date first added" until ~2023) or a format
    change must fail the scrape loudly instead of silently disabling the filter."""
    cols = [c for c in table.columns if re.search(r"added", str(c), re.I)]
    if not cols:
        raise ValueError("constituents table has no 'Date added' column")
    raw = table[cols[0]].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    added = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    if added.isna().mean() > 0.05:
        raise ValueError(f"could not parse {added.isna().sum()} of {len(added)} inclusion dates")
    return added


def scrape_sp500(retries: int = 3, timeout: int = 20) -> pd.DataFrame:
    """Current S&P 500 constituents as DataFrame[ticker, name, sector, added].

    Wikipedia rejects the default urllib user agent, so the page is fetched
    with requests and a descriptive User-Agent, then parsed with read_html.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(WIKI_URL, headers=HTTP_HEADERS, timeout=timeout)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
            table = tables[0]
            missing = {"Symbol", "Security", "GICS Sector"} - set(table.columns)
            if missing:
                raise ValueError(f"constituents table is missing columns {sorted(missing)}")
            added = parse_inclusion_dates(table)
            out = pd.DataFrame({
                "ticker": table["Symbol"].map(yahoo_symbol),
                "name": table["Security"].astype(str).str.strip(),
                "sector": table["GICS Sector"].astype(str).str.strip(),
                "added": [d.date() if pd.notna(d) else None for d in added],
            }).drop_duplicates("ticker")
            if len(out) < 400:      # the index has ~503 lines; far fewer means a bad parse
                raise ValueError(f"only {len(out)} rows parsed")
            return out.reset_index(drop=True)
        except Exception as exc:    # network error, HTTP error, page layout change ...
            last_error = exc
            log.warning("Wikipedia scrape attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Wikipedia scrape failed: {last_error}")


def fallback_universe() -> pd.DataFrame:
    """The hardcoded list. Inclusion dates are unknown (None), but every name on
    it has been in the index for many years: main() passes it through the
    start-members filter with allow_unknown=True."""
    out = pd.DataFrame(FALLBACK_UNIVERSE, columns=["ticker", "name", "sector"])
    out["added"] = None
    return out


def start_members(constituents: pd.DataFrame, first_rank: date, allow_unknown: bool = False) -> pd.DataFrame:
    """Constituents that were already in the index on the first rebalance date.

    Today's list includes names added DURING the backtest, often because they
    had just rallied; ranking them before they joined would select on the
    future. Names with an unknown inclusion date are dropped unless
    allow_unknown is set (only for the hardcoded fallback list)."""
    keep = constituents["added"].map(
        lambda d: (allow_unknown if d is None or pd.isna(d) else d <= first_rank))
    return constituents[keep.astype(bool)].reset_index(drop=True)


def _hash_key(seed: int, ticker: str) -> str:
    return hashlib.sha256(f"{seed}:{ticker}".encode()).hexdigest()


def allocate_by_sector(counts: pd.Series, n: int) -> pd.Series:
    """Split n slots across sectors in proportion to their size (largest remainder)."""
    n = min(n, int(counts.sum()))
    exact = counts / counts.sum() * n
    alloc = exact.astype(int)
    remainder = (exact - alloc).sort_values(ascending=False, kind="stable")
    for sector in remainder.index[: n - int(alloc.sum())]:
        alloc[sector] += 1
    return alloc


def stratified_sample(universe: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sector-proportional sample of n tickers.

    Within each sector, tickers are ordered by sha256(seed:ticker) and the first
    k taken. Unlike DataFrame.sample, a constituent change only swaps the one
    name involved instead of reshuffling the whole sample, so reruns are stable.
    """
    if n >= len(universe):
        return universe.sort_values("ticker").reset_index(drop=True)
    alloc = allocate_by_sector(universe["sector"].value_counts().sort_index(), n)
    keyed = universe.assign(_key=universe["ticker"].map(lambda t: _hash_key(seed, t)))
    picks = [grp.sort_values("_key").head(int(alloc[sector]))
             for sector, grp in keyed.groupby("sector") if alloc.get(sector, 0) > 0]
    return (pd.concat(picks).drop(columns="_key")
            .sort_values("ticker").reset_index(drop=True))


# -----------------------------------------------------------------------------
# Dates
# -----------------------------------------------------------------------------
def last_complete_month_end(now_et: datetime) -> date:
    """Calendar last day of the most recent month whose trading is over.

    The current month counts as complete only once its final weekday has
    closed (after 6pm New York time, so the day's bar has settled); otherwise
    the answer is the end of the previous month.
    """
    today = now_et.date()
    month_last_day = (pd.Timestamp(today) + pd.offsets.MonthEnd(0)).date()
    last_weekday = month_last_day
    while last_weekday.weekday() >= 5:
        last_weekday -= timedelta(days=1)
    if today > last_weekday or (today == last_weekday and now_et.hour >= 18):
        return month_last_day
    return today.replace(day=1) - timedelta(days=1)


def first_rank_date(end_month_end: date, backtest_months: int) -> date:
    """Calendar month-end of the first rebalance: backtest_months before the end."""
    period = pd.Timestamp(end_month_end).to_period("M") - backtest_months
    return period.to_timestamp(how="end").date()


def price_window(end_month_end: date, backtest_months: int) -> tuple[date, date]:
    """(start, end_exclusive) covering the warm-up plus the backtest.

    Momentum at the first rebalance month-end t0 needs month-end t0-12, and the
    last rebalance needs one more month to realize its return, so the window
    spans backtest_months + 12 months after the first month-end.
    """
    first_month = (pd.Timestamp(end_month_end).to_period("M")
                   - (backtest_months + MOMENTUM_WARMUP_MONTHS))
    start = first_month.to_timestamp().date()            # 1st day of that month
    return start, end_month_end + timedelta(days=1)


# -----------------------------------------------------------------------------
# Prices
# -----------------------------------------------------------------------------
def tidy_prices(frame: pd.DataFrame, start: date, end_exclusive: date) -> pd.DataFrame:
    """yfinance OHLCV -> DataFrame[open, high, low, close, adj_close, volume]
    indexed by plain dates, with unusable rows dropped."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(PRICE_COLUMNS.values()))
    df = frame.rename(columns=PRICE_COLUMNS)
    df = df[[c for c in PRICE_COLUMNS.values() if c in df.columns]].copy()
    if "close" not in df.columns:
        return pd.DataFrame(columns=list(PRICE_COLUMNS.values()))
    if "adj_close" not in df.columns:
        log.warning("no 'Adj Close' column; using close (dividends ignored)")
        df["adj_close"] = df["close"]
    for col in PRICE_COLUMNS.values():
        if col not in df.columns:
            df[col] = float("nan")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)       # ^IRX comes back in Chicago time, stocks in New York
    df.index = idx.normalize().date
    df = df[(df.index >= start) & (df.index < end_exclusive)]
    df = df[pd.to_numeric(df["close"], errors="coerce") > 0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[list(PRICE_COLUMNS.values())]


def _split_download(data: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """yf.download(group_by='ticker') -> {symbol: frame}, tolerating both the
    MultiIndex layout (several symbols) and the flat one (a single symbol)."""
    out = {}
    if data is None or data.empty:
        return out
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        for sym in symbols:
            if sym in level0:
                out[sym] = data[sym]
    elif len(symbols) == 1:
        out[symbols[0]] = data
    return out


def download_prices(symbols: list[str], start: date, end_exclusive: date,
                    batch_size: int = 50, retries: int = 3) -> tuple[dict, list]:
    """Batch download, then per-symbol retries for anything that came back
    empty. Returns ({symbol: tidy frame}, [symbols that never downloaded])."""
    good: dict[str, pd.DataFrame] = {}
    pending: list[str] = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            data = yf.download(batch, start=start.isoformat(), end=end_exclusive.isoformat(),
                               auto_adjust=False, actions=False, group_by="ticker",
                               threads=True, progress=False, timeout=30)
            frames = _split_download(data, batch)
        except Exception as exc:     # rate limit, network, yfinance parsing change ...
            log.warning("batch download failed (%s); will retry one by one", exc)
            frames = {}
        for sym in batch:
            try:                     # one malformed payload must not sink the whole run
                tidy = tidy_prices(frames.get(sym), start, end_exclusive)
            except Exception as exc:
                log.warning("%s: unusable batch data (%s); will retry on its own", sym, exc)
                tidy = pd.DataFrame()
            if tidy.empty:
                pending.append(sym)
            else:
                good[sym] = tidy

    failed = []
    for sym in pending:
        for attempt in range(1, retries + 1):
            try:
                hist = yf.Ticker(sym).history(start=start.isoformat(), end=end_exclusive.isoformat(),
                                              auto_adjust=False, actions=False, timeout=30)
                tidy = tidy_prices(hist, start, end_exclusive)
                if not tidy.empty:
                    good[sym] = tidy
                    break
            except Exception as exc:
                log.warning("%s: attempt %d/%d failed: %s", sym, attempt, retries, exc)
            if attempt < retries:
                time.sleep(1.5 * attempt)
        else:
            failed.append(sym)
    return good, failed


def flag_suspect_moves(frames: dict[str, pd.DataFrame], threshold: float = 0.4) -> None:
    """Warn about one-day moves above 40%: real ones happen, but so do bad Yahoo prints."""
    for sym, df in frames.items():
        moves = df["adj_close"].astype(float).pct_change().abs()
        big = moves[moves > threshold]
        if not big.empty:
            log.warning("%s: %d daily move(s) > %.0f%%, largest %.0f%% on %s",
                        sym, len(big), threshold * 100, big.max() * 100, big.idxmax())


# -----------------------------------------------------------------------------
# Fundamentals
# -----------------------------------------------------------------------------
def _num(value) -> float | None:
    """Finite float or None. yfinance returns None, 'Infinity', NaN ... for missing data."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _retry(fetch, what: str, retries: int = 3):
    """Call fetch() until it returns something non-empty; None after `retries` tries."""
    for attempt in range(1, retries + 1):
        try:
            result = fetch()
            if result is not None and len(result) and not getattr(result, "empty", False):
                return result
        except Exception as exc:     # rate limit, network, Yahoo layout change ...
            log.warning("%s attempt %d/%d failed: %s", what, attempt, retries, exc)
        if attempt < retries:
            time.sleep(1.5 * attempt)
    return None


def _quote_price(info: dict) -> float | None:
    return next((p for p in (_num(info.get(k)) for k in
                 ("currentPrice", "regularMarketPrice", "previousClose")) if p and p > 0), None)


def reference_shares(info: dict) -> float | None:
    """Today's whole-company share count in this ticker's class units, or None
    when the quote only offers a single-class count (sharesOutstanding is one
    class for GOOG / BRK-B), which would make a misleading yardstick."""
    price, market_cap = _quote_price(info), _num(info.get("marketCap"))
    if market_cap and price:
        return market_cap / price
    implied = _num(info.get("impliedSharesOutstanding"))
    return implied if implied and implied > 0 else None


def parse_snapshot(ticker: str, info: dict, fallback_date: date) -> tuple | None:
    """yfinance .info -> a 'snapshot' fundamentals row, or None if nothing usable.

    shares_out is market_cap / price: the whole company's share count in units
    of this ticker's class (Yahoo's sharesOutstanding counts one class only for
    GOOGL or BRK-B), so shares_out x close rebuilds a whole-company market cap.
    """
    price = _quote_price(info)
    market_cap = _num(info.get("marketCap"))
    shares = reference_shares(info) or _num(info.get("sharesOutstanding"))
    if shares and price and market_cap is None:
        market_cap = shares * price
    trailing_pe = _num(info.get("trailingPE"))
    if trailing_pe is not None and trailing_pe <= 0:
        trailing_pe = None
    eps = _num(info.get("trailingEps"))
    if eps is None and trailing_pe and price:
        eps = price / trailing_pe
    ptb = _num(info.get("priceToBook"))
    if market_cap is None and shares is None and eps is None:
        return None
    stamp = _num(info.get("regularMarketTime"))
    as_of = (pd.Timestamp(stamp, unit="s", tz="UTC").tz_convert("America/New_York").date()
             if stamp else fallback_date)
    return (ticker, as_of, "snapshot", None, market_cap, trailing_pe, ptb, shares, eps, None, None)


INCOME_ROWS = {
    "net_income": ["Net Income Common Stockholders", "Net Income",
                   "Net Income From Continuing Operation Net Minority Interest"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "shares": ["Diluted Average Shares", "Basic Average Shares"],
}
BOOK_ROWS = ["Common Stock Equity", "Stockholders Equity"]


def _first(frame: pd.DataFrame | None, names: list[str], col) -> float | None:
    if frame is None or frame.empty or col not in frame.columns:
        return None
    for name in names:
        if name in frame.index:
            value = _num(frame.at[name, col])
            if value is not None:
                return value
    return None


def parse_annual_reports(ticker: str, income: pd.DataFrame | None, balance: pd.DataFrame | None,
                         closes: pd.Series | None, reference_shares: float | None = None) -> list[tuple]:
    """yfinance annual income statement + balance sheet -> 'annual_report' rows.

    Each fiscal year becomes a row dated fiscal year-end + 4 months, a date by
    which the 10-K was certainly public. yfinance reports EPS and share counts
    split-adjusted and in this ticker's class units (BRK-B in B-share
    equivalents), consistent with the split-adjusted closes. market_cap, P/E
    and P/B are evaluated at that date when a close is available (for
    reference; factor_queries.sql recomputes them at every month-end).

    Yahoo's statements have holes and glitches, so the share count of each
    fiscal year is checked against a yardstick: reference_shares (today's
    count) or, without one, the median of the ticker's own years. A count that
    is missing or off by more than 2x is replaced by, in order of preference:
      1. net income / that year's reported EPS (point-in-time; fixes TRV's
         FY2024 count, reported 100x too high next to a correct EPS),
      2. the nearest plausible fiscal year's count, earlier years first (fills
         CLX FY2025, where Yahoo has net income but no shares or EPS),
      3. the yardstick itself (DLR: every year is 4x off and only today's
         count is right; a small look-ahead from buybacks since, documented).
    EPS is then recomputed from net income, the reliable company-level number.
    """
    if income is None or income.empty:
        return []
    years = []
    for col in income.columns:
        net_income = _first(income, INCOME_ROWS["net_income"], col)
        eps = _first(income, INCOME_ROWS["eps"], col)
        shares = _first(income, INCOME_ROWS["shares"], col)
        if net_income is None and eps is None:
            continue                                   # an empty (all-NaN) fiscal year
        years.append({"col": col, "fye": pd.Timestamp(col).normalize(), "ni": net_income, "eps": eps,
                      "shares": shares if shares and shares > 0 else None})
    reported = [y["shares"] for y in years if y["shares"]]
    yardstick = reference_shares or (statistics.median(reported) if reported else None)

    def plausible(s):
        return s is not None and s > 0 and (yardstick is None or 0.5 <= s / yardstick <= 2.0)

    good = [y for y in years if plausible(y["shares"])]
    rows = []
    for y in years:
        shares, net_income, eps = y["shares"], y["ni"], y["eps"]
        if not plausible(shares):
            from_eps = net_income / eps if net_income and eps and net_income * eps > 0 else None
            earlier = [g for g in good if g["fye"] < y["fye"]]
            later = [g for g in good if g["fye"] > y["fye"]]
            neighbour = (max(earlier, key=lambda g: g["fye"]) if earlier
                         else min(later, key=lambda g: g["fye"]) if later else None)
            if plausible(from_eps):
                shares, how = from_eps, "net income / reported EPS"
            elif neighbour and net_income is not None:
                shares, how = neighbour["shares"], f"FY{neighbour['fye'].year}'s count"
            elif yardstick and net_income is not None:
                shares, how = yardstick, "today's count" if reference_shares else "the median count"
            else:
                log.warning("%s FY%s: no usable share count; fiscal year skipped", ticker, y["fye"].year)
                continue
            problem = ("missing" if y["shares"] is None
                       else f"{y['shares'] / yardstick:.2f}x the expected level")
            log.warning("%s FY%s: reported share count %s; using %s", ticker, y["fye"].year, problem, how)
            eps = None                                 # re-derived from net income below
        if net_income is not None:
            implied = net_income / shares
            # Keep Yahoo's diluted EPS unless it disagrees with net income /
            # shares by more than 25% (a unit or split mismatch).
            if eps is None or implied == 0 or not 0.75 <= eps / implied <= 1.25:
                eps = implied
        else:
            net_income = eps * shares
        book = _first(balance, BOOK_ROWS, y["col"])
        as_of = (y["fye"] + pd.DateOffset(months=ANNUAL_REPORT_LAG_MONTHS)).date()
        market_cap = trailing_pe = ptb = None
        if closes is not None and not closes.empty:
            known = closes[closes.index <= as_of]
            if not known.empty and (as_of - known.index[-1]).days <= 7:
                market_cap = shares * float(known.iloc[-1])
                trailing_pe = market_cap / net_income if net_income > 0 else None
                ptb = market_cap / book if book and book > 0 else None
        rows.append((ticker, as_of, "annual_report", y["fye"].date(), market_cap, trailing_pe, ptb,
                     shares, eps, net_income, book))
    return rows


def fiscal_year_gaps(rows: list[tuple]) -> list[str]:
    """Missing fiscal years between reported ones (a gap of more than ~13 months)."""
    ends = sorted(r[3] for r in rows if r[2] == "annual_report")
    return [f"{a} -> {b}" for a, b in zip(ends, ends[1:]) if (b - a).days > 400]


def fetch_fundamentals(tickers: list[str], fallback_date: date, frames: dict,
                       stored_shares: dict | None = None, pause: float = 0.2) -> tuple[list, list, list]:
    """Snapshot + annual-report rows for every ticker.

    stored_shares ({ticker: shares_out of its latest stored snapshot}) is the
    share-count yardstick when today's quote fails, so a flaky run can't
    switch the glitch guard off and overwrite good rows with bad ones.
    Returns (rows, tickers without a snapshot, tickers without annual reports).
    """
    stored_shares = stored_shares or {}
    rows, no_snapshot, no_reports = [], [], []
    for n, ticker in enumerate(tickers, 1):
        tk = yf.Ticker(ticker)
        info = _retry(lambda: tk.info, f"{ticker} quote snapshot")
        try:                                           # one malformed payload must not sink the run
            snap = parse_snapshot(ticker, info, fallback_date) if info else None
            yardstick = (reference_shares(info) if info else None) or stored_shares.get(ticker)
        except Exception as exc:
            log.warning("%s: unusable quote snapshot (%s)", ticker, exc)
            snap, yardstick = None, stored_shares.get(ticker)
        if snap is None:
            no_snapshot.append(ticker)
        else:
            rows.append(snap)
        income = _retry(lambda: tk.income_stmt, f"{ticker} income statement")
        balance = _retry(lambda: tk.balance_sheet, f"{ticker} balance sheet") if income is not None else None
        closes = frames[ticker]["close"].astype(float) if ticker in frames else None
        try:
            reports = parse_annual_reports(ticker, income, balance, closes, reference_shares=yardstick)
        except Exception as exc:
            log.warning("%s: unusable financial statements (%s)", ticker, exc)
            reports = []
        if not reports:
            no_reports.append(ticker)
        gaps = fiscal_year_gaps(reports)
        if gaps:
            log.warning("%s: fiscal year(s) missing from Yahoo between %s; the as-of join will "
                        "leave those months without fundamentals", ticker, ", ".join(gaps))
        rows.extend(reports)
        if n % 25 == 0:
            log.info("  fundamentals %d/%d", n, len(tickers))
        time.sleep(pause)
    return rows, no_snapshot, no_reports


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
def _db_value(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def price_rows(symbol: str, df: pd.DataFrame):
    for d, r in zip(df.index, df.itertuples(index=False)):
        vol = _db_value(r.volume)
        yield (symbol, d,
               _db_value(_num(r.open)), _db_value(_num(r.high)), _db_value(_num(r.low)),
               float(r.close), _db_value(_num(r.adj_close)),
               int(vol) if vol is not None else None)


def stored_state(conn) -> tuple[pd.DataFrame, dict]:
    """What earlier runs left in the database: the current universe (reused if
    Wikipedia is down) and each ticker's latest snapshot share count (the
    glitch-guard yardstick if today's quote fails)."""
    with conn.cursor() as cur:
        cur.execute("""SELECT ticker, name, sector, added_to_index FROM tickers
                       WHERE kind = 'stock' AND in_universe ORDER BY ticker""")
        universe = pd.DataFrame(cur.fetchall(), columns=["ticker", "name", "sector", "added"])
        cur.execute("""SELECT DISTINCT ON (ticker) ticker, shares_out FROM fundamentals
                       WHERE source = 'snapshot' AND shares_out > 0
                       ORDER BY ticker, as_of_date DESC""")
        shares = dict(cur.fetchall())
    return universe, shares


def load(conn, tickers: pd.DataFrame, frames: dict, fundamentals: list) -> None:
    """Replace prices and the universe in one transaction; add the fundamentals rows.

    Prices are fully reloaded every run, so the table always matches this run's
    window. Nothing else is ever deleted: a stock missing from this run (left
    the index, or a source was down) is only flagged in_universe = FALSE, so its
    fundamentals -- including live snapshots, which can't be re-downloaded
    later -- survive as point-in-time history.
    """
    with conn:                                   # commit on success, rollback on error
        with conn.cursor() as cur:
            cur.execute("UPDATE tickers SET in_universe = FALSE WHERE NOT (ticker = ANY(%s))",
                        (tickers["ticker"].tolist(),))
            ticker_rows = [(r.ticker, _db_value(r.name), _db_value(r.sector), r.kind, _db_value(r.added))
                           for r in tickers.itertuples(index=False)]
            execute_values(cur, """
                INSERT INTO tickers (ticker, name, sector, kind, added_to_index, in_universe) VALUES %s
                ON CONFLICT (ticker) DO UPDATE
                   SET name = EXCLUDED.name, sector = EXCLUDED.sector, kind = EXCLUDED.kind,
                       added_to_index = EXCLUDED.added_to_index, in_universe = TRUE
            """, ticker_rows, template="(%s, %s, %s, %s, %s, TRUE)")

            cur.execute("DELETE FROM prices")
            rows = [row for sym, df in frames.items() for row in price_rows(sym, df)]
            execute_values(cur, """
                INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume) VALUES %s
                ON CONFLICT (ticker, date) DO UPDATE
                   SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                       close = EXCLUDED.close, adj_close = EXCLUDED.adj_close, volume = EXCLUDED.volume
            """, rows, page_size=2000)
            log.info("loaded %d price rows for %d series", len(rows), len(frames))

            if fundamentals:
                execute_values(cur, """
                    INSERT INTO fundamentals (ticker, as_of_date, source, period_end, market_cap,
                                              trailing_pe, price_to_book, shares_out, trailing_eps,
                                              net_income, book_equity) VALUES %s
                    ON CONFLICT (ticker, as_of_date, source) DO UPDATE
                       SET period_end = EXCLUDED.period_end, market_cap = EXCLUDED.market_cap,
                           trailing_pe = EXCLUDED.trailing_pe, price_to_book = EXCLUDED.price_to_book,
                           shares_out = EXCLUDED.shares_out, trailing_eps = EXCLUDED.trailing_eps,
                           net_income = EXCLUDED.net_income, book_equity = EXCLUDED.book_equity
                """, fundamentals)
                log.info("loaded %d fundamentals rows (%d annual reports, %d snapshots)", len(fundamentals),
                         sum(r[2] == "annual_report" for r in fundamentals),
                         sum(r[2] == "snapshot" for r in fundamentals))


def build_factors(conn, snapshot_fundamentals: bool = False) -> None:
    mode = "snapshot" if snapshot_fundamentals else "point_in_time"
    if snapshot_fundamentals:
        log.warning("BIAS-AUDIT MODE: applying today's fundamentals snapshot to every month "
                    "(look-ahead by construction). Rerun without --snapshot-fundamentals afterwards.")
    log.info("building factor_scores (sql/factor_queries.sql, fundamentals: %s) ...", mode)
    run_sql_file(conn, SQL_DIR / "factor_queries.sql", settings={"backtest.fundamentals_mode": mode})


def summarize(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE kind = 'stock' AND in_universe),
                   (SELECT COUNT(*) FROM prices), (SELECT MIN(date) FROM prices),
                   (SELECT MAX(date) FROM prices),
                   (SELECT COUNT(DISTINCT f.ticker) FROM fundamentals f JOIN tickers t USING (ticker)
                     WHERE f.source = 'annual_report' AND t.in_universe),
                   (SELECT COUNT(DISTINCT f.ticker) FROM fundamentals f JOIN tickers t USING (ticker)
                     WHERE f.source = 'snapshot' AND t.in_universe)
            FROM tickers""")
        n_stocks, n_prices, d0, d1, n_reports, n_snaps = cur.fetchone()
        cur.execute("""
            SELECT COUNT(*), COUNT(DISTINCT month_end), MIN(month_end), MAX(month_end),
                   MIN(month_end) FILTER (WHERE momentum_z IS NOT NULL),
                   COUNT(*) FILTER (WHERE factors_used = 3),
                   COUNT(*) FILTER (WHERE fundamentals_as_of > month_end)
            FROM factor_scores""")
        n_scores, n_months, m0, m1, mom0, n_full, n_ahead = cur.fetchone()
    log.info("database: %d stocks, %d price rows (%s .. %s); annual reports for %d stocks, "
             "snapshots for %d", n_stocks, n_prices, d0, d1, n_reports, n_snaps)
    log.info("factor_scores: %d rows over %d month-ends (%s .. %s); momentum live from %s; "
             "%d rows use all 3 factors; %d rows use fundamentals dated after their month-end",
             n_scores, n_months, m0, m1, mom0, n_full, n_ahead)


# -----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-tickers", type=int, default=100, help="universe size (default 100)")
    p.add_argument("--backtest-months", type=int, default=24,
                   help="monthly rebalances to support; 12 warm-up months are added (default 24)")
    p.add_argument("--end", type=date.fromisoformat, default=None,
                   help="last date to include, YYYY-MM-DD (default: end of the last completed month)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    p.add_argument("--universe", choices=["start-members", "current"], default="start-members",
                   help="start-members (default): only stocks already in the index at the first "
                        "rebalance; current: today's membership, including names added during the "
                        "backtest (inclusion-biased; for the bias audit)")
    p.add_argument("--tickers", type=str, default=None,
                   help="comma-separated explicit universe; skips the Wikipedia sample")
    p.add_argument("--factors-only", action="store_true",
                   help="skip the download; just rebuild factor_scores from the data already loaded")
    p.add_argument("--snapshot-fundamentals", action="store_true",
                   help="bias audit: apply today's fundamentals snapshot to every month (look-ahead)")
    args = p.parse_args(argv)
    if args.n_tickers < MIN_UNIVERSE:
        p.error(f"--n-tickers must be at least {MIN_UNIVERSE}")
    if args.backtest_months < 1:
        p.error("--backtest-months must be positive")
    return args


def resolve_end(requested: date | None, now_et: datetime) -> date:
    """The last month-end to include. A requested --end must be a calendar
    month-end that has already closed, so no partial month enters the data."""
    latest = last_complete_month_end(now_et)
    if requested is None:
        return latest
    if requested != (pd.Timestamp(requested) + pd.offsets.MonthEnd(0)).date():
        sys.exit(f"--end {requested} is not a month-end (e.g. 2026-06-30); partial months are not allowed.")
    if requested > latest:
        sys.exit(f"--end {requested} is after the last completed month ({latest}).")
    return requested


def choose_universe(args, first_rank: date, stored: pd.DataFrame) -> pd.DataFrame:
    if args.tickers:
        wanted = list(dict.fromkeys(yahoo_symbol(t) for t in args.tickers.split(",") if t.strip()))
        clash = sorted(set(wanted) & {BENCHMARK, *OPTIONAL_SERIES})
        if clash:
            sys.exit(f"--tickers can't include benchmark series ({', '.join(clash)}); they are loaded anyway.")
        try:
            known = scrape_sp500().set_index("ticker")
        except RuntimeError:
            known = fallback_universe().set_index("ticker")
        log.info("explicit universe: %d tickers", len(wanted))
        return pd.DataFrame({"ticker": wanted,
                             "name": [known["name"].get(t) for t in wanted],
                             "sector": [known["sector"].get(t) for t in wanted],
                             "added": [known["added"].get(t) for t in wanted]})
    try:
        constituents = scrape_sp500()
        log.info("Wikipedia: %d S&P 500 constituents", len(constituents))
    except RuntimeError as exc:
        if not stored.empty:
            # A sampled universe is already in the database: keep it rather than
            # swapping in a different list because Wikipedia had a bad minute.
            fallback = stored.copy()
            log.warning("%s -- reusing the %d-stock universe stored by the last run", exc, len(fallback))
        else:
            fallback = fallback_universe()
            log.warning("%s -- using the hardcoded %d-stock fallback list", exc, len(fallback))
        if args.universe == "start-members":
            fallback = start_members(fallback, first_rank, allow_unknown=fallback["added"].isna().all())
        return fallback.reset_index(drop=True)
    if args.universe == "start-members":
        eligible = start_members(constituents, first_rank)
        excluded = len(constituents) - len(eligible)
        (log.info if excluded else log.warning)(
            "%d of them were already in the index on %s (%d later additions excluded)",
            len(eligible), first_rank, excluded)
    else:
        eligible = constituents
        log.warning("--universe current: includes stocks added to the index during the "
                    "backtest (inclusion bias; for the bias audit)")
    universe = stratified_sample(eligible, args.n_tickers, args.seed)
    log.info("sampled %d tickers across %d sectors", len(universe), universe["sector"].nunique())
    return universe


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # we report failures ourselves
    args = parse_args(argv)

    if args.factors_only:
        conn = connect()
        try:
            run_sql_file(conn, SQL_DIR / "schema.sql")
            build_factors(conn, args.snapshot_fundamentals)
            summarize(conn)
        finally:
            conn.close()
        return

    now_et = pd.Timestamp.now(tz="America/New_York").to_pydatetime()
    end = resolve_end(args.end, now_et)
    start, end_exclusive = price_window(end, args.backtest_months)
    first_rank = first_rank_date(end, args.backtest_months)
    log.info("price window %s .. %s (%d-month backtest from %s + %d-month momentum warm-up)",
             start, end, args.backtest_months, first_rank, MOMENTUM_WARMUP_MONTHS)

    conn = connect()                     # read what earlier runs stored, then let go during downloads
    try:
        run_sql_file(conn, SQL_DIR / "schema.sql")
        stored_universe, stored_shares = stored_state(conn)
    finally:
        conn.close()

    # 0. canary: without the S&P 500 there is no benchmark, so find out first ------
    frames, _ = download_prices([BENCHMARK], start, end_exclusive)
    if BENCHMARK not in frames:
        sys.exit(f"Could not download the benchmark {BENCHMARK} -- Yahoo looks unavailable; "
                 "nothing was written. Try again later.")

    # 1. universe ----------------------------------------------------------------
    universe = choose_universe(args, first_rank, stored_universe)

    # 2. prices ------------------------------------------------------------------
    stock_symbols = universe["ticker"].tolist()
    more, failed = download_prices(stock_symbols + list(OPTIONAL_SERIES), start, end_exclusive)
    frames.update(more)
    for sym in OPTIONAL_SERIES:
        if sym not in frames:
            log.warning("%s unavailable; backtest.py will fall back (see README)", sym)
    failed_stocks = [s for s in failed if s in stock_symbols]
    if failed_stocks:
        log.warning("dropping %d ticker(s) with no price data: %s", len(failed_stocks), ", ".join(failed_stocks))
    universe = universe[universe["ticker"].isin(frames)].reset_index(drop=True)
    if len(universe) < MIN_UNIVERSE:
        sys.exit(f"Only {len(universe)} stocks downloaded (need {MIN_UNIVERSE}) -- nothing was written.")
    flag_suspect_moves({s: f for s, f in frames.items() if s in set(universe["ticker"])})
    log.info("prices: %d stocks + %d index series", len(universe),
             sum(s in frames for s in (BENCHMARK, *OPTIONAL_SERIES)))

    # 3. fundamentals --------------------------------------------------------------
    log.info("fetching fundamentals (annual reports + snapshot) for %d stocks ...", len(universe))
    fundamentals, no_snapshot, no_reports = fetch_fundamentals(universe["ticker"].tolist(),
                                                               now_et.date(), frames, stored_shares)
    if no_reports:
        log.warning("no annual reports for %d ticker(s) (momentum-only in the backtest): %s",
                    len(no_reports), ", ".join(no_reports))
    if no_snapshot:
        log.warning("no quote snapshot for %d ticker(s): %s", len(no_snapshot), ", ".join(no_snapshot))

    # 4. load + factors --------------------------------------------------------------
    meta = [(BENCHMARK, "S&P 500 (price index)", None, "index", None)]
    meta += [(sym, name, None, kind, None) for sym, (name, kind) in OPTIONAL_SERIES.items() if sym in frames]
    tickers = pd.concat([universe.assign(kind="stock")[["ticker", "name", "sector", "kind", "added"]],
                         pd.DataFrame(meta, columns=["ticker", "name", "sector", "kind", "added"])],
                        ignore_index=True)
    conn = connect()
    try:
        load(conn, tickers, frames, fundamentals)
        build_factors(conn, args.snapshot_fundamentals)
        summarize(conn)
    finally:
        conn.close()
    log.info("done. Next: python backtest.py")


if __name__ == "__main__":
    main()
