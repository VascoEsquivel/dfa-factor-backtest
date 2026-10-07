# Factor-Based Portfolio Backtesting Engine

A PostgreSQL + Python engine that scores S&P 500 stocks on the three factors
behind Dimensional's investment approach (**momentum, size and value**),
computes the factor exposures with SQL window functions, and backtests a
factor-tilted portfolio against the S&P 500.

Each month-end the engine ranks ~100 S&P 500 stocks on a composite of the three
factors, holds the top quintile (20 stocks, equal-weighted) for the next month,
and benchmarks the result against the S&P 500 total-return index. Everything is
point-in-time: a ranking uses only prices and financial statements that were
public on the ranking date.

**Headline result (24 monthly rebalances, returns earned Oct 2024 – Sep 2026):**

|                            | Factor portfolio | S&P 500 (total return) |
|----------------------------|-----------------:|-----------------------:|
| Cumulative return          | **+44.9%**       | +36.1%                 |
| Annualized return          | **+20.4%**       | +16.7%                 |
| Annualized volatility      | 19.0%            | 12.6%                  |
| Sharpe ratio               | 0.86 (± 0.72)    | 0.97 (± 0.72)          |
| Max drawdown (daily)       | −21.2%           | −18.7%                 |
| Max drawdown (month-end)   | −11.3%           | −7.5%                  |

![Growth of $1: factor portfolio vs. S&P 500](docs/backtest_results.png)

The tilt beat the index on return, but it did so with 50% more volatility. On a
risk-adjusted basis it did not beat the index: its Sharpe ratio was 0.86 against
0.97. With only 24 months of data, neither gap is statistically meaningful.

A Fama-French regression shows the portfolio carried the exposures it was built
for:

- **Value:** a strong loading (HML +0.86, t = 3.4).
- **Momentum:** a positive loading (+0.24, t = 1.6), though not significant.
- **Alpha:** none that is significant.

The more interesting result is the [bias audit](#bias-audit-what-the-naive-version-got-wrong).
The textbook first version of this project used today's fundamentals and today's
index membership. It reported **+49% a year**. Most of that came from selection
and look-ahead bias. Point-in-time data, and a universe fixed at the start of the
test, cut it to +20.4%.

---

## How it works

```
Wikipedia (S&P 500 list)  --+
yfinance (prices, annual    +--> fetch_data.py --> PostgreSQL --> sql/factor_queries.sql --> factor_scores
  statements, quotes)       |                      (tickers, prices,                              |
Ken French data library ----+--------------------   fundamentals)             backtest.py <------+
                                                                              (report, chart, CSVs)
```

| Table | Contents |
|---|---|
| `tickers` | ticker, name, GICS sector, kind (stock / index / rate), S&P 500 inclusion date, in-universe flag |
| `prices` | daily OHLCV, close and dividend-adjusted close, ~3 years |
| `fundamentals` | point-in-time rows: one per annual report (dated when it became public) plus a live quote snapshot per fetch |
| `monthly_prices` (view) | last close of each calendar month, keyed by the market's last trading day |
| `factor_scores` | per stock per month-end: raw momentum / market cap / P/E, three z-scores, composite, quintile, and which fundamentals row was used |

### Data: `fetch_data.py`

1. **Universe.** Scrapes the current S&P 500 list from Wikipedia.
   - Only stocks that were already in the index on the first rebalance date
     (30 Sep 2024) are kept. That excludes 37 of today's 503 members. Many of
     those were added *because* they rallied during the test window, and ranking
     them before they joined would be selecting on the future.
   - From the remaining members it draws 100 stocks, sampled in proportion to
     GICS sector weights.
   - The sample is deterministic. Each sector's names are ordered by a hash of
     the ticker, so a future index change swaps at most one name instead of
     reshuffling the sample.
   - If the scrape fails, the run reuses the universe stored by the last run. On
     a first run there is nothing stored, so it uses a hardcoded list of 100
     long-standing members instead.
2. **Prices.** ~3 years of daily OHLCV from yfinance: a 12-month momentum
   warm-up plus the 24-month backtest, ending at the last *completed* month. It
   also pulls four reference series:
   - the S&P 500 price index (`^GSPC`)
   - the S&P 500 total-return index (`^SP500TR`)
   - the equal-weight S&P 500 ETF (`RSP`)
   - the 13-week T-bill yield (`^IRX`)
3. **Fundamentals.** Two kinds of rows go into `fundamentals`:
   - **One row per annual report**, about 4 fiscal years from yfinance's income
     statement and balance sheet: net income, diluted EPS, diluted shares, and
     book equity. Each row is dated fiscal year-end + 4 months. S&P 500 companies
     must file their 10-K within 60 days, so by that date the numbers were public.
   - **Today's quote snapshot**: market cap, trailing P/E, P/B, and EPS. It
     only drives the current holdings, not the backtest. Snapshots accumulate
     across runs, so re-running monthly builds a genuine point-in-time history.
4. **Factors.** Runs `sql/factor_queries.sql`.

**Failure handling.** yfinance and Wikipedia are both flaky, so:

- The S&P 500 index is downloaded first as a canary. If Yahoo is down, the run
  stops immediately and writes nothing.
- Stock prices are downloaded in batches. Any ticker that comes back empty or
  malformed is retried on its own, then dropped with a warning. The run only
  stops if fewer than 25 stocks survive.
- A fundamentals failure leaves that stock with a momentum-only score.
- The data load is one transaction. If the factor rebuild that follows fails,
  the previous `factor_scores` table stays intact, and
  `python fetch_data.py --factors-only` redoes just that step.
- Nothing is ever deleted except prices, which are reloaded in full each run. A
  stock that drops out of a run is only flagged out of the universe, so its
  stored snapshots survive.

**Yahoo's statement data has holes and glitches**, so each fiscal year's share
count is checked against today's count. If today's quote fails, the check uses
the last stored snapshot instead. A count that is missing, or more than 2x off,
is repaired in this order of preference:

1. **The year's own net income ÷ EPS.**
   - Travelers' FY2024 share count was reported 110x too high next to a correct
     EPS, so this recovers the true point-in-time count.
2. **The nearest earlier year's count.**
   - Clorox FY2025 has net income on Yahoo but a blank share count and EPS.
     Before this fix, the year was silently dropped, and Clorox was scored on
     stale numbers.
3. **Today's count.**
   - Every year of Digital Realty's statements is ~4x off.
   - This last resort carries a small look-ahead, because share counts drift
     with buybacks.

Every repair is logged.

### The factors: `sql/factor_queries.sql`

All three are computed at each month-end *t*, then z-scored cross-sectionally:
each stock is compared with the other stocks *in the same month*.

| Factor | Raw measure (stored) | Z-scored on | Higher z means | Why |
|---|---|---|---|---|
| **Momentum** | 12-1 return: price(t−1) / price(t−12) − 1, via `LAG()` | the return | stronger past winner | Stocks that outperformed over the past year tend to keep outperforming. Skipping the latest month is the Jegadeesh–Titman / Fama–French UMD convention, because one-month returns tend to reverse. |
| **Size** | market cap = shares × close(t) | −ln(market cap) | **smaller** company | Small caps have historically earned a premium. DFA tilts small. The log tames the 1,000x spread between the smallest and largest caps. |
| **Value** | P/E = close(t) / EPS | earnings yield E/P = 1/(P/E) | **cheaper** stock | Cheap stocks have historically earned a premium (DFA's value tilt). E/P gives the same ranking as "low P/E", but stays well behaved near zero earnings. A loss-making company has no meaningful P/E, so it gets no value score. |

Each z-score is winsorized at ±3 so one outlier can't dominate. The composite is
the average of whichever z-scores a stock has. `NTILE(5)` then assigns
quintiles, and quintile 1 is the portfolio.

The SQL centerpiece (abridged):

```sql
lagged AS (                                    -- momentum inputs
    SELECT ticker, month_end, close,
           LAG(adj_close, 1)  OVER w AS adj_close_lag1,     -- month-end t-1
           LAG(adj_close, 12) OVER w AS adj_close_lag12     -- month-end t-12
    FROM stock_months
    WINDOW w AS (PARTITION BY ticker ORDER BY month_end)
),
fundamentals_pit AS (                          -- point-in-time as-of join
    SELECT l.ticker, l.month_end, f.as_of_date, f.shares_out, f.trailing_eps
    FROM lagged l LEFT JOIN LATERAL (
        SELECT * FROM fundamentals f
        WHERE f.ticker = l.ticker
          AND f.as_of_date <= l.month_end                        -- public by t
          AND f.as_of_date >  l.month_end - INTERVAL '13 months' -- and from the latest fiscal year
        ORDER BY f.as_of_date DESC LIMIT 1) f ON TRUE
),
...
zscores AS (                                   -- cross-sectional standardization
    SELECT ...,
           (size_sig - AVG(size_sig) OVER w) / NULLIF(STDDEV_SAMP(size_sig) OVER w, 0) AS size_z
    FROM signals
    WINDOW w AS (PARTITION BY month_end)
)
... NTILE(5) OVER (PARTITION BY month_end ORDER BY composite_score DESC, ticker)
```

Annual reports arrive every 12 months, so a row older than 13 months means
Yahoo is missing a fiscal year. In that case the stock gets no size or value
score that month, rather than a stale one. In this sample that happens once:
Paramount Skydance, whose post-merger fiscal year isn't on Yahoo, is
momentum-only for its last 3 months. The report lists every such stock-month.

One Postgres detail matters here. `LEAST` and `GREATEST` skip NULL arguments,
so `LEAST(3, NULL)` is 3. The first version of the winsorization therefore gave
every *missing* signal (a loss-making company's P/E, a spin-off's momentum) the
maximum score of +3. The fix and a regression test are both in the repo.

### The backtest: `backtest.py`

- **Portfolio.** At each month-end *t*, hold the top quintile equal-weighted and
  earn each stock's dividend-adjusted return from *t* to *t+1*.
- **No look-ahead.** The forward return is attached with a one-month shift
  *after* ranking, so a ranking never sees the return it is graded on. The
  pandas top quintile is asserted to match the SQL `NTILE` bucket every month.
- **Benchmarks.**
  - S&P 500 total return (`^SP500TR`), since the portfolio's returns include
    dividends. If it has a gap in the test window, the run falls back to `^GSPC`
    with a warning.
  - The S&P 500 price index (`^GSPC`).
  - The **equal-weighted universe**: all ~100 stocks with no tilt. This
    separates what factor selection added from what equal weighting did.
  - **RSP**, the real equal-weight S&P 500. It is used to measure survivorship
    bias.
- **Metrics.**
  - Cumulative and annualized return, and volatility.
  - Sharpe ratio on excess returns over T-bills, with its Lo (2002) standard
    error.
  - Max drawdown, both on daily paths (equal-weight buy-and-hold within each
    month) and on month-end values.
  - Active return, tracking error, information ratio, beta, and turnover.
- **Attribution.** Single-factor top-quintile portfolios.
- **Factor regression.** A Fama-French 3-factor + momentum regression on Ken
  French's monthly factors, showing which exposures the portfolio actually
  carried.
- **Outputs.** A chart, `monthly_returns.csv`, `holdings.csv`, and
  `summary.json`. Sanity checks run *before* any file is written, so a bad run
  can't overwrite good results. They fail the run if:
  - the portfolio's returns equal the benchmark's;
  - any metric is NaN or infinite;
  - the independently built daily path doesn't land exactly where compounding
    the monthly returns does.

---

## Results

`python backtest.py` output for the committed data (fetched 6 Oct 2026; prices
through 30 Sep 2026):

```
rebalances : 24 month-ends, 2024-09-30 .. 2026-08-31 (returns earned Oct 2024 .. Sep 2026)
universe   : ~100 scored stocks per month, holding 20 (top quintile); all 3 factors on 94% of stock-months
coverage   : momentum 100%, size 100%, value 94% of stock-months (value is blank when earnings <= 0)
look-ahead : 0% of stock-months use fundamentals dated after their month-end
no fundamentals: PSKY 3 month(s) 2026-06..2026-08  (momentum-only those months: no point-in-time annual report within 13 months)
risk-free  : 13-week T-bill (^IRX), avg 3.95%/yr

1) Performance
                          Factor portfolio        S&P 500 TR     S&P 500 price       EW universe  S&P 500 EW (RSP)
cumulative return                  +44.9%            +36.1%            +32.8%            +31.2%            +20.0%
annualized return                  +20.4%            +16.7%            +15.2%            +14.5%             +9.5%
annualized volatility               19.0%             12.6%             12.6%             12.3%             12.0%
sharpe ratio                         0.86              0.97              0.88              0.85              0.49
sharpe std error                  +/-0.72           +/-0.72           +/-0.72           +/-0.72           +/-0.71
max drawdown (daily)               -21.2%            -18.7%            -18.9%            -17.3%            -17.8%
max drawdown (month-end)           -11.3%             -7.5%             -7.8%             -8.3%             -9.1%

2) Factor portfolio vs. S&P 500 TR (^SP500TR)
   active return +4.1%/yr (mean monthly difference x 12; the gap in annualized returns above is +3.7 points)
   tracking error 13.6%, information ratio 0.30, beta 1.05, beat the index in 13 of 24 months

3) Attribution: top quintile of each factor alone (equal weight)
                      ann. return  volatility  Sharpe  max DD (m/e)
   Momentum only           +31.8%       23.4%    1.13        -10.7%
   Size only                +8.4%       14.5%    0.36        -11.3%
   Value only              +10.7%       16.3%    0.46        -14.1%
   Factor portfolio        +20.4%       19.0%    0.86        -11.3%

4) Turnover: 24% of the portfolio replaced per month (one-way, avg); at 10 bps per trade that is roughly 0.57%/yr of costs (not deducted above)

5) Fama-French 3-factor + momentum regression, 23 months (2024-10 .. 2026-08), t-stats in parentheses
                                  alpha/yr        Mkt-RF           SMB           HML           Mom    R2
   Factor portfolio           -0.6% (-0.1)  +1.15 (+5.5)  +0.25 (+0.8)  +0.86 (+3.4)  +0.24 (+1.6)  0.75
   Equal-weight universe      -1.9% (-0.4)  +0.87 (+8.2)  +0.17 (+1.1)  +0.57 (+4.4)  +0.01 (+0.1)  0.85
   Factor - EW universe       +1.3% (+0.2)  +0.28 (+1.6)  +0.07 (+0.3)  +0.29 (+1.4)  +0.23 (+1.9)  0.36
```

What to take from it:

- **The tilt worked as designed.** The portfolio's return is explained by the
  exposures it was built to carry:
  - a market beta above 1;
  - a strong value loading (HML +0.86, t = 3.4);
  - a positive momentum loading (+0.24, t = 1.6).

  The alpha is about zero (t = −0.1). That is the expected result for a factor
  portfolio: it earns factor premia, not stock-picking skill. Only the market
  and value loadings are statistically significant.
- **Within the universe, momentum names did best.**
  - Sorting on one factor at a time, the momentum names returned +31.8% a year.
  - Size (+8.4%) and value (+10.7%) both trailed the S&P 500. In 2024–26 the
    largest stocks led the market.
  - The regression credits the composite's edge over its universe mainly to a
    higher beta and to value and momentum exposure. None of those three is
    individually significant over 23 months.
  - The SMB loading is weak (+0.25, t = 0.8). "Small" inside the S&P 500 still
    means large-cap, so a size tilt here is mild compared with DFA's small-cap
    funds.
- **The fair comparison is against the equal-weighted universe.** The factor
  portfolio beat its own untilted universe by 5.9 points a year in annualized
  return. Both carry the same survivorship bias, which is described next.
- **Drawdowns depend on how you measure them.** The April 2025 sell-off
  happened mid-month:
  - Month-end values show only −7.5% for the S&P 500 and −11.3% for the
    portfolio.
  - Daily paths show the real peak-to-trough: −18.7% and −21.2%.

---

## Bias audit: what the naive version got wrong

The same code can reproduce the shortcuts it was built to avoid, so their cost
is measured rather than guessed:

| Universe | Fundamentals | Factor portfolio (ann.) | Value-only (ann.) | EW universe (ann.) |
|---|---|---:|---:|---:|
| today's S&P 500 members | today's snapshot | +49.3% | +33.1% | +20.6% |
| today's S&P 500 members | point-in-time | +43.9% | +11.7% | +20.6% |
| members at the start | today's snapshot | +35.5% | +29.9% | +14.5% |
| **members at the start** | **point-in-time** (default) | **+20.4%** | **+10.7%** | **+14.5%** |

(An earlier build also carried the `LEAST(3, NULL) = 3` bug described above. It
reported +52.8%.)

- **Look-ahead in fundamentals cost ~19–21 points a year on value alone, and
  ~5–15 points on the composite.** Dividing past prices by *today's* EPS makes a
  company whose earnings later took off look cheap in 2024, before the earnings
  existed. Using annual reports dated when they were filed removes that.
- **Inclusion bias cost ~14–24 points a year on the composite, and ~6 on the
  untilted universe.** Today's index contains companies added during the test
  *because* they had rallied. Ranking them earlier selects on the future, and
  momentum is the factor most exposed: momentum-only fell from +59.1% to +31.8%
  a year once later additions were excluded.
- **Some survivorship bias remains.** The universe holds stocks that were
  members at the start *and are still members*. Roughly 37 companies were
  removed in between (as many as were added), and they are missing.
  - Their cost is measurable. The equal-weighted universe returned 14.5% a year,
    but RSP, the real equal-weight S&P 500 with point-in-time membership,
    returned 9.5%.
  - So absolute returns here are inflated by up to ~5 points a year.
  - Comparisons *within* the universe (factor portfolio vs. equal-weighted
    universe) are much less affected, because both legs carry the same bias.

Reproduce any row:

```bat
python fetch_data.py --universe current                      & rem today's members
python fetch_data.py --factors-only --snapshot-fundamentals  & rem today's fundamentals for every month
python backtest.py
python fetch_data.py                                         & rem back to the defaults
```

The snapshot rows depend on the day they are fetched, so a later rerun of the
two "today's snapshot" rows will differ somewhat.

---

## Known limitations

- **Short sample.** yfinance's free statement history goes back ~4 fiscal years.
  That supports a 24-month point-in-time backtest, not a 20-year one, so treat
  the numbers as a demonstration of method, not evidence of a premium.
- **Survivorship.** The universe is a fixed list of stocks that were S&P 500
  members at the start and still are today. Names removed in between are
  missing, worth up to ~5 points a year of inflated absolute returns, as
  measured against RSP above.
- **Fundamentals are annual and approximate.**
  - **Annual, not quarterly.** The value signal can be up to ~17 months old.
  - **A fixed lag, not filing dates.** Reports are dated FYE + 4 months, a
    conservative rule rather than each company's actual filing date.
  - **Restated figures.** yfinance shows each year as *currently* reported, so a
    later restatement leaks back in time. Monolithic Power's FY2024 earnings,
    restated in its FY2025 10-K, are one example. It did not change any holdings
    here.
  - **Spin-off price rescaling.** Yahoo rescales pre-spin-off closes. Becton
    Dickinson's pre-Feb-2026 closes are scaled by 0.79, which understates its
    market cap and P/E in earlier months.
  - **Repaired share counts.** Where a count had to be replaced with today's
    (Digital Realty), share-count drift since then is a small look-ahead.
  - **Today's numbers for the current month.** The live snapshot (Yahoo's TTM
    figures) only drives the latest month's ranking, which is not part of the
    backtest.
- **No transaction costs, slippage or taxes.** Turnover averages 24% of the
  portfolio per month. At 10 bps per trade that is ~0.6% a year, which is not
  deducted above. Trades are also assumed to fill at the ranking day's close.
- **Universe is large-cap only.** "Small" means the smaller end of the S&P 500,
  so the size tilt is much weaker than in a true small-cap portfolio.
- **Momentum is fully point-in-time.** It uses only month-end prices up to *t−1*.
  Dividend-adjusted prices are rescaled by later dividends, but only by a common
  factor that cancels in returns.
- **yfinance is an unofficial API.** A later re-download can differ slightly
  (revised data, a ticker that fails that day). The numbers above are from the
  fetch on 6 Oct 2026.

---

## Setup (Windows)

Requires Python 3.10+ and a free Postgres database ([Neon](https://neon.tech) or
[Supabase](https://supabase.com); the free tiers are plenty, since the data is ~15 MB).

```bat
git clone https://github.com/VascoEsquivel/dfa-factor-backtest.git
cd dfa-factor-backtest
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and paste your connection string:

```
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

Then run:

```bat
python fetch_data.py      & rem ~2 minutes: universe, prices, fundamentals, factor_scores
python backtest.py        & rem report + output\backtest_results.png and CSVs
```

| Flag | Default | Meaning |
|---|---|---|
| `fetch_data.py --n-tickers N` | 100 | universe size |
| `fetch_data.py --backtest-months N` | 24 | rebalances to support (12 warm-up months are added) |
| `fetch_data.py --end YYYY-MM-DD` | last completed month | stop the data at an earlier month-end (must be a month-end) |
| `fetch_data.py --tickers A,B,C` | | explicit universe instead of the sample |
| `fetch_data.py --factors-only` | | rebuild `factor_scores` without downloading |
| `fetch_data.py --universe current` | `start-members` | bias audit: today's membership |
| `fetch_data.py --snapshot-fundamentals` | | bias audit: today's fundamentals for every month |
| `backtest.py --months N` | all | only the last N rebalances |
| `backtest.py --no-ff` | | skip the Fama-French regression |
| `backtest.py --docs` | | also refresh `docs/backtest_results.png` |

`psql "%DATABASE_URL%" -f sql/factor_queries.sql` rebuilds the factors by hand.

**Tests.** There are 75. The 16 database tests only run when `TEST_DATABASE_URL`
points at a scratch Postgres database over a direct (non-pooled) connection.
They build everything in a throwaway schema and drop it afterwards.

```bat
set TEST_DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/scratch_db
python -m unittest discover tests
```

They check, among other things:

- momentum, z-scores, and composites against hand calculations
- that shocking all prices or fundamentals *after* month *t* leaves every score
  at or before *t* unchanged
- that missing signals stay NULL (the `LEAST`/`GREATEST` trap)
- the 13-month staleness cut-off
- that the bias-audit setting can't leak out of its transaction
- metrics on hand-computed series
- that the daily path matches monthly compounding and sees intra-month
  drawdowns
- each Yahoo share-count repair: Travelers, Clorox, and Digital Realty, including
  when today's quote fails
- that fetching survives failed batches, dead tickers, malformed payloads, a
  failed Wikipedia scrape, and a renamed "Date added" column

## Project layout

```
sql/schema.sql            tables + the monthly_prices view
sql/factor_queries.sql    momentum / size / value, z-scores, composite, quintiles (window functions)
fetch_data.py             Wikipedia + yfinance -> Postgres, then runs the factor SQL
backtest.py               backtest, metrics, Fama-French regression, chart, CSVs
db.py                     connection helpers (.env), .sql runner
tests/                    SQL integration tests + offline logic tests
docs/backtest_results.png the chart above
```

## Resume summary

> Built a factor-based portfolio backtesting engine using PostgreSQL and Python,
> computing momentum and value factor exposures via SQL and backtesting a
> factor-tilted portfolio against the S&P 500 benchmark.

Numbers behind it:

- **Data.** 100 S&P 500 stocks; 24 monthly rebalances (Oct 2024 – Sep 2026).
- **Performance.** The top-quintile momentum + size + value portfolio returned
  +44.9% cumulative (20.4% a year), against +36.1% (16.7% a year) for the S&P 500
  total return.
- **Risk.** Sharpe ratio 0.86 vs. 0.97; daily max drawdown −21.2% vs. −18.7%.
- **Exposures.** Value loading 0.86 (t = 3.4), alpha not significant.
- **Bias audit.** Point-in-time fundamentals and a start-of-test universe
  removed look-ahead and inclusion biases. Those biases had inflated a naive
  version to 49% a year.
