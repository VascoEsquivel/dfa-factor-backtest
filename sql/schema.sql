-- =============================================================================
-- Factor-Based Portfolio Backtesting Engine -- schema
--
--   tickers         the universe: ~100 S&P 500 stocks, plus the benchmark
--                   series (^GSPC, ^SP500TR, RSP) and the T-bill yield (^IRX)
--   prices          daily OHLCV for all of the above (yfinance)
--   fundamentals    point-in-time fundamentals: one row per annual report
--                   (dated when it became public) plus one live snapshot per fetch
--   monthly_prices  (view) the last close of every calendar month
--   factor_scores   momentum / size / value + cross-sectional z-scores and the
--                   composite, one row per ticker per month-end
--                   (rebuilt by sql/factor_queries.sql)
--
-- Re-runnable: every CREATE is IF NOT EXISTS / OR REPLACE.
-- =============================================================================

CREATE TABLE IF NOT EXISTS tickers (
    ticker   TEXT PRIMARY KEY,                -- Yahoo symbol: BRK-B, not BRK.B
    name     TEXT,
    sector   TEXT,                            -- GICS sector (from Wikipedia)
    kind     TEXT NOT NULL DEFAULT 'stock'
             CHECK (kind IN ('stock', 'index', 'rate')),
             -- 'index' = ^GSPC / ^SP500TR / RSP benchmark levels,
             -- 'rate'  = ^IRX, whose "close" is a yield in percent, not a price
    added_to_index DATE,                      -- S&P 500 inclusion date (Wikipedia)
    in_universe    BOOLEAN NOT NULL DEFAULT TRUE
                   -- FALSE once a stock drops out of the sample. Rows are never
                   -- deleted, so its stored fundamentals history survives.
);
ALTER TABLE tickers ADD COLUMN IF NOT EXISTS in_universe BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS prices (
    ticker     TEXT    NOT NULL REFERENCES tickers (ticker) ON DELETE CASCADE,
    date       DATE    NOT NULL,
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION NOT NULL,     -- split-adjusted (and, by Yahoo, spin-off-adjusted),
                                              -- NOT dividend-adjusted
    adj_close  DOUBLE PRECISION,              -- split- AND dividend-adjusted (total return)
    volume     BIGINT,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS prices_date_idx ON prices (date);

-- Point-in-time fundamentals. Two kinds of rows:
--   annual_report  one per fiscal year (yfinance income statement + balance
--                  sheet, ~4 years back). as_of_date = fiscal year-end + 4
--                  months: S&P 500 companies must file their 10-K within 60
--                  days, so by then the numbers were public.
--   snapshot       today's quote-page numbers (Yahoo's trailing-12-month EPS,
--                  market cap, P/E, P/B), dated the day they were pulled.
--                  Re-running fetch_data.py later adds more snapshots.
-- factor_queries.sql only ever uses a row whose as_of_date <= the month-end.
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker         TEXT NOT NULL REFERENCES tickers (ticker) ON DELETE CASCADE,
    as_of_date     DATE NOT NULL,             -- first date these numbers were public
    source         TEXT NOT NULL CHECK (source IN ('annual_report', 'snapshot')),
    period_end     DATE,                      -- fiscal year-end (annual_report rows)
    market_cap     DOUBLE PRECISION,          -- USD at as_of_date, whole company
    trailing_pe    DOUBLE PRECISION,          -- price / trailing EPS at as_of_date (NULL if EPS <= 0)
    price_to_book  DOUBLE PRECISION,          -- stored for reference; not used in the composite
    shares_out     DOUBLE PRECISION,          -- whole-company share count, in units of THIS
                                              -- ticker's class and split-adjusted (GOOGL, BRK-B ...)
    trailing_eps   DOUBLE PRECISION,          -- EPS per share_out unit, USD (diluted, split-adjusted)
    net_income     DOUBLE PRECISION,          -- to common shareholders (annual_report rows)
    book_equity    DOUBLE PRECISION,          -- common equity (annual_report rows)
    PRIMARY KEY (ticker, as_of_date, source)
);

-- One row per ticker per calendar month: the last close on or before that
-- month's final trading day. month_end is the MARKET's last trading date of the
-- month (max date over all stocks), so every ticker in a month shares one key.
CREATE OR REPLACE VIEW monthly_prices AS
WITH month_ends AS (
    SELECT date_trunc('month', p.date)::date AS month,
           MAX(p.date)                       AS month_end
    FROM prices p
    JOIN tickers t ON t.ticker = p.ticker AND t.kind = 'stock'
    GROUP BY 1
)
SELECT DISTINCT ON (p.ticker, me.month_end)
       p.ticker,
       me.month_end,
       p.date AS price_date,                  -- = month_end unless the stock didn't trade that day
       p.close,
       p.adj_close
FROM prices p
JOIN month_ends me ON me.month = date_trunc('month', p.date)::date
ORDER BY p.ticker, me.month_end, p.date DESC;

CREATE TABLE IF NOT EXISTS factor_scores (
    ticker              TEXT NOT NULL REFERENCES tickers (ticker) ON DELETE CASCADE,
    month_end           DATE NOT NULL,
    momentum_12m        DOUBLE PRECISION,     -- 12-1 momentum: return from month-end t-12 to t-1
    size_market_cap     DOUBLE PRECISION,     -- shares_out x close at month_end (USD)
    value_pe            DOUBLE PRECISION,     -- close at month_end / trailing EPS (NULL if EPS <= 0)
    momentum_z          DOUBLE PRECISION,     -- higher = stronger past winner
    size_z              DOUBLE PRECISION,     -- inverted: higher = SMALLER company
    value_z             DOUBLE PRECISION,     -- inverted: higher = CHEAPER (lower P/E)
    composite_score     DOUBLE PRECISION,     -- mean of the non-null z-scores
    factors_used        SMALLINT,             -- how many z-scores went into the composite
    composite_quintile  SMALLINT,             -- NTILE(5) by composite_score within the month; 1 = top
    fundamentals_as_of  DATE,                 -- fundamentals row behind size/value; > month_end = look-ahead
    PRIMARY KEY (ticker, month_end)
);

CREATE INDEX IF NOT EXISTS factor_scores_month_idx ON factor_scores (month_end);
