-- =============================================================================
-- factor_queries.sql
--
-- Rebuilds factor_scores from prices + fundamentals. One row per stock per
-- month-end t, holding only what was knowable at the close on t:
--
--   momentum   12-1 momentum: adj_close(t-1) / adj_close(t-12) - 1, via LAG()
--              PARTITION BY ticker ORDER BY month_end. Skipping the latest
--              month is the Jegadeesh-Titman / Fama-French UMD convention
--              (it sidesteps the one-month reversal effect).
--   size       market cap at t = shares_out x close(t). Z-scored on -ln(cap):
--              caps are roughly log-normal, and the sign flip makes SMALLER =
--              higher z (DFA tilts toward small caps).
--   value      P/E at t = close(t) / trailing EPS. Z-scored on the earnings
--              yield E/P = 1 / (P/E): it is the same ranking as "lower P/E =
--              higher z" (DFA tilts toward cheap), but stays well behaved when
--              earnings are near zero. EPS <= 0 -> no P/E -> NULL value score.
--
-- Each signal is z-scored cross-sectionally (WINDOW w AS (PARTITION BY
-- month_end)), winsorized at +/-3, and the non-null z-scores are averaged into
-- composite_score. NTILE(5) then assigns quintiles; quintile 1 is the portfolio.
-- (The IS NOT NULL guards in the z-scores matter: Postgres' LEAST/GREATEST
-- skip NULL arguments, so LEAST(3, NULL) is 3 -- without them a missing
-- signal would be winsorized into the maximum score.)
--
-- NO LOOK-AHEAD. Every input to a row keyed t was public at t:
--   * prices: only month-ends <= t (forward returns t -> t+1 are attached
--     later, in backtest.py);
--   * fundamentals: an as-of join picks the latest row with as_of_date <= t,
--     i.e. the last annual report that had been filed (fiscal year-end + 4
--     months), or a live snapshot taken on or before t. Nothing older than 13
--     months is used (annual reports arrive every 12): when Yahoo is missing a
--     fiscal year, the stock gets no size/value score rather than a stale one.
--     fundamentals_as_of records which row each score used.
--
-- Bias-audit mode: run with the setting backtest.fundamentals_mode = 'snapshot'
-- (python fetch_data.py --factors-only --snapshot-fundamentals) to apply TODAY'S
-- snapshot to every month instead, the naive approach. It is look-ahead by
-- construction and exists only to measure how much that bias inflates results.
--
-- Run:  psql "%DATABASE_URL%" -f sql/factor_queries.sql
--       (fetch_data.py runs it automatically after loading data)
-- =============================================================================

BEGIN;

TRUNCATE factor_scores;

INSERT INTO factor_scores (ticker, month_end,
                           momentum_12m, size_market_cap, value_pe,
                           momentum_z, size_z, value_z,
                           composite_score, factors_used, composite_quintile,
                           fundamentals_as_of)
WITH params AS (
    SELECT 3.0::float8 AS z_cap,        -- winsorize z-scores at +/- 3 standard deviations
           10          AS min_names,    -- need >= 10 stocks with a signal to z-score it that month
           INTERVAL '13 months' AS max_staleness,   -- one annual cycle + a month: an older row
                                                    -- means a fiscal year is missing, so use none
           COALESCE(NULLIF(current_setting('backtest.fundamentals_mode', true), ''),
                    'point_in_time') AS fundamentals_mode
),
stock_months AS (
    SELECT mp.ticker, mp.month_end, mp.close, mp.adj_close
    FROM monthly_prices mp
    JOIN tickers t ON t.ticker = mp.ticker
    WHERE t.kind = 'stock' AND t.in_universe
),
lagged AS (
    SELECT ticker, month_end, close,
           LAG(adj_close, 1)  OVER w AS adj_close_lag1,     -- month-end t-1
           LAG(adj_close, 12) OVER w AS adj_close_lag12,    -- month-end t-12
           LAG(month_end, 12) OVER w AS month_end_lag12
    FROM stock_months
    WINDOW w AS (PARTITION BY ticker ORDER BY month_end)
),
fundamentals_pit AS (
    -- As-of join: the most recent fundamentals row that was public at t. A
    -- stock with none gets NULL size and value scores (its composite is then
    -- momentum alone) rather than borrowing numbers from the future.
    SELECT l.ticker, l.month_end, f.as_of_date, f.shares_out, f.trailing_eps
    FROM lagged l
    CROSS JOIN params p
    LEFT JOIN LATERAL (
        SELECT f.as_of_date, f.shares_out, f.trailing_eps
        FROM fundamentals f
        WHERE f.ticker = l.ticker
          AND CASE WHEN p.fundamentals_mode = 'snapshot'
                   THEN f.source = 'snapshot'                       -- bias audit: today's numbers, every month
                   ELSE f.as_of_date <= l.month_end                 -- public by t ...
                    AND f.as_of_date >  l.month_end - p.max_staleness   -- ... and still current
              END
        ORDER BY f.as_of_date DESC,
                 (f.source = 'snapshot') DESC                       -- same day: the fresher TTM snapshot
        LIMIT 1
    ) f ON TRUE
),
raw AS (
    SELECT l.ticker, l.month_end,
           -- 12 rows back must be exactly 12 calendar months back; a gap in a
           -- ticker's history would otherwise silently stretch the window.
           CASE WHEN l.adj_close_lag12 > 0
                 AND date_trunc('month', l.month_end_lag12)
                     = date_trunc('month', l.month_end) - INTERVAL '12 months'
                THEN l.adj_close_lag1 / l.adj_close_lag12 - 1
           END                                                   AS momentum_12m,
           CASE WHEN f.shares_out > 0 THEN f.shares_out * l.close END AS size_market_cap,
           CASE WHEN f.trailing_eps > 0 THEN l.close / f.trailing_eps END AS value_pe,
           f.as_of_date                                          AS fundamentals_as_of
    FROM lagged l
    JOIN fundamentals_pit f ON f.ticker = l.ticker AND f.month_end = l.month_end
),
signals AS (
    SELECT r.*,
           r.momentum_12m                                          AS mom_sig,
           CASE WHEN r.size_market_cap > 0 THEN -LN(r.size_market_cap) END AS size_sig,   -- smaller = higher
           CASE WHEN r.value_pe > 0 THEN 1.0 / r.value_pe END      AS value_sig     -- earnings yield: cheaper = higher
    FROM raw r
),
zscores AS (
    SELECT s.ticker, s.month_end, s.momentum_12m, s.size_market_cap, s.value_pe,
           s.fundamentals_as_of,
           CASE WHEN s.mom_sig IS NOT NULL AND COUNT(s.mom_sig) OVER w >= p.min_names THEN
               GREATEST(-p.z_cap, LEAST(p.z_cap,
                   (s.mom_sig - AVG(s.mom_sig) OVER w) / NULLIF(STDDEV_SAMP(s.mom_sig) OVER w, 0)))
           END AS momentum_z,
           CASE WHEN s.size_sig IS NOT NULL AND COUNT(s.size_sig) OVER w >= p.min_names THEN
               GREATEST(-p.z_cap, LEAST(p.z_cap,
                   (s.size_sig - AVG(s.size_sig) OVER w) / NULLIF(STDDEV_SAMP(s.size_sig) OVER w, 0)))
           END AS size_z,
           CASE WHEN s.value_sig IS NOT NULL AND COUNT(s.value_sig) OVER w >= p.min_names THEN
               GREATEST(-p.z_cap, LEAST(p.z_cap,
                   (s.value_sig - AVG(s.value_sig) OVER w) / NULLIF(STDDEV_SAMP(s.value_sig) OVER w, 0)))
           END AS value_z
    FROM signals s
    CROSS JOIN params p
    WINDOW w AS (PARTITION BY s.month_end)           -- cross-sectional: all stocks, one month
),
composite AS (
    SELECT z.*,
           (z.momentum_z IS NOT NULL)::int
         + (z.size_z     IS NOT NULL)::int
         + (z.value_z    IS NOT NULL)::int AS factors_used,
           (COALESCE(z.momentum_z, 0) + COALESCE(z.size_z, 0) + COALESCE(z.value_z, 0))
           / NULLIF((z.momentum_z IS NOT NULL)::int
                  + (z.size_z     IS NOT NULL)::int
                  + (z.value_z    IS NOT NULL)::int, 0) AS composite_score
    FROM zscores z
)
SELECT c.ticker, c.month_end,
       c.momentum_12m, c.size_market_cap, c.value_pe,
       c.momentum_z, c.size_z, c.value_z,
       c.composite_score, c.factors_used,
       CASE WHEN c.composite_score IS NOT NULL THEN
           -- Partitioning on the NULL flag keeps unscored stocks out of the buckets.
           -- ticker breaks exact ties so the portfolio is deterministic.
           NTILE(5) OVER (PARTITION BY c.month_end, (c.composite_score IS NULL)
                          ORDER BY c.composite_score DESC, c.ticker)
       END AS composite_quintile,
       c.fundamentals_as_of
FROM composite c;

COMMIT;

ANALYZE factor_scores;
