"""
Integration tests for sql/schema.sql + sql/factor_queries.sql against a real Postgres.

Each test class builds the schema inside a throwaway Postgres schema
(dfa_test_<pid>), loads a small synthetic market, runs the factor SQL and
drops the schema afterwards. The tests only run when TEST_DATABASE_URL is
set, and never fall back to DATABASE_URL: point it at a scratch database over
a direct (non-pooled) connection, e.g.

    set TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/scratch
    python -m unittest discover tests
"""

import math
import os
import statistics
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import SQL_DIR, normalize_url, run_sql_file  # noqa: E402

DB_URL = os.getenv("TEST_DATABASE_URL")
Z_CAP = 3.0                                  # must match the params CTE in the SQL

N_STOCKS = 12                                # > min_names (10), so every signal is z-scored
MONTHS = 16                                  # Jan 2024 .. Apr 2025


def _pooled(url: str) -> bool:
    """Transaction-mode poolers (Supabase :6543, Neon -pooler hosts) can route
    consecutive statements to different sessions, which breaks SET search_path."""
    parsed = urlparse(normalize_url(url))
    return parsed.port == 6543 or "-pooler" in (parsed.hostname or "")


def month_ends(n: int, start=(2024, 1)) -> list[date]:
    """Calendar month-ends, pulled back to Friday when they fall on a weekend
    (so they look like real last trading days)."""
    out, (y, m) = [], start
    for _ in range(n):
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        d = nxt - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        out.append(d)
        y, m = nxt.year, nxt.month
    return out


MONTH_ENDS = month_ends(MONTHS)


def synthetic_prices():
    """Stock i grows at its own monthly rate; one mid-month row per month as
    well, so the view has to pick the last trading day. adj_close drifts a
    little above close (as if dividends were reinvested)."""
    rows = []
    for i in range(N_STOCKS):
        ticker = f"S{i:02d}"
        growth = 1 + (i - 5) / 100            # -5% .. +6% a month
        for k, d in enumerate(MONTH_ENDS):
            close = 10.0 * (1 + i) * growth ** k
            adj = close * (1 + 0.001 * k)
            mid = d - timedelta(days=10)
            rows.append((ticker, mid, close * 0.99, adj * 0.99))
            rows.append((ticker, d, close, adj))
    return rows


def fundamentals_rows():
    """Annual reports for each stock: one public on 2023-04-30 and one on
    2024-04-30 (fiscal years ending Dec 2022 / Dec 2023), plus a live snapshot
    dated after the last month-end. S00's latest EPS is negative (no P/E)."""
    rows = []
    for i in range(N_STOCKS):
        ticker = f"S{i:02d}"
        shares = 1e6 * (i + 1) ** 2                   # caps spread over two orders of magnitude
        for as_of, eps in ((date(2023, 4, 30), 1.0 + i / 10), (date(2024, 4, 30), 1.5 + i / 7)):
            if ticker == "S00" and as_of.year == 2024:
                eps = -0.5
            rows.append((ticker, as_of, "annual_report", date(as_of.year - 1, 12, 31),
                         shares, eps, eps * shares))
        rows.append((ticker, date(2025, 6, 30), "snapshot", None, shares * 1.01, 9.99, None))
    return rows


@unittest.skipUnless(DB_URL, "set TEST_DATABASE_URL (a scratch database) to run DB tests")
class FactorSQLTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if _pooled(DB_URL):
            raise unittest.SkipTest("TEST_DATABASE_URL points at a connection pooler; use a direct connection")
        cls.schema = f"dfa_test_{os.getpid()}"
        cls.conn = psycopg2.connect(normalize_url(DB_URL))
        cls.conn.autocommit = True
        with cls.conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {cls.schema} CASCADE; CREATE SCHEMA {cls.schema}")
        run_sql_file(cls.conn, SQL_DIR / "schema.sql", search_path=cls.schema)

    @classmethod
    def tearDownClass(cls):
        with cls.conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {cls.schema} CASCADE")
        cls.conn.close()

    # -- helpers -----------------------------------------------------------------
    def setUp(self):
        self.load()

    def load(self, prices=None, fundamentals=None):
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema}")
            cur.execute("TRUNCATE factor_scores, fundamentals, prices, tickers CASCADE")
            for i in range(N_STOCKS):
                cur.execute("INSERT INTO tickers (ticker, name, sector, kind) VALUES (%s, %s, 'Test', 'stock')",
                            (f"S{i:02d}", f"Stock {i}"))
            cur.execute("INSERT INTO tickers (ticker, name, kind) VALUES ('^IDX', 'Index', 'index')")
            for t, d, c, a in prices or synthetic_prices():
                cur.execute("INSERT INTO prices (ticker, date, close, adj_close) VALUES (%s, %s, %s, %s)",
                            (t, d, c, a))
            for d in MONTH_ENDS:            # an index row on a LATER date than any stock: must not move month_end
                cur.execute("INSERT INTO prices (ticker, date, close, adj_close) VALUES ('^IDX', %s, 100, 100)",
                            (d + timedelta(days=1) if d.weekday() < 4 else d,))
            for t, as_of, src, pe, shares, eps, ni in fundamentals or fundamentals_rows():
                cur.execute("""INSERT INTO fundamentals (ticker, as_of_date, source, period_end, shares_out,
                                                         trailing_eps, net_income)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)""", (t, as_of, src, pe, shares, eps, ni))

    def build(self, mode=None):
        run_sql_file(self.conn, SQL_DIR / "factor_queries.sql", search_path=self.schema,
                     settings={"backtest.fundamentals_mode": mode} if mode else None)

    def scores(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema}")
            cur.execute("""SELECT ticker, month_end, momentum_12m, size_market_cap, value_pe,
                                  momentum_z, size_z, value_z, composite_score, factors_used,
                                  composite_quintile, fundamentals_as_of
                           FROM factor_scores ORDER BY month_end, ticker""")
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def month(self, rows, d):
        return [r for r in rows if r["month_end"] == d]

    # -- monthly_prices ------------------------------------------------------------
    def test_monthly_prices_takes_last_trading_day(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema}")
            cur.execute("SELECT month_end, close FROM monthly_prices WHERE ticker = 'S03' ORDER BY month_end")
            rows = cur.fetchall()
        self.assertEqual([r[0] for r in rows], MONTH_ENDS)        # stock dates define month_end, not ^IDX
        self.assertAlmostEqual(rows[4][1], 40.0 * (1 + (3 - 5) / 100) ** 4)

    # -- momentum --------------------------------------------------------------------
    def test_momentum_is_12_minus_1(self):
        self.build()
        rows = self.scores()
        t = MONTH_ENDS[14]
        for i in (0, 4, 11):
            g = 1 + (i - 5) / 100
            adj = lambda k: 10.0 * (1 + i) * g ** k * (1 + 0.001 * k)      # noqa: E731
            expected = adj(13) / adj(2) - 1                                 # t-1 over t-12
            got = next(r for r in self.month(rows, t) if r["ticker"] == f"S{i:02d}")["momentum_12m"]
            self.assertAlmostEqual(got, expected, places=12)

    def test_momentum_needs_12_months_of_history(self):
        self.build()
        rows = self.scores()
        for d in MONTH_ENDS[:12]:
            self.assertTrue(all(r["momentum_12m"] is None and r["momentum_z"] is None for r in self.month(rows, d)))
        self.assertTrue(all(r["momentum_12m"] is not None for r in self.month(rows, MONTH_ENDS[12])))

    def test_momentum_null_across_a_gap(self):
        # Drop S05's prices for one month: 12 rows back is then 13 calendar months back.
        gap = MONTH_ENDS[6]
        prices = [p for p in synthetic_prices()
                  if not (p[0] == "S05" and (p[1].year, p[1].month) == (gap.year, gap.month))]
        self.load(prices=prices)
        self.build()
        rows = self.scores()
        s05 = {r["month_end"]: r for r in rows if r["ticker"] == "S05"}
        s04 = {r["month_end"]: r for r in rows if r["ticker"] == "S04"}
        for d in MONTH_ENDS[12:]:
            self.assertIsNone(s05[d]["momentum_12m"], d)      # window spans the missing month
            self.assertIsNotNone(s04[d]["momentum_12m"], d)   # control: unaffected neighbour

    # -- size / value ------------------------------------------------------------------
    def test_size_uses_point_in_time_shares_times_price(self):
        self.build()
        rows = self.scores()
        t = MONTH_ENDS[5]                     # June 2024: the 2024-04-30 report is public
        r = next(r for r in self.month(rows, t) if r["ticker"] == "S03")
        close = 40.0 * (1 + (3 - 5) / 100) ** 5
        self.assertAlmostEqual(r["size_market_cap"], 16e6 * close, places=4)
        self.assertEqual(r["fundamentals_as_of"], date(2024, 4, 30))

    def test_smaller_and_cheaper_score_higher(self):
        self.build()
        for d in MONTH_ENDS[12:]:
            month = sorted(self.month(self.scores(), d), key=lambda r: r["size_market_cap"])
            sizes = [r["size_z"] for r in month]
            self.assertEqual(sizes, sorted(sizes, reverse=True), "size_z must fall as market cap rises")
            valued = sorted((r for r in month if r["value_pe"] is not None), key=lambda r: r["value_pe"])
            values = [r["value_z"] for r in valued]
            self.assertEqual(values, sorted(values, reverse=True), "value_z must fall as P/E rises")

    def test_negative_eps_gives_null_value_not_max_score(self):
        # Regression test: Postgres' LEAST/GREATEST skip NULLs, so an unguarded
        # winsorization turned a missing signal into +3.
        self.build()
        for r in self.month(self.scores(), MONTH_ENDS[14]):
            if r["ticker"] == "S00":
                self.assertIsNone(r["value_pe"])
                self.assertIsNone(r["value_z"])
                self.assertEqual(r["factors_used"], 2)
            else:
                self.assertEqual(r["factors_used"], 3)

    # -- z-scores / composite / quintile ---------------------------------------------------
    def test_zscores_are_cross_sectional_and_composite_is_their_mean(self):
        self.build()
        for d in (MONTH_ENDS[3], MONTH_ENDS[14]):
            month = self.month(self.scores(), d)
            for col in ("size_z", "value_z", "momentum_z"):
                zs = [r[col] for r in month if r[col] is not None]
                if not zs:
                    continue
                if max(abs(z) for z in zs) < Z_CAP:          # nothing winsorized: exact standardization
                    self.assertAlmostEqual(statistics.mean(zs), 0.0, places=9)
                    self.assertAlmostEqual(statistics.stdev(zs), 1.0, places=9)
                self.assertTrue(all(abs(z) <= Z_CAP + 1e-12 for z in zs))
            for r in month:
                zs = [r[c] for c in ("momentum_z", "size_z", "value_z") if r[c] is not None]
                self.assertEqual(r["factors_used"], len(zs))
                self.assertAlmostEqual(r["composite_score"], sum(zs) / len(zs), places=12)

    def test_size_z_matches_hand_computation(self):
        self.build()
        d = MONTH_ENDS[14]
        month = self.month(self.scores(), d)
        sig = {r["ticker"]: -math.log(r["size_market_cap"]) for r in month}
        mu, sd = statistics.mean(sig.values()), statistics.stdev(sig.values())
        for r in month:
            expected = max(-Z_CAP, min(Z_CAP, (sig[r["ticker"]] - mu) / sd))
            self.assertAlmostEqual(r["size_z"], expected, places=10)

    def test_top_quintile_is_ntile_bucket_one(self):
        self.build()
        month = self.month(self.scores(), MONTH_ENDS[14])
        top = [r["ticker"] for r in sorted(month, key=lambda r: (-r["composite_score"], r["ticker"]))]
        q1 = sorted(r["ticker"] for r in month if r["composite_quintile"] == 1)
        self.assertEqual(len(q1), math.ceil(N_STOCKS / 5))          # NTILE puts the remainder first
        self.assertEqual(q1, sorted(top[:len(q1)]))

    # -- no look-ahead ---------------------------------------------------------------------
    def test_fundamentals_are_point_in_time(self):
        self.build()
        rows = self.scores()
        self.assertTrue(all(r["fundamentals_as_of"] is None or r["fundamentals_as_of"] <= r["month_end"]
                            for r in rows))
        jan = next(r for r in self.month(rows, MONTH_ENDS[0]) if r["ticker"] == "S04")
        may = next(r for r in self.month(rows, MONTH_ENDS[4]) if r["ticker"] == "S04")
        self.assertEqual(jan["fundamentals_as_of"], date(2023, 4, 30))   # 2024 report not public yet
        self.assertEqual(may["fundamentals_as_of"], date(2024, 4, 30))
        # The live snapshot (dated 2025-06-30) is never used for these months.
        self.assertTrue(all(r["fundamentals_as_of"] != date(2025, 6, 30) for r in rows))

    def test_stale_fundamentals_are_ignored(self):
        # Only the 2023-04-30 report (the next fiscal year is "missing"): once it
        # is 13 months old the stock gets no size/value score instead of a stale one.
        funds = [f for f in fundamentals_rows() if f[1] == date(2023, 4, 30)]
        self.load(fundamentals=funds)
        self.build()
        rows = self.scores()
        late = self.month(rows, MONTH_ENDS[4])                           # 2024-05-31: exactly 13 months
        self.assertTrue(all(r["size_z"] is None and r["value_z"] is None for r in late))
        self.assertTrue(all(r["size_z"] is not None for r in self.month(rows, MONTH_ENDS[3])))

    def test_stocks_out_of_the_universe_are_not_scored(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema}")
            cur.execute("UPDATE tickers SET in_universe = FALSE WHERE ticker = 'S07'")
        self.build()
        tickers = {r["ticker"] for r in self.scores()}
        self.assertNotIn("S07", tickers)
        self.assertEqual(len(tickers), N_STOCKS - 1)

    def test_future_prices_do_not_change_past_scores(self):
        self.build()
        cutoff = MONTH_ENDS[13]
        original = self.scores()
        before = [r for r in original if r["month_end"] <= cutoff]
        shocked = [(t, d, c * (5 if t in ("S02", "S07") else 0.2), a * (5 if t in ("S02", "S07") else 0.2))
                   if d > cutoff else (t, d, c, a) for t, d, c, a in synthetic_prices()]
        self.load(prices=shocked)
        self.build()
        shocked_scores = self.scores()
        after = [r for r in shocked_scores if r["month_end"] <= cutoff]
        self.assertEqual(before, after)
        # ... and the test has power: the shock does move the later months.
        self.assertNotEqual(self.month(original, MONTH_ENDS[15]), self.month(shocked_scores, MONTH_ENDS[15]))

    def test_future_fundamentals_do_not_change_past_scores(self):
        self.build()
        before = [r for r in self.scores() if r["month_end"] < date(2024, 4, 30)]
        funds = [(t, a, s, p, sh * 7, e * -3, n) if a == date(2024, 4, 30) else (t, a, s, p, sh, e, n)
                 for t, a, s, p, sh, e, n in fundamentals_rows()]
        self.load(fundamentals=funds)
        self.build()
        after = [r for r in self.scores() if r["month_end"] < date(2024, 4, 30)]
        self.assertEqual(before, after)

    # -- bias-audit mode --------------------------------------------------------------------
    def test_snapshot_mode_is_lookahead_and_scoped_to_one_run(self):
        self.build(mode="snapshot")
        rows = self.scores()
        self.assertTrue(all(r["fundamentals_as_of"] == date(2025, 6, 30) for r in rows))
        self.build()                      # same session: SET LOCAL must not have leaked
        rows = self.scores()
        self.assertTrue(all(r["fundamentals_as_of"] is None or r["fundamentals_as_of"] <= r["month_end"]
                            for r in rows))


if __name__ == "__main__":
    unittest.main()
