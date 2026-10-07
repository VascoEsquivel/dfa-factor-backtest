"""Offline tests for fetch_data.py: no network, no database. yfinance and
requests are replaced with fakes, so failure handling can be exercised."""

import logging
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_data as fd  # noqa: E402

logging.getLogger("fetch_data").setLevel(logging.CRITICAL)    # the failure paths log warnings on purpose


def frame(dates, closes, adj=True, tz=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if tz:
        idx = idx.tz_localize(tz)
    df = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                       "Volume": [1000] * len(closes)}, index=idx)
    if adj:
        df["Adj Close"] = [c * 0.98 for c in closes]
    return df


def statements(eps=(2.0, 1.0), shares=(1e9, 1.1e9), ni=(2e9, 1.1e9), years=2):
    """A yfinance-style annual income statement + balance sheet, newest year first."""
    cols = [pd.Timestamp(f"{2025 - k}-12-31") for k in range(years)]
    income = pd.DataFrame({c: {"Net Income Common Stockholders": n, "Diluted EPS": e, "Diluted Average Shares": s}
                           for c, e, s, n in zip(cols, eps, shares, ni)})
    balance = pd.DataFrame({c: {"Common Stock Equity": 5e9 if k == 0 else float("nan")}
                            for k, c in enumerate(cols)})
    return income, balance


class UniverseTest(unittest.TestCase):

    def test_yahoo_symbol(self):
        self.assertEqual(fd.yahoo_symbol("BRK.B"), "BRK-B")
        self.assertEqual(fd.yahoo_symbol(" bf.b "), "BF-B")
        self.assertEqual(fd.yahoo_symbol("AAPL"), "AAPL")

    def test_fallback_list_is_clean(self):
        tickers = [t for t, _, _ in fd.FALLBACK_UNIVERSE]
        self.assertEqual(len(tickers), 100)
        self.assertEqual(len(set(tickers)), 100)
        self.assertTrue(all(t == fd.yahoo_symbol(t) for t in tickers))

    def test_allocation_is_proportional_and_exact(self):
        counts = pd.Series({"A": 50, "B": 30, "C": 20, "D": 3})
        alloc = fd.allocate_by_sector(counts, 20)
        self.assertEqual(int(alloc.sum()), 20)
        self.assertEqual(dict(alloc), {"A": 10, "B": 6, "C": 4, "D": 0})
        self.assertEqual(int(fd.allocate_by_sector(counts, 500).sum()), 103)   # capped at the universe

    def universe(self, n=60):
        sectors = ["Tech", "Health", "Energy"]
        return pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(n)],
                             "name": [f"Co {i}" for i in range(n)],
                             "sector": [sectors[i % 3] for i in range(n)],
                             "added": [date(2000, 1, 1)] * n})

    def test_stratified_sample_is_deterministic_and_balanced(self):
        u = self.universe()
        a, b = fd.stratified_sample(u, 15, seed=42), fd.stratified_sample(u, 15, seed=42)
        self.assertEqual(a["ticker"].tolist(), b["ticker"].tolist())
        self.assertEqual(len(a), 15)
        self.assertEqual(a["sector"].value_counts().to_dict(), {"Tech": 5, "Health": 5, "Energy": 5})
        self.assertNotEqual(a["ticker"].tolist(), fd.stratified_sample(u, 15, seed=7)["ticker"].tolist())

    def test_stratified_sample_is_stable_when_the_index_changes(self):
        u = self.universe(60)
        base = set(fd.stratified_sample(u, 15, seed=42)["ticker"])
        # Swap one constituent for a newcomer in the same sector: at most one pick may change.
        changed = pd.concat([u.iloc[1:], pd.DataFrame([{"ticker": "NEW", "name": "New", "sector": u.iloc[0]["sector"],
                                                         "added": date(2000, 1, 1)}])], ignore_index=True)
        after = set(fd.stratified_sample(changed, 15, seed=42)["ticker"])
        self.assertLessEqual(len(base - after), 1)

    def test_start_members_drops_later_additions(self):
        u = pd.DataFrame({"ticker": ["OLD", "NEW", "UNKNOWN"], "name": ["", "", ""], "sector": ["X"] * 3,
                          "added": [date(1990, 1, 1), date(2025, 3, 24), None]})
        kept = fd.start_members(u, date(2024, 9, 30))
        self.assertEqual(kept["ticker"].tolist(), ["OLD"])
        # Unknown dates pass only when explicitly allowed (the hardcoded fallback list) ...
        self.assertEqual(len(fd.start_members(fd.fallback_universe(), date(2024, 9, 30), allow_unknown=True)), 100)
        # ... never by inference: a scrape that lost its dates must not switch the filter off.
        self.assertEqual(len(fd.start_members(fd.fallback_universe(), date(2024, 9, 30))), 0)

    def page(self, header="Date added", date_fmt="1999-01-0{d}"):
        rows = "".join(f"<tr><td>T{i}.B</td><td>Co {i}</td><td>Tech</td><td>{date_fmt.format(d=i % 9 + 1)}</td></tr>"
                       for i in range(450))
        html = ('<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>'
                f'<th>{header}</th></tr>{rows}</table>')
        return mock.Mock(text=html, status_code=200, raise_for_status=lambda: None)

    def test_scrape_parses_table_and_keeps_inclusion_dates(self):
        with mock.patch.object(fd.requests, "get", return_value=self.page()):
            out = fd.scrape_sp500()
        self.assertEqual(len(out), 450)
        self.assertEqual(out.iloc[0]["ticker"], "T0-B")
        self.assertEqual(out.iloc[0]["added"], date(1999, 1, 1))

    def test_scrape_finds_a_renamed_date_column_and_strips_footnotes(self):
        page = self.page(header="Date first added", date_fmt="2001-02-0{d} [3]")
        with mock.patch.object(fd.requests, "get", return_value=page):
            out = fd.scrape_sp500()
        self.assertEqual(out.iloc[0]["added"], date(2001, 2, 1))

    def test_scrape_fails_loudly_without_usable_dates(self):
        for page in (self.page(header="Founded", date_fmt="1999"),        # no 'added' column
                     self.page(date_fmt="March {d}, 1999")):              # unparseable format
            with mock.patch.object(fd.requests, "get", return_value=page), mock.patch.object(fd.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    fd.scrape_sp500(retries=1)

    def args(self, **kw):
        return SimpleNamespace(**{"tickers": None, "universe": "start-members", "n_tickers": 100, "seed": 42, **kw})

    def test_failed_scrape_reuses_the_stored_universe(self):
        stored = pd.DataFrame({"ticker": ["AAA", "BBB", "NEW"], "name": ["A", "B", "N"], "sector": ["X"] * 3,
                               "added": [date(2000, 1, 1), date(2010, 1, 1), date(2025, 6, 1)]})
        with mock.patch.object(fd, "scrape_sp500", side_effect=RuntimeError("down")):
            got = fd.choose_universe(self.args(), date(2024, 9, 30), stored)
            self.assertEqual(got["ticker"].tolist(), ["AAA", "BBB"])        # still filtered to start members
            fallback = fd.choose_universe(self.args(), date(2024, 9, 30), stored.iloc[0:0])
        self.assertEqual(len(fallback), 100)                                  # nothing stored: hardcoded list

    def test_explicit_tickers_may_not_include_benchmark_series(self):
        with self.assertRaises(SystemExit):
            fd.choose_universe(self.args(tickers="AAPL,RSP"), date(2024, 9, 30), pd.DataFrame())

    def test_scrape_failure_raises_after_retries(self):
        with mock.patch.object(fd.requests, "get", side_effect=fd.requests.ConnectionError("down")) as get, \
                mock.patch.object(fd.time, "sleep"):
            with self.assertRaises(RuntimeError):
                fd.scrape_sp500(retries=3)
        self.assertEqual(get.call_count, 3)

    def test_scrape_rejects_a_truncated_table(self):
        html = ('<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>'
                '<tr><td>A</td><td>A</td><td>X</td></tr></table>')
        resp = mock.Mock(text=html, raise_for_status=lambda: None)
        with mock.patch.object(fd.requests, "get", return_value=resp), mock.patch.object(fd.time, "sleep"):
            with self.assertRaises(RuntimeError):
                fd.scrape_sp500(retries=1)


class DatesTest(unittest.TestCase):

    def et(self, *args):
        return pd.Timestamp(datetime(*args), tz="America/New_York").to_pydatetime()

    def test_current_month_counts_only_after_its_last_close(self):
        self.assertEqual(fd.last_complete_month_end(self.et(2026, 9, 30, 23, 0)), date(2026, 9, 30))
        self.assertEqual(fd.last_complete_month_end(self.et(2026, 9, 30, 10, 0)), date(2026, 8, 31))
        self.assertEqual(fd.last_complete_month_end(self.et(2026, 9, 15, 20, 0)), date(2026, 8, 31))
        self.assertEqual(fd.last_complete_month_end(self.et(2026, 10, 1, 9, 0)), date(2026, 9, 30))

    def test_month_ending_on_a_weekend(self):
        # May 2025 ends on a Saturday; its last session is Friday the 30th.
        self.assertEqual(fd.last_complete_month_end(self.et(2025, 5, 30, 12, 0)), date(2025, 4, 30))
        self.assertEqual(fd.last_complete_month_end(self.et(2025, 5, 30, 19, 0)), date(2025, 5, 31))
        self.assertEqual(fd.last_complete_month_end(self.et(2025, 5, 31, 9, 0)), date(2025, 5, 31))

    def test_end_must_be_a_completed_month_end(self):
        now = self.et(2026, 10, 6, 12, 0)
        self.assertEqual(fd.resolve_end(None, now), date(2026, 9, 30))
        self.assertEqual(fd.resolve_end(date(2026, 6, 30), now), date(2026, 6, 30))
        for bad in (date(2026, 6, 15), date(2026, 10, 31)):            # mid-month; not over yet
            with self.assertRaises(SystemExit):
                fd.resolve_end(bad, now)

    def test_price_window_covers_warmup_and_backtest(self):
        self.assertEqual(fd.price_window(date(2026, 9, 30), 24), (date(2023, 9, 1), date(2026, 10, 1)))
        self.assertEqual(fd.first_rank_date(date(2026, 9, 30), 24), date(2024, 9, 30))
        self.assertEqual(fd.first_rank_date(date(2025, 2, 28), 12), date(2024, 2, 29))


class PricesTest(unittest.TestCase):
    start, end = date(2024, 1, 1), date(2024, 2, 1)

    def test_tidy_prices_cleans_rows(self):
        f = frame(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05", "2024-02-01"],
                  [10.0, 11.0, 11.5, 0.0, float("nan"), 12.0], tz="America/Chicago")
        out = fd.tidy_prices(f, self.start, self.end)
        self.assertEqual(list(out.index), [date(2024, 1, 2), date(2024, 1, 3)])     # dupes, 0, NaN, window
        self.assertEqual(out.loc[date(2024, 1, 3), "close"], 11.5)                  # last duplicate wins
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "adj_close", "volume"])

    def test_tidy_prices_without_adj_close_falls_back_to_close(self):
        out = fd.tidy_prices(frame(["2024-01-02"], [10.0], adj=False), self.start, self.end)
        self.assertEqual(out.iloc[0]["adj_close"], 10.0)

    def test_tidy_prices_handles_none_and_empty(self):
        self.assertTrue(fd.tidy_prices(None, self.start, self.end).empty)
        self.assertTrue(fd.tidy_prices(pd.DataFrame(), self.start, self.end).empty)

    def test_split_download_multiindex_and_flat(self):
        a, b = frame(["2024-01-02"], [1.0]), frame(["2024-01-02"], [2.0])
        multi = pd.concat({"AAA": a, "BBB": b}, axis=1)
        got = fd._split_download(multi, ["AAA", "BBB", "CCC"])
        self.assertEqual(sorted(got), ["AAA", "BBB"])
        self.assertEqual(sorted(fd._split_download(a, ["AAA"])), ["AAA"])
        self.assertEqual(fd._split_download(pd.DataFrame(), ["AAA"]), {})

    def test_download_survives_a_failed_batch_and_a_dead_ticker(self):
        good = frame(["2024-01-02", "2024-01-03"], [10.0, 11.0])

        class FakeTicker:
            def __init__(self, sym):
                self.sym = sym

            def history(self, **kwargs):
                if self.sym == "DEAD":
                    raise ValueError("possibly delisted")
                return good

        with mock.patch.object(fd.yf, "download", side_effect=RuntimeError("rate limited")), \
                mock.patch.object(fd.yf, "Ticker", FakeTicker), mock.patch.object(fd.time, "sleep"):
            frames, failed = fd.download_prices(["AAA", "DEAD", "BBB"], self.start, self.end)
        self.assertEqual(sorted(frames), ["AAA", "BBB"])
        self.assertEqual(failed, ["DEAD"])

    def test_download_retries_only_the_missing_tickers(self):
        batch = pd.concat({"AAA": frame(["2024-01-02"], [10.0])}, axis=1)
        calls = []

        class FakeTicker:
            def __init__(self, sym):
                calls.append(sym)

            def history(self, **kwargs):
                return frame(["2024-01-02"], [5.0])

        with mock.patch.object(fd.yf, "download", return_value=batch), \
                mock.patch.object(fd.yf, "Ticker", FakeTicker), mock.patch.object(fd.time, "sleep"):
            frames, failed = fd.download_prices(["AAA", "BBB"], self.start, self.end)
        self.assertEqual(calls, ["BBB"])
        self.assertEqual((sorted(frames), failed), (["AAA", "BBB"], []))


class FundamentalsTest(unittest.TestCase):

    def test_snapshot_uses_market_cap_over_price_for_shares(self):
        info = {"currentPrice": 200.0, "marketCap": 1e12, "trailingPE": 25.0, "trailingEps": 8.0,
                "priceToBook": "Infinity", "sharesOutstanding": 1e9,
                "regularMarketTime": pd.Timestamp("2026-09-30 16:00", tz="America/New_York").timestamp()}
        row = fd.parse_snapshot("BRK-B", info, date(2026, 1, 1))
        ticker, as_of, source, _, mcap, pe, ptb, shares, eps, _, _ = row
        self.assertEqual((ticker, source, mcap, pe, ptb, shares, eps),
                         ("BRK-B", "snapshot", 1e12, 25.0, None, 5e9, 8.0))
        self.assertEqual(as_of, date(2026, 9, 30))      # regularMarketTime, in New York time

    def test_snapshot_garbage_and_negative_pe(self):
        self.assertIsNone(fd.parse_snapshot("X", {}, date(2026, 1, 1)))
        row = fd.parse_snapshot("X", {"regularMarketPrice": 10.0, "sharesOutstanding": 100.0,
                                      "trailingPE": -4.0, "trailingEps": None}, date(2026, 1, 1))
        self.assertEqual(row[1], date(2026, 1, 1))                 # no timestamp -> fallback date
        self.assertEqual((row[4], row[5], row[7], row[8]), (1000.0, None, 100.0, None))

    def test_annual_reports_are_dated_when_public(self):
        income, balance = statements()
        closes = pd.Series([40.0, 50.0], index=[date(2026, 4, 29), date(2026, 5, 15)])
        rows = fd.parse_annual_reports("X", income, balance, closes)
        latest = next(r for r in rows if r[3] == date(2025, 12, 31))
        self.assertEqual(latest[1], date(2026, 4, 30))              # FYE + 4 months
        self.assertEqual(latest[2], "annual_report")
        self.assertAlmostEqual(latest[4], 1e9 * 40.0)               # close on/before the as-of date
        self.assertAlmostEqual(latest[5], 40e9 / 2e9)               # P/E
        self.assertAlmostEqual(latest[6], 40e9 / 5e9)               # P/B
        older = next(r for r in rows if r[3] == date(2024, 12, 31))
        self.assertEqual(older[1], date(2025, 4, 30))
        self.assertIsNone(older[4])                                 # no close that early
        self.assertIsNone(older[10])                                # NaN book equity -> None

    def test_eps_inconsistent_with_net_income_is_replaced(self):
        income, balance = statements(eps=(200.0, 1.0))         # e.g. per-class-A EPS
        rows = fd.parse_annual_reports("X", income, balance, None)
        self.assertAlmostEqual(next(r for r in rows if r[3] == date(2025, 12, 31))[8], 2.0)

    def by_year(self, rows):
        return {r[3].year: r for r in rows}

    def test_glitched_count_next_to_a_correct_eps_is_recovered_point_in_time(self):
        # TRV FY2024: share count reported ~100x too high, EPS correct -> NI / EPS.
        income, balance = statements(shares=(1e11, 1.1e9), eps=(2.0, 1.0))
        rows = self.by_year(fd.parse_annual_reports("X", income, balance, None, reference_shares=0.95e9))
        self.assertAlmostEqual(rows[2025][7], 2e9 / 2.0)             # the year's own count, not today's
        self.assertAlmostEqual(rows[2025][8], 2.0)
        self.assertEqual(rows[2024][7], 1.1e9)                       # within 2x: untouched

    def test_a_single_year_glitch_is_caught_without_a_reference(self):
        income, balance = statements(shares=(1e11, 1.1e9, 1.05e9), eps=(2.0, 1.0, 1.0), ni=(2e9, 1.1e9, 1.05e9),
                                     years=3)
        rows = self.by_year(fd.parse_annual_reports("X", income, balance, None))   # yardstick = median
        self.assertAlmostEqual(rows[2025][7], 1e9)

    def test_every_year_glitched_falls_back_to_the_reference(self):
        # DLR: count AND EPS both 4x off in every year; only today's count is right.
        income, balance = statements(shares=(4e9, 4.4e9), eps=(0.5, 0.25), ni=(2e9, 1.1e9))
        rows = self.by_year(fd.parse_annual_reports("X", income, balance, None, reference_shares=1e9))
        self.assertEqual((rows[2025][7], rows[2024][7]), (1e9, 1e9))
        self.assertAlmostEqual(rows[2025][8], 2.0)                   # EPS re-derived from net income

    def test_missing_count_is_filled_from_the_earlier_year(self):
        # CLX FY2025: net income present, shares and EPS blank -> previous year's count.
        income, balance = statements(shares=(1.2e9, float("nan"), 1e9), eps=(2.0, float("nan"), 1.0),
                                     ni=(2.4e9, 1.5e9, 1e9), years=3)
        rows = self.by_year(fd.parse_annual_reports("X", income, balance, None, reference_shares=1.2e9))
        self.assertEqual(set(rows), {2023, 2024, 2025})              # the year is kept, not dropped
        self.assertEqual(rows[2024][7], 1e9)                         # FY2023's count (earlier first)
        self.assertAlmostEqual(rows[2024][8], 1.5)

    def test_fiscal_year_gaps(self):
        rows = [("X", None, "annual_report", date(y, 6, 30)) + (None,) * 7 for y in (2022, 2023, 2025)]
        self.assertEqual(fd.fiscal_year_gaps(rows), ["2023-06-30 -> 2025-06-30"])

    def test_reference_shares_ignores_single_class_counts(self):
        self.assertEqual(fd.reference_shares({"currentPrice": 10.0, "marketCap": 1e10}), 1e9)
        self.assertEqual(fd.reference_shares({"impliedSharesOutstanding": 2e9}), 2e9)
        self.assertIsNone(fd.reference_shares({"sharesOutstanding": 5e8}))     # GOOG: one class only

    def test_losses_and_missing_data(self):
        income, balance = statements(eps=(-1.0, float("nan")), ni=(-1e9, float("nan")),
                                          shares=(1e9, float("nan")))
        rows = fd.parse_annual_reports("X", income, balance, pd.Series([10.0], index=[date(2026, 4, 30)]))
        self.assertEqual(len(rows), 1)                              # the all-NaN year is skipped
        self.assertIsNone(rows[0][5])                               # no P/E on a loss
        self.assertEqual(rows[0][8], -1.0)
        self.assertEqual(fd.parse_annual_reports("X", pd.DataFrame(), None, None), [])
        self.assertEqual(fd.parse_annual_reports("X", None, None, None), [])

    def test_fetch_fundamentals_tolerates_failures(self):
        income, balance = statements()

        class FakeTicker:
            def __init__(self, sym):
                self.sym = sym

            @property
            def info(self):
                if self.sym == "NOINFO":
                    raise ValueError("404")
                return {"currentPrice": 10.0, "marketCap": 1e10}

            @property
            def income_stmt(self):
                return pd.DataFrame() if self.sym == "NOSTMT" else income

            @property
            def balance_sheet(self):
                return balance

        with mock.patch.object(fd.yf, "Ticker", FakeTicker), mock.patch.object(fd.time, "sleep"):
            rows, no_snap, no_reports = fd.fetch_fundamentals(["OK", "NOINFO", "NOSTMT"], date(2026, 9, 30), {})
        self.assertEqual(no_snap, ["NOINFO"])
        self.assertEqual(no_reports, ["NOSTMT"])
        self.assertEqual(sorted({r[0] for r in rows}), ["NOINFO", "NOSTMT", "OK"])

    def test_failed_quote_does_not_switch_off_the_glitch_guard(self):
        # Every year 4x off (DLR) and today's quote fails: the stored snapshot's
        # count is the yardstick, so the bad counts can't overwrite good rows.
        income, balance = statements(shares=(4e9, 4.4e9), eps=(0.5, 0.25), ni=(2e9, 1.1e9))

        class FakeTicker:
            def __init__(self, sym):
                self.sym = sym

            @property
            def info(self):
                raise ValueError("rate limited")

            income_stmt = property(lambda self: income)
            balance_sheet = property(lambda self: balance)

        with mock.patch.object(fd.yf, "Ticker", FakeTicker), mock.patch.object(fd.time, "sleep"):
            rows, no_snap, _ = fd.fetch_fundamentals(["DLR"], date(2026, 9, 30), {}, stored_shares={"DLR": 1e9})
        self.assertEqual(no_snap, ["DLR"])
        self.assertEqual({r[7] for r in rows}, {1e9})

    def test_one_malformed_statement_does_not_sink_the_run(self):
        bad = pd.DataFrame({"not a date": {"Net Income": 1.0}})      # pd.Timestamp("not a date") raises

        class FakeTicker:
            def __init__(self, sym):
                self.sym = sym

            info = property(lambda self: {"currentPrice": 10.0, "marketCap": 1e10})
            income_stmt = property(lambda self: bad if self.sym == "BAD" else statements()[0])
            balance_sheet = property(lambda self: statements()[1])

        with mock.patch.object(fd.yf, "Ticker", FakeTicker), mock.patch.object(fd.time, "sleep"):
            rows, _, no_reports = fd.fetch_fundamentals(["BAD", "OK"], date(2026, 9, 30), {})
        self.assertEqual(no_reports, ["BAD"])
        self.assertTrue(any(r[0] == "OK" and r[2] == "annual_report" for r in rows))


if __name__ == "__main__":
    unittest.main()
