"""Offline tests for backtest.py: metrics on hand-computed series, the
forward-return alignment (no look-ahead), portfolio formation and the
Fama-French parsing. No database or network."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest as bt  # noqa: E402

MONTHS = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-30"])


def monthly(prices: dict, kind="stock") -> pd.DataFrame:
    """{ticker: [adj_close per month]} -> monthly_prices-shaped long frame."""
    rows = [{"ticker": t, "month_end": d, "close": p, "adj_close": p, "kind": kind}
            for t, series in prices.items() for d, p in zip(MONTHS, series)]
    return pd.DataFrame(rows)


class MetricsTest(unittest.TestCase):

    def test_max_drawdown_counts_the_first_month(self):
        self.assertAlmostEqual(bt.max_drawdown(pd.Series([-0.10, 0.05])), -0.10)
        self.assertAlmostEqual(bt.max_drawdown(pd.Series([0.10, -0.20, 0.30])), -0.20)
        self.assertEqual(bt.max_drawdown(pd.Series([0.01, 0.02])), 0.0)
        # peak 1.1*1.1 = 1.21, trough 1.21*0.9*0.9 = 0.9801 -> -19%
        self.assertAlmostEqual(bt.max_drawdown(pd.Series([0.1, 0.1, -0.1, -0.1, 0.5])), -0.19)

    def test_performance_matches_hand_calculation(self):
        r = pd.Series([0.02, -0.01, 0.03, 0.00, 0.01, 0.04])
        rf = pd.Series([0.001] * 6)
        p = bt.performance(r, rf)
        growth = 1.02 * 0.99 * 1.03 * 1.00 * 1.01 * 1.04
        self.assertAlmostEqual(p["cumulative_return"], growth - 1)
        self.assertAlmostEqual(p["annualized_return"], growth ** 2 - 1)          # 6 months -> x2
        self.assertAlmostEqual(p["annualized_volatility"], r.std(ddof=1) * math.sqrt(12))
        excess = r - 0.001
        sr_m = excess.mean() / excess.std(ddof=1)
        self.assertAlmostEqual(p["sharpe_ratio"], sr_m * math.sqrt(12))
        self.assertAlmostEqual(p["sharpe_std_error"], math.sqrt((1 + sr_m ** 2 / 2) / 6) * math.sqrt(12))
        self.assertEqual(p["months"], 6)

    def test_relative_metrics(self):
        b = pd.Series([0.01, -0.02, 0.03, 0.015])
        r = 2 * b
        rel = bt.relative(r, b)
        self.assertAlmostEqual(rel["beta"], 2.0)
        self.assertAlmostEqual(rel["tracking_error"], b.std(ddof=1) * math.sqrt(12))
        self.assertAlmostEqual(rel["active_return_annualized"], b.mean() * 12)
        self.assertEqual(rel["months_outperformed"], 3)


class AlignmentTest(unittest.TestCase):

    def test_forward_return_is_next_month(self):
        m = monthly({"AAA": [100, 110, 99, 99, 120]})
        fwd, bench, rf = bt.forward_returns(pd.concat([m, monthly({"^GSPC": [10, 11, 12, 13, 14]}, "index")]))
        self.assertAlmostEqual(fwd.at[MONTHS[0], "AAA"], 0.10)        # Jan row = Jan->Feb return
        self.assertAlmostEqual(fwd.at[MONTHS[1], "AAA"], -0.10)
        self.assertTrue(math.isnan(fwd.at[MONTHS[4], "AAA"]))         # nothing after the last month
        self.assertAlmostEqual(bench.at[MONTHS[0], "^GSPC"], 0.10)
        self.assertTrue((rf == 0).all())                              # no ^IRX -> 0% risk-free

    def test_changing_the_future_does_not_change_the_past(self):
        base = monthly({"AAA": [100, 110, 99, 99, 120], "BBB": [50, 50, 55, 60, 50]})
        bumped = base.copy()
        bumped.loc[bumped["month_end"] == MONTHS[4], "adj_close"] *= 3
        a, _, _ = bt.forward_returns(base)
        b, _, _ = bt.forward_returns(bumped)
        pd.testing.assert_frame_equal(a.loc[:MONTHS[2]], b.loc[:MONTHS[2]])   # rows up to t-2 untouched
        self.assertFalse(np.allclose(a.loc[MONTHS[3]], b.loc[MONTHS[3]]))      # only t-1's forward return moves

    def test_risk_free_from_irx(self):
        irx = monthly({"^IRX": [4.8, 4.8, 4.2, 4.2, 4.2]}, "rate")
        _, _, rf = bt.forward_returns(pd.concat([monthly({"AAA": [1, 2, 3, 4, 5]}), irx]))
        self.assertAlmostEqual(rf.iloc[0], 0.048 / 12)

    def test_gap_in_month_ends_stops_the_backtest(self):
        m = monthly({"AAA": [1, 2, 3, 4, 5]})
        with self.assertRaises(SystemExit):
            bt.forward_returns(m[m["month_end"] != MONTHS[2]])

    def test_next_month_end(self):
        self.assertEqual(bt.next_month_end(pd.Timestamp("2025-05-30")), pd.Timestamp("2025-06-30"))
        self.assertEqual(bt.next_month_end(pd.Timestamp("2024-01-31")), pd.Timestamp("2024-02-29"))
        self.assertEqual(bt.next_month_end(pd.Timestamp("2025-12-31")), pd.Timestamp("2026-01-31"))


class PortfolioTest(unittest.TestCase):

    def scores(self, composite: dict, quintile1: set, t=MONTHS[0], momentum=0.5):
        return pd.DataFrame([{"ticker": k, "month_end": t, "composite_score": v,
                              "composite_quintile": 1 if k in quintile1 else 2,
                              "momentum_z": momentum, "size_z": v, "value_z": -v}
                             for k, v in composite.items()])

    def test_top_quintile_is_ceil_n_over_5_with_ticker_tiebreak(self):
        s = pd.Series({"A": 1.0, "B": 3.0, "C": 3.0, "D": 0.5, "E": -1.0, "F": 2.0, "G": float("nan")})
        self.assertEqual(bt.top_quintile(s), ["B", "C"])           # 6 scored -> ceil(6/5) = 2
        self.assertEqual(bt.top_quintile(pd.Series({"A": 1.0})), ["A"])

    def test_portfolio_is_equal_weighted_forward_return(self):
        comp = {f"S{i}": float(i) for i in range(10)}                 # top quintile = S9, S8
        sc = self.scores(comp, {"S9", "S8"})
        fwd = pd.DataFrame({f"S{i}": [0.01 * i] for i in range(10)}, index=[MONTHS[0]])
        rets, holdings = bt.run_backtest(sc, fwd, [MONTHS[0]])
        self.assertEqual(holdings[MONTHS[0]], ["S9", "S8"])
        self.assertAlmostEqual(rets.at[MONTHS[0], "Factor portfolio"], (0.09 + 0.08) / 2)
        self.assertAlmostEqual(rets.at[MONTHS[0], "Equal-weight universe"], 0.045)
        self.assertAlmostEqual(rets.at[MONTHS[0], "Value only"], (0.00 + 0.01) / 2)   # value_z = -composite

    def test_disagreement_with_sql_quintile_is_an_error(self):
        sc = self.scores({f"S{i}": float(i) for i in range(10)}, {"S0", "S1"})
        fwd = pd.DataFrame({f"S{i}": [0.0] for i in range(10)}, index=[MONTHS[0]])
        with self.assertRaises(AssertionError):
            bt.run_backtest(sc, fwd, [MONTHS[0]])

    def test_rebalance_dates_need_momentum_and_a_next_month(self):
        sc = pd.concat([self.scores({"A": 1.0, "B": 0.0}, {"A"}, t=t, momentum=(None if i == 0 else 0.3))
                        for i, t in enumerate(MONTHS)])
        fwd = pd.DataFrame(0.0, index=MONTHS, columns=["A", "B"])
        dates = bt.rebalance_dates(sc, fwd, None)
        self.assertEqual(dates, list(MONTHS[1:4]))       # Jan: no momentum; May: no t+1
        self.assertEqual(bt.rebalance_dates(sc, fwd, 2), list(MONTHS[2:4]))
        # A month in the middle without momentum must stop the run, not leave a hole.
        holed = sc.copy()
        holed.loc[holed["month_end"] == MONTHS[2], "momentum_z"] = None
        with self.assertRaises(SystemExit):
            bt.rebalance_dates(holed, fwd, None)

    def test_incomplete_total_return_series_falls_back_to_price_index(self):
        dates = list(MONTHS[1:4])
        bench = pd.DataFrame({"^SP500TR": 0.01, "^GSPC": 0.009}, index=MONTHS)
        self.assertEqual(bt.choose_benchmark(dates, bench), "^SP500TR")
        bench.loc[MONTHS[0], "^SP500TR"] = float("nan")          # a hole OUTSIDE the test window
        self.assertEqual(bt.choose_benchmark(dates, bench), "^SP500TR")
        bench.loc[MONTHS[2], "^SP500TR"] = float("nan")          # a hole inside it
        self.assertEqual(bt.choose_benchmark(dates, bench), "^GSPC")
        self.assertEqual(bt.choose_benchmark(dates, bench.drop(columns="^SP500TR")), "^GSPC")
        with self.assertRaises(SystemExit):
            bt.choose_benchmark(dates, bench.drop(columns=["^SP500TR", "^GSPC"]))

    def test_performance_table_keeps_the_benchmark_in_both_modes(self):
        label, cols = bt.performance_columns("^SP500TR", has_rsp=True)
        self.assertEqual(label, "S&P 500 TR")
        self.assertEqual(list(cols), ["Factor portfolio", "S&P 500 TR", "S&P 500 price", "EW universe",
                                      "S&P 500 EW (RSP)"])
        label, cols = bt.performance_columns("^GSPC", has_rsp=False)
        self.assertEqual(label, "S&P 500 price")
        self.assertEqual(cols, {"Factor portfolio": "Factor portfolio", "S&P 500 price": "Benchmark",
                                "EW universe": "Equal-weight universe"})

    def test_turnover(self):
        fwd = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [1.0, 0.0]}, index=MONTHS[:2])
        same = bt.turnover({MONTHS[0]: ["A", "B"], MONTHS[1]: ["A", "B"]}, fwd)
        self.assertAlmostEqual(same.iloc[0], 0.0)
        swap = bt.turnover({MONTHS[0]: ["A", "B"], MONTHS[1]: ["C", "B"]}, fwd)
        self.assertAlmostEqual(swap.iloc[0], 0.5)
        # Same names, but C doubled: weights drifted to A 1/3, C 2/3 and are reset to 1/2 each.
        drift = bt.turnover({MONTHS[0]: ["A", "C"], MONTHS[1]: ["A", "C"]}, fwd)
        self.assertAlmostEqual(drift.iloc[0], 0.5 * (abs(0.5 - 1 / 3) + abs(0.5 - 2 / 3)))


class DailyPathTest(unittest.TestCase):
    """Daily growth-of-$1: same end point as monthly compounding, but it sees
    the intra-month low that month-end sampling misses."""

    def setUp(self):
        days = pd.to_datetime(["2025-01-31", "2025-02-10", "2025-02-28", "2025-03-14", "2025-03-31"])
        self.daily = pd.DataFrame({"A": [100, 60, 110, 110, 121],     # crashes mid-February, recovers
                                   "B": [50, 50, 55, 44, 55]}, index=days, dtype=float)
        self.ends = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
        self.members = {self.ends[0]: ["A", "B"], self.ends[1]: ["A"]}

    def test_daily_path_matches_monthly_compounding_at_month_ends(self):
        path = bt.daily_wealth(self.members, self.daily, self.ends)
        feb = (110 / 100 + 55 / 50) / 2                              # equal weight, buy-and-hold
        self.assertAlmostEqual(path[self.ends[1]], feb)
        self.assertAlmostEqual(path[self.ends[2]], feb * 121 / 110)  # March: A only
        self.assertAlmostEqual(path.iloc[0], 1.0)

    def test_daily_drawdown_sees_the_intra_month_crash(self):
        path = bt.daily_wealth(self.members, self.daily, self.ends)
        monthly = pd.Series([path[self.ends[1]] - 1, path[self.ends[2]] / path[self.ends[1]] - 1])
        self.assertEqual(bt.max_drawdown(monthly), 0.0)              # month-ends only ever rise
        self.assertAlmostEqual(bt.path_drawdown(path), (60 / 100 + 50 / 50) / 2 - 1)   # -20% on Feb 10

    def test_index_wealth_rebases_to_one(self):
        w = bt.index_wealth(self.daily["B"], self.ends[0], self.ends[2])
        self.assertAlmostEqual(w.iloc[0], 1.0)
        self.assertAlmostEqual(w.iloc[-1], 1.1)

    def test_sanity_check_catches_a_daily_path_that_disagrees(self):
        returns = pd.DataFrame({"Factor portfolio": [0.10, 0.05], "Benchmark": [0.01, 0.02]}, index=self.ends[:2])
        perf = {"x": bt.performance(returns["Factor portfolio"], pd.Series(0.0, index=self.ends[:2]))}
        good = {"Factor portfolio": pd.Series([1.0, 1.1, 1.155])}
        bad = {"Factor portfolio": pd.Series([1.0, 1.1, 1.30])}
        self.assertEqual(bt.sanity_checks(returns, perf, good), [])
        self.assertEqual(len(bt.sanity_checks(returns, perf, bad)), 1)


class FamaFrenchTest(unittest.TestCase):
    TEXT = """This file was created using the 202608 CRSP database.
The 1-month TBill rate ...

,Mkt-RF,SMB,HML,RF
202606,  -1.07,   3.39,   3.58,   0.29
202607,  -0.61,  -1.92,   2.11,   0.33
202608,   2.56,   0.34,  -3.54,   0.29

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  2025,  17.00,  -1.00,   2.00,   4.20
"""

    def test_parse_monthly_block_only(self):
        df = bt.parse_ff_csv(self.TEXT, ["Mkt-RF", "SMB", "HML", "RF"])
        self.assertEqual([str(p) for p in df.index], ["2026-06", "2026-07", "2026-08"])
        self.assertAlmostEqual(df.at[pd.Period("2026-08", "M"), "HML"], -0.0354)

    def test_ols_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
        y = X @ np.array([0.5, 2.0, -1.0]) + rng.normal(scale=0.01, size=200)
        beta, t, r2 = bt.ols(y, X)
        np.testing.assert_allclose(beta, [0.5, 2.0, -1.0], atol=0.01)
        self.assertGreater(r2, 0.999)
        self.assertTrue((np.abs(t) > 100).all())

    def test_regression_keys_returns_to_the_month_earned(self):
        idx = pd.to_datetime([f"2025-{m:02d}-28" for m in range(1, 13)]) + pd.offsets.MonthEnd(0)
        rng = np.random.default_rng(1)
        ff = pd.DataFrame(rng.normal(0, 0.03, size=(14, 5)), columns=["Mkt-RF", "SMB", "HML", "RF", "Mom"],
                          index=pd.period_range("2025-01", periods=14, freq="M"))
        ff["RF"] = 0.003
        # Portfolio return ranked at Jan-end is EARNED in February: make it exactly Feb's Mkt-RF + RF.
        earned = ff.shift(-1).iloc[:12]
        rets = pd.DataFrame({"Factor portfolio": earned["Mkt-RF"].to_numpy() + 0.003,
                             "Equal-weight universe": (earned["Mkt-RF"] + 0.5 * earned["SMB"]).to_numpy() + 0.003},
                            index=idx)
        reg = bt.factor_regressions(rets, ff)
        self.assertAlmostEqual(reg.at["Factor portfolio", "mkt"], 1.0, places=9)
        self.assertAlmostEqual(reg.at["Factor portfolio", "r2"], 1.0, places=9)
        self.assertEqual(reg.at["Factor portfolio", "first_month"], "2025-02")
        self.assertAlmostEqual(reg.at["Factor - EW universe", "smb"], -0.5, places=9)


if __name__ == "__main__":
    unittest.main()
