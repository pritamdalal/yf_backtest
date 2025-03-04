import pytest
import numpy as np
import pandas as pd
import datetime
from FixedWieightBacktester import FixedWeightBacktester
from MarketCorrections import MarketCorrections


@pytest.fixture
def price_test_data() -> pd.DataFrame:
    df_spy = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "spy")
    df_agg = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "agg")
    df_hyg = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "hyg")
    df_tlt = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "tlt")
    df_gld = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "gld")
    df_buffer_010 = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "mqu1bslq")
    df_buffer_020 = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "mquslblr")
    df_buffer_100 = pd.read_excel("data/bufr_bufd_mquslblr.xlsx", "mqu1pplr")
    df_sv_hedged_income = pd.read_excel(
        "data/bufr_bufd_mquslblr.xlsx",
        "sv_hedged_income"
    )
    df_sv_hedged_balanced = pd.read_excel(
        "data/bufr_bufd_mquslblr.xlsx",
        "sv_hedged_balanced"
    )
    df_sv_hedged_enhanced_growth = pd.read_excel(
        "data/bufr_bufd_mquslblr.xlsx",
        "sv_hedged_enhanced_growth"
    )
    df_sv_equity_buffer = pd.read_excel(
        "data/bufr_bufd_mquslblr.xlsx",
        "sv_equity_buffer"
    )
    df_sv_equity_buffer_growth = pd.read_excel(
        "data/bufr_bufd_mquslblr.xlsx",
        "sv_equity_buffer_growth"
    )

    df_px = (
        df_buffer_100
        .merge(df_buffer_010, how="left", on="date")
        .merge(df_buffer_020, how="left", on="date")
        .merge(df_spy, how="left", on="date")
        .merge(df_agg, how="left", on="date")
        .merge(df_hyg, how="left", on="date")
        .merge(df_tlt, how="left", on="date")
        .merge(df_gld, how="left", on="date")
        .merge(df_sv_hedged_income, how="left", on="date")
        .merge(df_sv_hedged_balanced, how="left", on="date")
        .merge(df_sv_hedged_enhanced_growth, how="left", on="date")
        .merge(df_sv_equity_buffer, how="left", on="date")
        .merge(df_sv_equity_buffer_growth, how="left", on="date")
    )
    cols_to_change = {
        "mqu1pplr": "buffer_100",
        "mquslblr": "buffer_020",
        "mqu1bslq": "buffer_010",
    }
    df_px.rename(columns=cols_to_change, inplace=True)
    df_px = (
        df_px
        .query("'2007-04-11' <= date & date <= '2024-12-31'")
        .reset_index(drop=True)
    )
    return df_px


@pytest.fixture
def corrections_test_data() -> pd.DataFrame:
    mc = MarketCorrections(asset="SPY", correction=-0.05)
    return mc.corrections


class TesterFixedWeightBacktester:
    def test_spy50_hyg50_daily(self,
                               price_test_data,
                               corrections_test_data):
        portfolio = {
            "spy": 0.5,
            "hyg": 0.5
        }
        date_start = datetime.date(2007, 4, 11)
        date_end = datetime.date(2024, 12, 31)
        drb = FixedWeightBacktester(
            portfolio,
            price_test_data,
            corrections_test_data,
            date_start,
            date_end,
            "daily")
        drb.calc_daily_returns()
        drb.calc_portfolio_statistics()
        drb.calc_period_drawdowns()

        accuracy = 7
        # cumulative return
        assert np.round(drb.cumulative_return["portfolio"], accuracy) == \
            np.round(2.78577924743747, accuracy)
        # annualized return
        assert np.round(drb.annual_return["portfolio"], accuracy) == \
            np.round(0.0780835704957896, accuracy)
        # volatility
        assert np.round(drb.volatility["portfolio"], accuracy) == \
            np.round(0.143624563266703, accuracy)
        # sharpe
        assert np.round(drb.sharpe_ratio["portfolio"], accuracy) == \
            np.round(0.595393084079564, accuracy)
        # maximum drawdown
        assert np.round(drb.drawdown_max["portfolio"], accuracy) == \
            np.round(-0.441957538955252, accuracy)

    def test_balanced_1_monthly(self,
                                price_test_data,
                                corrections_test_data):
        portfolio = {
            "spy": 0.45,
            "agg": 0.1,
            "tlt": 0.2,
            "buffer_010": 0.1,
            "buffer_020": 0.1,
            "buffer_100": 0.05,
        }
        date_start = datetime.date(2007, 4, 11)
        date_end = datetime.date(2024, 12, 31)
        drb = FixedWeightBacktester(
            portfolio,
            price_test_data,
            corrections_test_data,
            date_start,
            date_end,
            "monthly")
        drb.calc_daily_returns()
        drb.calc_portfolio_statistics()
        drb.calc_period_drawdowns()

        accuracy = 7
        # cumulative return
        assert np.round(drb.cumulative_return["portfolio"], accuracy) == \
            np.round(2.48981811791493, accuracy)
        # annualized return
        assert np.round(drb.annual_return["portfolio"], accuracy) == \
            np.round(0.0731386282783451, accuracy)
        # volatility
        assert np.round(drb.volatility["portfolio"], accuracy) == \
            np.round(0.103533683282768, accuracy)
        # sharpe
        assert np.round(drb.sharpe_ratio["portfolio"], accuracy) == \
            np.round(0.733648500090021, accuracy)
        # maximum drawdown
        assert np.round(drb.drawdown_max["portfolio"], accuracy) == \
            np.round(-0.317950340976042, accuracy)
