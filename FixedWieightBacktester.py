import numpy as np
import pandas as pd
import datetime
from Utilities import period_max_drawdown


class FixedWeightBacktester:
    """
    A backtester for a weighted portfolio of assets with variable
    rebalancing of the weights.

    Attributes
    ----------
    portfolio: dict[str, float]
        Defines the assets and weights in the portfolio.

    prices: pd.DataFrame
        Contains the prices of historical prices of assets.
        This is typically the result of the PriceFetcher.fetch() method.

    market_corrections: pd.DataFrame
        Contains the start, bottom, end date of prices corrections of a
        particular asset.  Typically some kind of broad market index like
        SPY will be used.  This is usually the result of the MarketCorrections
        class.  It is modified by the calc_period_drawdown() method to contain
        the drawdowns of the weighted portfolio during the drawdown periods.

    date_start: datetime.date
        The start date of the backtest.

    date_end: datetime.date
        The end date of the backtest.

    assets: list[str]
        The component assets in the portfolio of the backtest.
        This is extracted from the portfolio.

    weights: list[str]
        The weights of the component assets in the portfolio.
        This is extracted from the portfolio

    returns: pd.DataFrame
        The prices, daily returns, equity curve, drawdowns of the assets
        and the weighted portfolio that is being backtested.

    cumulative_returns: dict[str, float]
        The cumulative returns for each of the component assets and
        the weighted portfolio.

    annual_returns: dict[str, float]
        The annualized returns for each of the component assets and
        the weighted portfolio.

    volatility: dict[str, float]
        The annualized volatility for each of the component assets and
        the weighted portfolio.

    sharpe_ratio: dict[str, float]
        The annualized sharpe-ratio for each of the component assets and
        the weighted portfolio.

    sharpe_ratio: dict[str, float]
        The annualized sharpe-ratio for each of the component assets and
        the weighted portfolio.

    drawdown_max: dict[str, float]
        The maximum for each of the component assets and
        the weighted portfolio.

    annual_performance: pd.DataFrame
        The performance of the weighted portfolio for each calendar
        year in the backtest.
    """
    def __init__(self,
                 portfolio: dict[str, float],
                 prices: pd.DataFrame,
                 market_corrections: pd.DataFrame,
                 date_start: datetime.date,
                 date_end: datetime.date,
                 frequency_rebalance: str):
        """
        portfolio: dict[str, float]
            Defines the assets and weights in the portfolio.

        prices: pd.DataFrame
            Contains the prices of historical prices of assets.
            This is typically the result of the PriceFetcher.fetch() method.

        market_corrections: pd.DataFrame
            Contains the start, bottom, end date of prices corrections of a
            particular asset.  Typically some kind of broad
            market index like SPY will be used.  This is usually the result
            of the MarketCorrections class.  It is modified by the
            calc_period_drawdown() method to contain the drawdowns of the
            weighted portfolio during the drawdown periods.

        date_start: datetime.date
            The start date of the backtest.

        date_end: datetime.date
            The end date of the backtest.
        """

        self.portfolio = portfolio
        self.prices = prices
        self.market_corrections = (
            market_corrections
            .query("@date_start <= start & end <= @date_end").copy()
        )

        self.date_start = date_start
        self.date_end = date_end
        self.frequency_rebalance = frequency_rebalance

        # isolating weights and assets from portfolio
        self.assets = []
        self.weights = []
        for asset, weight in self.portfolio.items():
            self.assets.append(asset)
            self.weights.append(weight)

    def calc_daily_returns(self) -> None:
        """
        Calculates the prices, daily returns, equity curve, drawdowns of
        the assets and the weighted portfolio that is being backtested.
        """

        self.returns = (
            self.prices[["date"] + self.assets]
                .query("@self.date_start <= date & date <= @self.date_end")
                .copy()
                .reset_index(drop=True)
        )

        # calculating component asset daily returns
        for ix_asset in self.assets:
            ret_col_name = "ret_" + ix_asset
            self.returns[ret_col_name] = self.returns[ix_asset].pct_change()
        self.returns.fillna(0, inplace=True)

        # calculating portfolio daily returns
        if self.frequency_rebalance is None:
            cols = []
            for ix_asset in self.assets:
                cols.append("ret_" + ix_asset)
            self.returns["ret_portfolio"] = \
                np.sum(np.array(self.returns[cols]) * self.weights, axis=1)
        else:
            self.calc_rebalanced_portfolio()

        # calculating equity curve for components and portfolio
        for ix_asset in self.assets:
            ret_col_name = "ret_" + ix_asset
            equity_col_name = "equity_" + ix_asset
            self.returns[equity_col_name] = \
                (1 + self.returns[ret_col_name]).cumprod()
        self.returns["equity_portfolio"] = \
            (1 + self.returns["ret_portfolio"]).cumprod()

        # calculating drawdowns for components and portfolio
        for ix_asset in self.assets:
            equity_col_name = "equity_" + ix_asset
            drawdown_col_name = "drawdown_" + ix_asset
            self.returns[drawdown_col_name] = (
                self.returns[equity_col_name] /
                self.returns[equity_col_name].cummax()) - 1
        self.returns["drawdown_portfolio"] = (
            self.returns["equity_portfolio"] /
            self.returns["equity_portfolio"].cummax()) - 1

    def calc_rebalanced_portfolio(self) -> None:
        """
        Calculates the daily returns of a rebalanced portfolio.
        """
        df = self.returns[["date"]].copy()

        # determining rebalance dates
        if self.frequency_rebalance == "annual":
            self.returns["year"] = self.returns["date"].dt.year
            df["year"] = df["date"].dt.year
            df_date_rebalance = \
                df.groupby(["year"])[["date"]].max().reset_index()
            df_date_rebalance.rename(
                columns={"date": "date_rebalance"},
                inplace=True
            )
            self.returns = self.returns.merge(
                df_date_rebalance,
                how="left",
                on=["year"]
            )
        elif self.frequency_rebalance == "semiannual":
            self.returns["year"] = self.returns["date"].dt.year
            self.returns["month"] = self.returns["date"].dt.month
            self.returns["half"] = np.where(self.returns["month"] <= 6, 1, 2)
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["half"] = np.where(df["month"] <= 6, 1, 2)
            df_date_rebalance = \
                df.groupby(["year", "half"])[["date"]].max().reset_index()
            df_date_rebalance.rename(
                columns={"date": "date_rebalance"},
                inplace=True
            )
            self.returns = self.returns.merge(
                df_date_rebalance,
                how="left",
                on=["year", "half"]
            )
        elif self.frequency_rebalance == "quarterly":
            self.returns["year"] = self.returns["date"].dt.year
            self.returns["quarter"] = self.returns["date"].dt.quarter
            df["year"] = df["date"].dt.year
            df["quarter"] = df["date"].dt.quarter
            df_date_rebalance = \
                df.groupby(["year", "quarter"])[["date"]].max().reset_index()
            df_date_rebalance.rename(
                columns={"date": "date_rebalance"},
                inplace=True
            )
            self.returns = self.returns.merge(
                    df_date_rebalance,
                    how="left",
                    on=["year", "quarter"]
            )
        elif self.frequency_rebalance == "monthly":
            self.returns["year"] = self.returns["date"].dt.year
            self.returns["month"] = self.returns["date"].dt.month
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df_date_rebalance = \
                df.groupby(["year", "month"])[["date"]].max().reset_index()
            df_date_rebalance.rename(
                columns={"date": "date_rebalance"},
                inplace=True
            )
            self.returns = self.returns.merge(
                df_date_rebalance,
                how="left",
                on=["year", "month"]
            )
        elif self.frequency_rebalance == "daily":
            self.returns["date_rebalance"] = self.returns["date"]

        # debugging
        # display(self.returns)
        # initializing values for iteration through self.returns
        lst_date = []
        before_rebal = {}
        lst_total_value = []
        total_value = 0
        lst_date.append(self.returns["date"].iloc[0])
        for ix_asset in self.assets:
            before_rebal[ix_asset] = [self.portfolio[ix_asset]]
            total_value += before_rebal[ix_asset][-1]
        lst_total_value.append(total_value)
        after_rebal = {}
        for ix_asset in self.assets:
            after_rebal[ix_asset] = [self.portfolio[ix_asset]]

        # iterating through self.returns to calculate portfolio values
        for _, row in self.returns[1:].iterrows():
            lst_date.append(row["date"])
            # calculating end-of-day value of each asset allocation
            for ix_asset, _ in before_rebal.items():
                before_rebal[ix_asset].append(
                    after_rebal[ix_asset][-1] * (1 + row["ret_" + ix_asset])
                )

            # calculating total portfolio value
            total_value = 0
            for ix_asset, _ in before_rebal.items():
                total_value += before_rebal[ix_asset][-1]
            lst_total_value.append(total_value)

            # rebalancing if needed
            if row["date"] == row["date_rebalance"]:
                for ix_asset, _ in after_rebal.items():
                    after_rebal[ix_asset].append(
                        total_value * self.portfolio[ix_asset]
                    )
            else:
                for ix_asset, _ in after_rebal.items():
                    after_rebal[ix_asset].append(before_rebal[ix_asset][-1])

        before_rebal["date"] = lst_date
        after_rebal["date"] = lst_date

        # adding columns to self.returns
        df_before_rebal = pd.DataFrame(before_rebal)
        for ix_asset, _ in before_rebal.items():
            df_before_rebal.rename(
                columns={ix_asset: "before_rebal_" + ix_asset},
                inplace=True
            )
        df_before_rebal
        df_after_rebal = pd.DataFrame(after_rebal)
        for ix_asset, _ in after_rebal.items():
            df_after_rebal.rename(
                columns={ix_asset: "after_rebal_" + ix_asset},
                inplace=True
            )
        df_after_rebal
        df_total_value = pd.DataFrame({
            "date": lst_date,
            "portfolio_total_value": lst_total_value,
        })
        # using a .merge rather than .concat to make this less brittle
        # self.returns = pd.concat(
        #   [self.returns,
        #   df_before_rebal,
        #   df_total_value,
        #   df_after_rebal],
        #   axis=1
        # )
        self.returns = (
            self.returns
                .merge(df_before_rebal, how="left",
                       left_on=["date"], right_on=["before_rebal_date"])
                .merge(df_total_value, how="left", on=["date"])
                .merge(df_after_rebal, how="left",
                       left_on=["date"], right_on=["after_rebal_date"])
        ).drop(columns=["before_rebal_date", "after_rebal_date"])
        self.returns["ret_portfolio"] = \
            self.returns["portfolio_total_value"].pct_change()
        self.returns.fillna(0, inplace=True)

        # debugging
        # display(df_before_rebal)
        # display(df_after_rebal)
        # display(self.returns)

    def calc_portfolio_statistics(self) -> None:
        """
        Calculates the portfolio statistics and annual performance of the
        component assets and the weighted portfolio being backtested.
        """
        # cumulative return
        self.cumulative_return = {}
        for ix_asset in self.assets:
            equity_col_name = "equity_" + ix_asset
            self.cumulative_return[ix_asset] = \
                (self.returns[equity_col_name].iloc[-1] - 1)
        self.cumulative_return["portfolio"] = \
            self.returns["equity_portfolio"].iloc[-1] - 1

        # annual return
        self.annual_return = {}
        for ix_asset in self.assets:
            equity_col_name = "equity_" + ix_asset
            self.annual_return[ix_asset] = (
                self.returns[equity_col_name].iloc[-1]
                ** (252/(len(self.returns) - 1)) - 1
            )
        self.annual_return["portfolio"] = \
            self.returns["equity_portfolio"].iloc[-1] \
            ** (252/(len(self.returns) - 1)) - 1

        # volatility
        self.volatility = {}
        for ix_asset in self.assets:
            ret_col_name = "ret_" + ix_asset
            self.volatility[ix_asset] = \
                self.returns[ret_col_name][1:].std() * np.sqrt(252)
        self.volatility["portfolio"] = \
            self.returns["ret_portfolio"][1:].std() * np.sqrt(252)

        # sharpe-ratio
        self.sharpe_ratio = {}
        for ix_asset in self.assets:
            ret_col_name = "ret_" + ix_asset
            self.sharpe_ratio[ix_asset] = (
                self.returns[ret_col_name][1:].mean() /
                self.returns[ret_col_name][1:].std()
            ) * np.sqrt(252)
        self.sharpe_ratio["portfolio"] = (
            self.returns["ret_portfolio"][1:].mean() /
            self.returns["ret_portfolio"][1:].std()
        ) * np.sqrt(252)

        # maximum drawdown
        self.drawdown_max = {}
        for ix_asset in self.assets:
            drawdown_col_name = "drawdown_" + ix_asset
            self.drawdown_max[ix_asset] = self.returns[drawdown_col_name].min()
        self.drawdown_max["portfolio"] = \
            self.returns["drawdown_portfolio"].min()

        # annual performance
        df_portfolio = self.returns[["date", "ret_portfolio"]].copy()
        df_portfolio["date"] = pd.to_datetime(df_portfolio["date"])
        df_portfolio["year"] = df_portfolio["date"].dt.year
        self.annual_performance = (
            df_portfolio
            .groupby(["year"])[["ret_portfolio"]]
            .agg(lambda x: np.prod(1 + x) - 1)
            .reset_index()
        )

    def calc_period_drawdowns(self) -> None:
        """
        Calculates the performance of the weighted portfolio during
        the drawdown periods in self.market_corrections
        """
        drawdowns_portfolio = []
        for ix in self.market_corrections.index:
            dt_start = self.market_corrections.at[ix, "start"]
            dt_end = self.market_corrections.at[ix, "end"]
            drawdown_portfolio = \
                period_max_drawdown(
                    asset="portfolio",
                    date_start=dt_start,
                    date_end=dt_end,
                    df_ret=self.returns,
                )
            drawdowns_portfolio.append(drawdown_portfolio)
        self.market_corrections["drawdown_portfolio"] = drawdowns_portfolio
