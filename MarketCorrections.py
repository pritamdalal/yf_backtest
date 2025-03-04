import pandas as pd
import yfinance as yf
import datetime


class MarketCorrections:
    """
    Finds market corrections for all available data for a given
    asset and a given correction threshold level.

    Attrbutes
    ---------
    asset: str
        Ticker of asset for which to find corrections.

    correction: float
        Threshold level which determines the market correction to
        be identified.  For example a value of -0.05 will find all
        corrections greater that 5%.

    prices: pd.DataFrame
        Price history of asset. QUESTION: Should I make this private?

    drawdown_periods: pd.DataFrame
        All drawdown periods for the asset.  QUESTION: should I make
        this private, or simply not make a it an attribute.

    corrections: pd.DataFrame
        All drawdown periods that exceed the threshold. NOTE: I don't
        love the name of this attribute.
    """
    def __init__(self, asset, correction):
        """
        Initializing this class basically does all the work of
        creating a DataFrame that holds all the drawdown periods.
        Parameters:
        -----------
        asset: str
            Ticker of asset for which to find corrections.

        correction: float
            Threshold level which determines the market correction to
            be identified.  For example a value of -0.05 will find all
            corrections greater that 5%.
        """
        # downloading the prices from Yahoo finance
        self.correction = correction
        self.asset = asset
        self.prices = yf.download(
            self.asset,
            start="1900-01-01",
            end=datetime.date.today() + datetime.timedelta(days=1),
            auto_adjust=False
        )

        # cleaning up the prices DataFrame
        self.prices = self.prices["Adj Close"].reset_index()
        self.prices["Date"] = self.prices["Date"].dt.date
        self.prices.columns = self.prices.columns.str.lower()
        self.prices = self.prices.rename_axis(None, axis=1)

        # calculating returns, equity curve, drawdown
        col_name_ret = "ret_" + self.asset.lower()
        self.prices[col_name_ret] = \
            self.prices[self.asset.lower()].pct_change()
        self.prices.fillna(0, inplace=True)
        col_name_equity = "equity_" + self.asset.lower()
        self.prices[col_name_equity] = \
            (1 + self.prices[col_name_ret]).cumprod()
        col_name_drawdown = "drawdown_" + self.asset.lower()
        self.prices[col_name_drawdown] = \
            (self.prices[col_name_equity] /
             self.prices[col_name_equity].cummax()) - 1

        # determining drawdown period start and end dates
        query = f"{col_name_drawdown} == 0"
        dates_start = \
            self.prices.query(query)["date"].iloc[:-1].values
        dates_end = self.prices.query(query)["date"].iloc[1:].values
        self.drawdown_periods = pd.DataFrame({
            "start": dates_start,
            "end": dates_end,
        }).reset_index(drop=True)

        # iterating through dates and collecting drawdowns and bottom dates
        period_drawdowns = []
        dates_bottom = []
        for ix in self.drawdown_periods.index:
            date_start = self.drawdown_periods.at[ix, "start"]
            mask_start = date_start <= self.prices["date"]
            date_end = self.drawdown_periods.at[ix, "end"]
            mask_end = self.prices["date"] <= date_end
            df_drawdown = self.prices[mask_start & mask_end]
            drawdown = df_drawdown[col_name_drawdown].min()
            query = f"{col_name_drawdown} == @drawdown"
            date_bottom = df_drawdown.query(query)["date"].iloc[0]
            period_drawdowns.append(drawdown)
            dates_bottom.append(date_bottom)
        self.drawdown_periods["bottom"] = dates_bottom
        self.drawdown_periods[col_name_drawdown] = period_drawdowns

        # filtering for corrections
        query = f"{col_name_drawdown} < @self.correction"
        self.corrections = \
            self.drawdown_periods.query(query).reset_index(drop=True)
