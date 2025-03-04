import yfinance as yf
import datetime


class PriceFetcher:
    """
    Downloads historical adjusted close prices from Yahoo Finance.

    Attributes:
    -----------
    assets: list[str]
        Assets for which to grab historical prices.

    prices: pd.DataFrame
        Adjusted close prices for each asset in assets.

    date_min: datetime.date
        First date for which all assets have a price.

    date_max: datetime.date
        Last date for which all assets have a price.
    """
    def __init__(self, assets: list[str]):
        """
        Parameters:
        assets: list[str]
            Assets for which to grab historical prices.
        """
        self.assets = [x.lower() for x in assets]

        # attributes
        self.prices = None
        self.date_min = None
        self.date_max = None

    def fetch(self):
        """
        Downloads adjusted close prices from Yahoo finance.
        """
        # downloading the prices from Yahoo finance
        self.prices = yf.download(
            self.assets,
            start="1900-01-01",
            end=datetime.date.today() + datetime.timedelta(days=1),
            auto_adjust=False
        )

        # cleaning up the prices DataFrame
        self.prices = self.prices["Adj Close"].reset_index()
        # self.prices["Date"] = self.prices["Date"].dt.date
        self.prices.columns = self.prices.columns.str.lower()
        self.prices = self.prices.rename_axis(None, axis=1)

        # finding the minimum and maximum dates in the price DataFrame
        dates_min = []
        dates_max = []
        for ix_asset in self.assets:
            query = f"{ix_asset}.notna()"
            dt_min = self.prices.query(query)["date"].min()
            dt_max = self.prices.query(query)["date"].max()
            dates_min.append(dt_min)
            dates_max.append(dt_max)
        self.date_min = max(dates_min)
        self.date_max = min(dates_max)

        return None
