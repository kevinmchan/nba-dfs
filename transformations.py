from abc import abstractmethod, ABCMeta

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformation(metaclass=ABCMeta):
    """Abstract base class for transformations"""

    def __init__(self):
        pass

    @staticmethod
    def check_required_columns(df, cols, name=None):
        for col in cols:
            assert col in df.columns, f"{name or 'Input'} dateframe must contain `{col}` column"

    @abstractmethod
    def historical_features(self, **historical_inputs) -> pd.DataFrame:
        """Applies transformation on historical sample and
        returns feature set for each player_id, game_id and team_id"""
        pass

    @abstractmethod
    def current_features(self, lineup: pd.DataFrame, **historical_inputs) -> pd.DataFrame:
        """Applies transformation on lineup and returns feature
        set for each player, game_id and team_id"""
        pass


class PlayerAverage(BaseTransformation):
    def __init__(self, window, stats):
        super().__init__()
        self._window = window
        self._stats = stats

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["player_id", "team_id", "game_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")

        sorted_dataset = historical_stats.sort_values(by=["player_id", "date"])
        averages = (
            sorted_dataset
            .groupby(["player_id"])[self._stats]
            .apply(lambda x: x.shift(1).rolling(window=self._window, min_periods=1).mean())
            .rename(columns=lambda col: f"{col}_{self._window}g_avg")
        )
        return sorted_dataset[["player_id", "team_id", "game_id"]].join(averages)

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["player_id", "game_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")
        self.check_required_columns(df=lineup, cols=["player_id"], name="Lineup")

        avgs = (
            historical_stats
            .groupby(["player_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats].mean())
            .rename(columns=lambda col: f"{col}_{self._window}g_avg")
            .reset_index()
        )
        return lineup[["player_id"]].merge(avgs, how="left", on=["player_id"])


class OpponentAverageAllowed(BaseTransformation):
    def __init__(self, window, stats):
        super().__init__()
        self._window = window
        self._stats = stats

    def historical_features(self, historical_stats) -> pd.DataFrame:
        required_columns = ["player_id", "team_id", "game_id", "opp_team_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")

        opp_allowed_per_game = (
            historical_stats
            .groupby(["game_id", "opp_team_id", "date"])[self._stats]
            .sum()
            .reset_index()
            .sort_values(by=["opp_team_id", "date"])
        )

        opp_avg_allowed = (
            opp_allowed_per_game
            .pipe(
                lambda x:
                    x[["opp_team_id", "game_id"]]
                    .join(
                        x.groupby(["opp_team_id"])[self._stats]
                        .apply(lambda y: y.shift(1).rolling(window=self._window, min_periods=1).mean())
                        .rename(columns=lambda col: f"{col}_opp_allowed_{self._window}g_avg")
                    )
            )
        )

        player_opp_avg_allowed = (
            historical_stats[["player_id", "team_id", "game_id", "opp_team_id"]]
            .merge(opp_avg_allowed, how="left", on=["game_id", "opp_team_id"])
            .drop(columns=["opp_team_id"])
        )
        return player_opp_avg_allowed

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["player_id", "team_id", "game_id", "opp_team_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")
        self.check_required_columns(df=lineup, cols=["player_id", "opp_team_id"], name="Lineup")

        opp_allowed_per_game = (
            historical_stats
            .groupby(["game_id", "opp_team_id", "date"])[self._stats]
            .sum()
            .reset_index()
            .sort_values(by=["opp_team_id", "date"])
        )

        opp_avg_allowed = (
            opp_allowed_per_game
            .groupby(["opp_team_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats].mean())
            .rename(columns=lambda col: f"{col}_opp_allowed_{self._window}g_avg")
            .reset_index()
        )

        return (
            lineup[["player_id", "opp_team_id"]]
            .merge(opp_avg_allowed, how="left", on=["opp_team_id"])
            .drop(columns=["opp_team_id"])
        )


class CurrentTeammateAvgStats(BaseTransformation):
    def __init__(self, window, stats):
        super().__init__()
        self._window = window
        self._stats = stats

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        sorted_dataset = historical_stats.sort_values(by=["player_id", "date"])
        player_averages = (
            sorted_dataset
            .groupby(["player_id"])[self._stats]
            .apply(lambda x: x.shift(1).rolling(window=self._window, min_periods=1).mean())
        )
        player_averages = sorted_dataset[["player_id", "team_id", "game_id"]].join(player_averages)

        teammate_averages = (
            historical_stats[["game_id", "team_id", "player_id"]]
            .merge(
                player_averages.rename(columns={"player_id": "teammate_id"}),
                on=["game_id", "team_id"],
                how="left"
            )
            .query("player_id != teammate_id")
            .groupby(["game_id", "team_id", "player_id"])[self._stats]
            .sum(min_count=1)
            .rename(columns=lambda col: f"{col}_{self._window}g_proj_teammate_avg")
            .reset_index()
        )
        return teammate_averages

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        player_averages = (
            historical_stats
            .groupby(["player_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats].mean())
            .reset_index()
        )

        teammate_averages = (
            lineup[["game_id", "team_id", "player_id"]]
            .merge(
                lineup[["game_id", "team_id", "player_id"]].rename(columns={"player_id": "teammate_id"}),
                on=["game_id", "team_id"]
            )
            .query("player_id != teammate_id")
            .merge(
                player_averages.rename(columns={"player_id": "teammate_id"}),
                on=["teammate_id"],
                how="left"
            )
            .groupby(["game_id", "team_id", "player_id"])[self._stats]
            .sum(min_count=1)
            .rename(columns=lambda col: f"{col}_{self._window}g_proj_teammate_avg")
            .reset_index()
        )
        return teammate_averages


class HistoricalTeammateStats(BaseTransformation):
    def __init__(self, window, stats):
        super().__init__()
        self._window = window
        self._stats = stats

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        # cross multiply player and teammate stats
        # take sum of teammate stats per game and team
        teammate_stats = (
            historical_stats[["game_id", "team_id", "player_id"]]
            .merge(
                historical_stats.rename(columns={"player_id": "teammate_id"}),
                on=["game_id", "team_id"],
            )
            .query("player_id != teammate_id")
            .groupby(["game_id", "team_id", "player_id", "date"])[self._stats].sum(min_count=1)
            .reset_index()
        )

        sorted_dataset = teammate_stats.sort_values(by=["player_id", "date"])
        averages = (
            sorted_dataset
            .groupby(["player_id"])[self._stats]
            .apply(lambda x: x.shift(1).rolling(window=self._window, min_periods=1).mean())
            .rename(columns=lambda col: f"{col}_{self._window}g_teammate_avg")
        )
        return sorted_dataset[["player_id", "team_id", "game_id"]].join(averages)

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        teammate_stats = (
            historical_stats[["game_id", "team_id", "player_id"]]
            .merge(
                historical_stats.rename(columns={"player_id": "teammate_id"}),
                on=["game_id", "team_id"],
            )
            .query("player_id != teammate_id")
            .groupby(["game_id", "team_id", "player_id", "date"])[self._stats].sum(min_count=1)
            .reset_index()
        )

        avgs = (
            teammate_stats
            .groupby(["player_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats].mean())
            .rename(columns=lambda col: f"{col}_{self._window}g_teammate_avg")
            .reset_index()
        )
        return lineup[["player_id"]].merge(avgs, how="left", on=["player_id"])


class PandasStandardScalar(StandardScaler):
    def transform(self, X):
        transformed = super().transform(X)
        return pd.DataFrame(transformed, columns=map("{}_scaled".format, X.columns))


class PandasImputer(SimpleImputer):
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=None):
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value
        )

    def transform(self, X):
        transformed = super().transform(X)
        return pd.DataFrame(transformed, columns=X.columns)


class PandasMissingIndicator(MissingIndicator):
    def __init__(self, missing_values=np.nan):
        super(PandasMissingIndicator, self).__init__(
            missing_values=missing_values,
            features="all"
        )

    def transform(self, X):
        transformed = super().transform(X)
        return X.join(pd.DataFrame(transformed, columns=map("{}_missing".format, X.columns)))


class PandasVarianceThreshold(VarianceThreshold):
    def transform(self, X):
        transformed = super().transform(X)
        return pd.DataFrame(transformed, columns=X.columns[self.get_support()])


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.columns, errors="ignore")


def fanduel_score(x, suffix=""):
    return (
        x["fg3m" + suffix] + x["ast" + suffix] * 1.5 + x["blk" + suffix] * 3 + x["fgm" + suffix] * 2
        + x["ftm" + suffix] + x["reb" + suffix] * 1.2 + x["stl" + suffix] * 3 - x["tov" + suffix]
    )
