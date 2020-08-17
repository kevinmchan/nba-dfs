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
    def __init__(self, window, stats, post_agg_stats):
        super().__init__()
        self._window = window
        self._stats = stats
        self._post_agg_stats = post_agg_stats

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["player_id", "team_id", "game_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")

        sorted_dataset = historical_stats.sort_values(by=["player_id", "date"])
        averages = (
            sorted_dataset
            .groupby(["player_id"])[self._stats]
            .apply(lambda x: x.shift(1).rolling(window=self._window, min_periods=1).mean())
        )

        for key, func in self._post_agg_stats.items():
            averages = averages.assign(**{key: func})
 
        averages = averages.rename(columns=lambda col: f"{col}_{self._window}g_avg")
 
        return sorted_dataset[["player_id", "team_id", "game_id"]].join(averages)

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["player_id", "game_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")
        self.check_required_columns(df=lineup, cols=["player_id"], name="Lineup")

        avgs = (
            historical_stats
            .groupby(["player_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats].mean())
        )

        for key, func in self._post_agg_stats.items():
            avgs = avgs.assign(**{key: func})
        
        avgs = (
            avgs
            .rename(columns=lambda col: f"{col}_{self._window}g_avg")
            .reset_index()
        )
        return lineup[["player_id"]].merge(avgs, how="left", on=["player_id"])


class PrevStartingRate(BaseTransformation):
    def __init__(self, window):
        super().__init__()
        self._window = window

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        return (
            historical_stats
            .sort_values(by=["player_id", "date"])
            .assign(prev_starts=lambda x:
                x
                .groupby(["player_id"])["starter"]
                .apply(lambda y: y.shift(1).rolling(window=self._window, min_periods=1).mean())
            )
            .rename(columns={"prev_starts": f"prev_starts_{self._window}g"})
            [["game_id", "team_id", "player_id", f"prev_starts_{self._window}g"]]
        )

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        return (
            historical_stats
            .groupby(["player_id"])
            .apply(lambda x: x.nlargest(self._window, "date")["starter"].mean())
            .rename(f"prev_starts_{self._window}g")
            .reset_index()
            .merge(lineup[["player_id"]])
        )


class OpponentAboveAverageAllowed(BaseTransformation):
    def __init__(self, window, stats, post_agg_stats):
        super().__init__()
        self._window = window
        self._stats = stats
        self._post_agg_stats = post_agg_stats

    def _sum_stats_by_game_team(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        return (
            historical_stats
            .groupby(["game_id", "team_id", "opp_team_id", "date"])[self._stats]
            .sum()
            .reset_index()
            .sort_values(by=["date", "team_id"])
        )
    
    def _historical_team_average(self, team_game_stats: pd.DataFrame) -> pd.DataFrame:
        return (
            team_game_stats
            .pipe(
                lambda x:
                    x[["team_id", "game_id", "date"]]
                    .join(
                        x.groupby(["team_id"])[self._stats]
                        .apply(lambda y: y.shift(1).rolling(window=self._window, min_periods=1).mean())
                    )
            )
        )

    def _add_post_agg_stats(self, stats: pd.DataFrame) -> pd.DataFrame:
        for key, func in self._post_agg_stats.items():
            stats = stats.assign(**{key: func})
        return stats

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        # total stats per game
        total_per_game = (
            historical_stats
            .pipe(self._sum_stats_by_game_team)
            .pipe(self._add_post_agg_stats)
        )
            
        # average team stat scored
        team_avg = (
            total_per_game
            .pipe(self._historical_team_average)
            .pipe(self._add_post_agg_stats)
        )

        # above average stats scored
        above_avg_allowed = total_per_game.copy()
        for stat in self._stats + list(self._post_agg_stats.keys()):
            above_avg_allowed[stat] = total_per_game[stat] - team_avg[stat]

        # average opponent allowed above average
        opp_avg_allowed = (
            above_avg_allowed
            .pipe(
                lambda x:
                    x[["opp_team_id", "game_id"]]
                    .join(
                        x.groupby(["opp_team_id"])[self._stats + list(self._post_agg_stats.keys())]
                        .apply(lambda y: y.shift(1).rolling(window=self._window, min_periods=1).mean())
                        .rename(columns=lambda col: f"{col}_opp_allowed_{self._window}g_above_avg")
                    )
            )
        )

        # merge opponent team stats to player-games
        player_opp_avg_allowed = (
            historical_stats[["player_id", "team_id", "game_id", "opp_team_id"]]
            .merge(opp_avg_allowed, how="left", on=["game_id", "opp_team_id"])
            .drop(columns=["opp_team_id"])
        )

        return player_opp_avg_allowed

    def current_features(self, lineup: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
        # total stats per game
        total_per_game = (
            historical_stats
            .pipe(self._sum_stats_by_game_team)
            .pipe(self._add_post_agg_stats)
        )
            
        # average team stat scored
        team_avg = (
            total_per_game
            .pipe(self._historical_team_average)
            .pipe(self._add_post_agg_stats)
        )

        # above average stats scored
        above_avg_allowed = total_per_game.copy()
        for stat in self._stats + list(self._post_agg_stats.keys()):
            above_avg_allowed[stat] = total_per_game[stat] - team_avg[stat]

        # average opponent allowed above average
        opp_avg_allowed = (
            above_avg_allowed
            .groupby(["opp_team_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats + list(self._post_agg_stats.keys())].mean())
            .rename(columns=lambda col: f"{col}_opp_allowed_{self._window}g_above_avg")
            .reset_index()
        )

        # merge opponent team stats to player-games
        player_opp_avg_allowed = (
            lineup[["player_id", "opp_team_id"]]
            .merge(opp_avg_allowed, how="left", on=["opp_team_id"])
            .drop(columns=["opp_team_id"])
        )

        return player_opp_avg_allowed


class OpponentAverageAllowed(BaseTransformation):
    def __init__(self, window, stats, post_agg_stats):
        super().__init__()
        self._window = window
        self._stats = stats
        self._post_agg_stats = post_agg_stats

    def historical_features(self, historical_stats: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["player_id", "team_id", "game_id", "opp_team_id", "date"] + self._stats
        self.check_required_columns(df=historical_stats, cols=required_columns, name="Stats")

        opp_allowed_per_game = (
            historical_stats
            .groupby(["game_id", "opp_team_id", "date"])[self._stats]
            .sum()
            .reset_index()
            .sort_values(by=["opp_team_id", "date"])
        )

        averages = (
            opp_allowed_per_game
            .groupby(["opp_team_id"])[self._stats]
            .apply(lambda y: y.shift(1).rolling(window=self._window, min_periods=1).mean())
        )

        for key, func in self._post_agg_stats.items():
            averages = averages.assign(**{key: func})
        
        averages = averages.rename(columns=lambda col: f"{col}_opp_allowed_{self._window}g_avg")

        opp_avg_allowed = (
            opp_allowed_per_game[["opp_team_id", "game_id"]]
            .join(averages)
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

        averages = (
            opp_allowed_per_game
            .groupby(["opp_team_id"])
            .apply(lambda x: x.nlargest(self._window, "date")[self._stats].mean())
        )

        for key, func in self._post_agg_stats.items():
            averages = averages.assign(**{key: func})

        opp_avg_allowed = (
            averages
            .rename(columns=lambda col: f"{col}_opp_allowed_{self._window}g_avg")
            .reset_index()
        )

        return (
            lineup[["player_id", "opp_team_id"]]
            .merge(opp_avg_allowed, how="left", on=["opp_team_id"])
            .drop(columns=["opp_team_id"])
        )


class CurrentTeammateAvgStats(BaseTransformation):
    def __init__(self, window, stats, post_agg_stats):
        super().__init__()
        self._window = window
        self._stats = stats
        self._post_agg_stats = post_agg_stats

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
        )

        for key, func in self._post_agg_stats.items():
            teammate_averages = teammate_averages.assign(**{key: func})

        teammate_averages = (
            teammate_averages
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
        )

        for key, func in self._post_agg_stats.items():
            teammate_averages = teammate_averages.assign(**{key: func})

        teammate_averages = (
            teammate_averages
            .rename(columns=lambda col: f"{col}_{self._window}g_proj_teammate_avg")
            .reset_index()
        )
        return teammate_averages


class HistoricalTeammateStats(BaseTransformation):
    def __init__(self, window, stats, post_agg_stats):
        super().__init__()
        self._window = window
        self._stats = stats
        self._post_agg_stats = post_agg_stats

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
        )

        for key, func in self._post_agg_stats.items():
            averages = averages.assign(**{key: func})
 
        averages = averages.rename(columns=lambda col: f"{col}_{self._window}g_teammate_avg")

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
        )

        for key, func in self._post_agg_stats.items():
            avgs = avgs.assign(**{key: func})

        avgs = (
            avgs
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
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


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
        x["fg3m" + suffix] * 3 + x["fg2m" + suffix] * 2 + x["ftm" + suffix] + x["non_scoring_pts" + suffix]
    )


def update_position(df, player_id, position):
    df = df.copy()
    for pos in ["C", "PF", "SF", "SG", "PG"]:
        df.loc[lambda x: x["player_id"] == player_id, f"position_{pos}"] = 0
    df.loc[lambda x: x["player_id"] == player_id, f"position_{position}"] = 1
    df.loc[lambda x: x["player_id"] == player_id, "position"] = position
    return df