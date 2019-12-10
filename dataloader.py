from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Union
import pytz

import pymongo


class DataLoader(metaclass=ABCMeta):
    """Parse json payload and upload to mongodb"""

    def __init__(self, client: pymongo.MongoClient):
        self._client = client

    @property
    def database_name(self) -> str:
        return "nbafantasy"

    @property
    @abstractmethod
    def collection_name(self) -> str:
        pass

    @property
    def collection(self) -> pymongo.collection.Collection:
        return self._client[self.database_name][self.collection_name]

    @abstractmethod
    def parse(self, payload: Dict) -> List[Dict]:
        pass

    @abstractmethod
    def upload(self, *args, **kwargs) -> None:
        pass

    def download(self, query: Optional[Dict] = None, fields: Optional[Union[List, Dict]] = None) -> List[Dict]:
        return self.collection.find(query, fields)


class GameDataLoader(DataLoader):
    """Parse season game data and upload to mongodb"""

    @property
    def collection_name(self) -> str:
        return "game"

    def parse(self, payload: Dict) -> List[Dict]:
        season = payload["season"]
        output = payload["games"].copy()
        for game in output:
            game["_id"] = game["schedule"]["id"]
            game["season"] = season
        return output

    def upload(self, document: Dict) -> None:
        self.collection.replace_one(
            {"_id": document["_id"]},
            document,
            upsert=True
        )


class PlayerDataLoader(DataLoader):
    """Parse players data and upload to mongodb"""

    @property
    def collection_name(self) -> str:
        return "player"

    def parse(self, payload: Dict) -> List[Dict]:
        output = []
        for player_payload in payload["players"].copy():
            player = player_payload["player"].copy()
            player["_id"] = player.pop("id")
            player["teamAsOfDate"] = player_payload["teamAsOfDate"]
            output.append(player)
        return output

    def upload(self, document: Dict) -> None:
        self.collection.replace_one(
            {"_id": document["_id"]},
            document,
            upsert=True
        )


class GameLogDataLoader(DataLoader):
    """Parse game log data and upload to mongodb"""

    @property
    def collection_name(self) -> str:
        return "gamelog"

    def parse(self, payload: Dict) -> List[Dict]:
        return payload["gamelogs"].copy()

    def upload(self, document: Dict) -> None:
        self.collection.replace_one(
            {
                "game.id": document["game"]["id"],
                "player.id": document["player"]["id"],
                "team.id": document["team"]["id"]
            },
            document,
            upsert=True
        )


class LineupDataLoader(DataLoader):
    """Parse game log data and upload to mongodb"""

    @property
    def collection_name(self) -> str:
        return "lineup"

    def parse(self, payload: Dict) -> List[Dict]:
        payload = payload.copy()
        payload["_id"] = payload["game"]["id"]
        return payload

    def upload(self, document: Dict) -> None:
        self.collection.replace_one(
            {
                "_id": document["_id"],
            },
            document,
            upsert=True
        )


class FanduelDataLoader(DataLoader):
    """Parse fanduel salary data and upload to mongodb"""

    @property
    def collection_name(self) -> str:
        return "dfs"

    def parse(self, payload: Dict) -> List[Dict]:
        return list(
            filter(
                lambda x: x["dfsSource"] == 'FanDuel',
                payload["dfsEntries"]
            )
        )[0]["dfsRows"]

    def upload(self, document: Dict) -> None:
        self.collection.replace_one(
            {
                "game.id": document["game"]["id"],
                "player.id": document["player"]["id"],
                "team.id": document["team"]["id"]
            },
            document,
            upsert=True
        )
