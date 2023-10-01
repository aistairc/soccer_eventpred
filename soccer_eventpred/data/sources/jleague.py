from typing import Any, Iterator, List

from soccer_eventpred.data.dataclass import JLeagueEvent, JLeagueEventSequence
from soccer_eventpred.data.sources.source import SoccerDataSource
from soccer_eventpred.env import DATA_DIR
from soccer_eventpred.util import load_jsonlines


@SoccerDataSource.register("jleague")
class JLeagueDataSource(SoccerDataSource):
    def __init__(self, data_name: str = "jleague", subset: str = "train.jsonl") -> None:
        self._datasource = load_jsonlines(
            DATA_DIR / data_name / "preprocessed/events" / subset
        )
        self._data: List[JLeagueEventSequence] = []
        self._build_data()

    def _build_data(self):
        for data in self._datasource:
            events = [JLeagueEvent(**event) for event in data.pop("events")]
            self._data.append(
                JLeagueEventSequence(
                    events=events,
                    **data,
                )
            )

    def collect(self) -> Iterator[Any]:
        yield from self._data
