from typing import Any, Iterator

from tango.common import Registrable


class SoccerDataSource(Registrable):
    def collect(self) -> Iterator[Any]:
        raise NotImplementedError
