from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class WyScoutEvent:
    match_period: str
    event_time_period: float
    event_time: float
    scaled_event_time: int
    team_name: str
    comb_event_name: str
    start_pos_x: int
    start_pos_y: int
    end_pos_x: int
    end_pos_y: int
    player_name: str
    tags: List[str]


@dataclass
class WyScoutEventSequence:
    competition: str
    wyscout_match_id: int
    wyscout_team_id_1: int
    player_list_1: List[str]
    events: List[WyScoutEvent]
    wyscout_team_id_2: Optional[int] = None
    player_list_2: Optional[List[str]] = None


@dataclass
class JLeagueEvent:
    match_period: str
    event_time_period: float
    event_time: float
    scaled_event_time: int
    team_name: str
    event_name: str
    start_pos_x: int
    start_pos_y: int
    player_name: str


@dataclass
class JLeagueEventSequence:
    match_id: int
    team_index: int
    events: List[WyScoutEvent]


@dataclass
class Batch:
    event_times: torch.LongTensor
    team_ids: torch.LongTensor
    event_ids: torch.LongTensor
    player_ids: torch.LongTensor
    start_pos_x: torch.LongTensor
    start_pos_y: torch.LongTensor
    mask: torch.BoolTensor
    end_pos_x: Optional[torch.LongTensor] = None
    end_pos_y: Optional[torch.LongTensor] = None


@dataclass
class Instance:
    event_times: List[int]
    team_ids: List[int]
    event_ids: List[int]
    player_ids: List[int]
    start_pos_x: List[int]
    start_pos_y: List[int]
    end_pos_x: Optional[List[int]] = None
    end_pos_y: Optional[List[int]] = None


@dataclass
class Prediction:
    event_times: List[int]
    team_ids: List[int]
    event_ids: List[int]
    player_ids: List[int]
    start_pos_x: List[int]
    start_pos_y: List[int]
    end_pos_x: Optional[List[int]] = None
    end_pos_y: Optional[List[int]] = None


@dataclass
class SingleEventBatch:
    event_times: torch.LongTensor
    team_ids: torch.LongTensor
    event_ids: torch.LongTensor
    player_ids: torch.LongTensor
    start_pos_x: torch.LongTensor
    start_pos_y: torch.LongTensor
    labels: torch.LongTensor
    mask: torch.BoolTensor
    end_pos_x: Optional[torch.LongTensor] = None
    end_pos_y: Optional[torch.LongTensor] = None


@dataclass
class SingleEventPrediction:
    event_times: List[int]
    team_ids: List[int]
    event_ids: List[int]
    player_ids: List[int]
    start_pos_x: List[int]
    start_pos_y: List[int]
    end_pos_x: Optional[List[int]] = None
    end_pos_y: Optional[List[int]] = None
