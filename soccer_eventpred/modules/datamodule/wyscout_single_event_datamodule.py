from copy import deepcopy
from typing import Dict, List, Optional, cast

import torch
from tqdm import tqdm

from soccer_eventpred.data.dataclass import Instance, SingleEventBatch
from soccer_eventpred.data.vocabulary import PAD_TOKEN, UNK_TOKEN, Vocabulary
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.modules.datamodule.soccer_dataset import SoccerEventDataset


@SoccerDataModule.register("wyscout_single")
class WyScoutSingleEventDataModule(SoccerDataModule):
    def __init__(
        self,
        train_datasource,
        val_datasource=None,
        test_datasource=None,
        batch_size=32,
        num_workers=0,
        label2events: Optional[Dict[str, List[str]]] = None,
        vocab: Optional[Vocabulary] = None,
    ):
        super().__init__()
        self._train_dataset = SoccerEventDataset()
        self._val_dataset = SoccerEventDataset() if val_datasource else None
        self._test_dataset = SoccerEventDataset() if test_datasource else None
        self._train_datasource = train_datasource
        self._val_datasource = val_datasource
        self._test_datasource = test_datasource
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab = vocab or Vocabulary()
        self._label2events = label2events
        if self._label2events is not None:
            self._event2label = {}
            for label, events in self._label2events.items():
                for event in events:
                    self._event2label[event] = label
        else:
            self._event2label = None
        self.event_counts = {}

    def prepare_data(self):
        if not self.vocab.size("events"):
            self.build_vocab()
        self._prepare_data(self._train_dataset, self._train_datasource)

        if self._val_datasource is not None:
            self._prepare_data(self._val_dataset, self._val_datasource)

        if self._test_datasource is not None:
            self._prepare_data(self._test_dataset, self._test_datasource)

    def _prepare_data(self, dataset, data_source):
        for event_sequence in data_source.collect():
            instance = self._prepare_instance(event_sequence)
            dataset.add(instance)

    # def build_teams_vocab(self, event_sequences=None) -> None:
    #     self.vocab.add(UNK_TOKEN, "teams")
    #     self.vocab.add(PAD_TOKEN, "teams")
    #     if event_sequences is None:
    #         event_sequences = self._train_datasource.collect()
    #     for event_sequence in event_sequences:
    #         for event in event_sequence.events:
    #             self.vocab.add(event.team_name, "teams")

    # def build_events_vocab(self, event_sequences=None) -> None:
    #     self.vocab.add(UNK_TOKEN, "events")
    #     self.vocab.add(PAD_TOKEN, "events")
    #     if event_sequences is None:
    #         event_sequences = self._train_datasource.collect()
    #     for event_sequence in event_sequences:
    #         for event in event_sequence.events:
    #             if self._event2label is not None:
    #                 self.vocab.add(self._event2label[event.comb_event_name], "events")
    #             else:
    #                 self.vocab.add(event.comb_event_name, "events")

    # def build_players_vocab(self, event_sequences=None) -> None:
    #     self.vocab.add(UNK_TOKEN, "players")
    #     self.vocab.add(PAD_TOKEN, "players")
    #     if event_sequences is None:
    #         event_sequences = self._train_datasource.collect()
    #     for event_sequence in event_sequences:
    #         for event in event_sequence.events:
    #             self.vocab.add(event.player_name, "players")

    def build_vocab(self, event_sequences=None):
        if (
            self.vocab.size("teams")
            and self.vocab.size("events")
            and self.vocab.size("players")
        ):
            return
        self.vocab.add(UNK_TOKEN, "teams")
        self.vocab.add(PAD_TOKEN, "teams")
        self.vocab.add(UNK_TOKEN, "events")
        self.vocab.add(PAD_TOKEN, "events")
        self.vocab.add(UNK_TOKEN, "players")
        self.vocab.add(PAD_TOKEN, "players")
        if event_sequences is None:
            event_sequences = self._train_datasource.collect()
        for event_sequence in event_sequences:
            for event in event_sequence.events:
                self.vocab.add(event.team_name, "teams")
                self.vocab.add(event.player_name, "players")
                if self._event2label is not None:
                    event_id = self.vocab.add(
                        self._event2label[event.comb_event_name], "events"
                    )
                    self.event_counts[event_id] = self.event_counts.get(event_id, 0) + 1
                else:
                    event_id = self.vocab.add(event.comb_event_name, "events")
                    self.event_counts[event_id] = self.event_counts.get(event_id, 0) + 1
        self.event_counts[self.vocab.get(UNK_TOKEN, "events")] = 0
        self.event_counts[self.vocab.get(PAD_TOKEN, "events")] = 0
        self.event_counts = [
            elem[1] for elem in sorted(self.event_counts.items(), key=lambda x: x[0])
        ]

    def _prepare_instance(self, event_sequence):
        event_times = [event.scaled_event_time for event in event_sequence.events]
        team_ids = [
            self.vocab.get(event.team_name, "teams") for event in event_sequence.events
        ]
        start_pos_x = [event.start_pos_x for event in event_sequence.events]
        start_pos_y = [event.start_pos_y for event in event_sequence.events]
        end_pos_x = [event.end_pos_x for event in event_sequence.events]
        end_pos_y = [event.end_pos_y for event in event_sequence.events]
        if self._event2label is not None:
            event_ids = [
                self.vocab.get(self._event2label[event.comb_event_name], "events")
                for event in event_sequence.events
            ]
        else:
            event_ids = [
                self.vocab.get(event.comb_event_name, "events")
                for event in event_sequence.events
            ]

        player_ids = [
            self.vocab.get(event.player_name, "players")
            for event in event_sequence.events
        ]
        return Instance(
            event_times,
            team_ids,
            event_ids,
            player_ids,
            start_pos_x,
            start_pos_y,
            end_pos_x,
            end_pos_y,
        )

    def setup(self, stage: str) -> None:
        ...

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._val_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._test_dataset)

    def build_dataloader(
        self, dataset, batch_size=None, shuffle=False, num_workers=0
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )

    def transfer_batch_to_device(
        self, batch: SingleEventBatch, device, dataloader_idx: int
    ) -> SingleEventBatch:
        return SingleEventBatch(
            event_times=batch.event_times.to(device),
            team_ids=batch.team_ids.to(device),
            event_ids=batch.event_ids.to(device),
            player_ids=batch.player_ids.to(device),
            start_pos_x=batch.start_pos_x.to(device),
            start_pos_y=batch.start_pos_y.to(device),
            end_pos_x=batch.end_pos_x.to(device),
            end_pos_y=batch.end_pos_y.to(device),
            mask=batch.mask.to(device),
            labels=batch.labels.to(device),
        )

    def batch_collator(self, instances: List[Instance]) -> SingleEventBatch:
        max_length = max(len(instance.event_ids) for instance in instances)
        event_times = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 120, dtype=torch.long),
        )
        team_ids = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "teams"),
                dtype=torch.long,
            ),
        )
        event_ids = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "events"),
                dtype=torch.long,
            ),
        )
        player_ids = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "players"),
                dtype=torch.long,
            ),
        )
        start_pos_x = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        start_pos_y = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        end_pos_x = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        end_pos_y = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        mask = cast(
            torch.BoolTensor,
            torch.zeros((len(instances), max_length), dtype=torch.bool),
        )
        labels = cast(
            torch.LongTensor,
            torch.zeros(len(instances), dtype=torch.long),
        )

        for i, instance in enumerate(instances):
            event_times[i, : len(instance.event_times)] = torch.tensor(
                instance.event_times, dtype=torch.long
            )
            team_ids[i, : len(instance.team_ids)] = torch.tensor(
                instance.team_ids, dtype=torch.long
            )
            event_ids[i, : len(instance.event_ids)] = torch.tensor(
                instance.event_ids, dtype=torch.long
            )
            player_ids[i, : len(instance.player_ids)] = torch.tensor(
                instance.player_ids, dtype=torch.long
            )
            start_pos_x[i, : len(instance.start_pos_x)] = torch.tensor(
                instance.start_pos_x, dtype=torch.long
            )
            start_pos_y[i, : len(instance.start_pos_y)] = torch.tensor(
                instance.start_pos_y, dtype=torch.long
            )
            end_pos_x[i, : len(instance.end_pos_x)] = torch.tensor(
                instance.end_pos_x, dtype=torch.long
            )
            end_pos_y[i, : len(instance.end_pos_y)] = torch.tensor(
                instance.end_pos_y, dtype=torch.long
            )
            mask[i, : len(instance.event_ids)] = True
            labels[i] = instance.event_ids[-1]

        return SingleEventBatch(
            event_times=event_times,
            team_ids=team_ids,
            event_ids=event_ids,
            player_ids=player_ids,
            start_pos_x=start_pos_x,
            start_pos_y=start_pos_y,
            end_pos_x=end_pos_x,
            end_pos_y=end_pos_y,
            mask=mask,
            labels=labels,
        )
