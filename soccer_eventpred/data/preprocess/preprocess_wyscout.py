from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from soccer_eventpred.env import DATA_DIR
from soccer_eventpred.util import load_json, save_as_jsonlines, save_formatted_json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFENSIVE_COMB_EVENTS = [
    "Duel_Air duel",
    "Duel_Ground defending duel",
    "Duel_Ground loose ball duel",
    "Interruption_Ball out of the field",
    "Interruption_Whistle",
    "Foul_Foul",
    "Foul_Hand foul",
    "Foul_Late card foul",
    "Foul_Out of game foul",
    "Foul_Protest",
    "Foul_Simulation",
    "Foul_Time lost foul",
    "Foul_Violent Foul",
    "Offside_",
]


def preprocess_wyscout_teams_data(
    input_path: Path | str = DATA_DIR / "wyscout/raw/mappings/teams.json",
    output_path: Path | str = DATA_DIR / "wyscout/preprocessed/mappings/id2team.json",
) -> None:
    data = load_json(input_path)
    output = {}
    for datum in data:
        datum["officialName"] = datum["officialName"].encode().decode("unicode_escape")
        output[int(datum.pop("wyId"))] = datum
    save_formatted_json(output, output_path)


def preprocess_wyscout_players_data(
    input_path: Path | str = DATA_DIR / "wyscout/raw/mappings/players.json",
    output_path: Path | str = DATA_DIR / "wyscout/preprocessed/mappings/id2player.json",
) -> None:
    data = load_json(input_path)
    output = {}
    for datum in data:
        datum["firstName"] = datum["firstName"].encode().decode("unicode_escape")
        datum["lastName"] = datum["lastName"].encode().decode("unicode_escape")
        datum["fullName"] = (
            datum["firstName"] + " " + datum["lastName"]
        )  # to make it unique
        output[int(datum.pop("wyId"))] = datum
    save_formatted_json(output, output_path)


def preprocess_wyscout_tags_data(
    input_path: Path | str = DATA_DIR / "wyscout/raw/mappings/tags2name.csv",
    output_path: Path
    | str = DATA_DIR / "wyscout/preprocessed/mappings/tagid2name.json",
) -> None:
    data = pd.read_csv(input_path)
    data.rename(
        columns={"Tag": "id", "Label": "label", "Description": "description"},
        inplace=True,
    )
    data = data.to_dict(orient="records")
    output = {}
    for datum in data:
        output[int(datum.pop("id"))] = datum
    save_formatted_json(output, output_path)


def preprocess_wyscout_events_data(
    input_dir: Path | str = DATA_DIR / "wyscout/raw/events",
    output_dir: Path | str = DATA_DIR / "wyscout/preprocessed/events",
    mappings_dir: Path | str = DATA_DIR / "wyscout/preprocessed/mappings",
    targets: List[str] = ["events_Spain.json"],
    offense_only: bool = True,
) -> None:
    output_dir = Path(output_dir).resolve()
    mappings_dir = Path(mappings_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_wyscout_data(input_dir, targets)
    df.rename(columns={"teamId": "wyscout_team_id"}, inplace=True)

    id2team = load_json(mappings_dir / "id2team.json")
    df["team_name"] = df["wyscout_team_id"].map(
        lambda x: id2team[str(x)]["officialName"]
    )

    df.rename(columns={"matchId": "wyscout_match_id"}, inplace=True)
    df.rename(
        columns={"eventSec": "event_time_period", "matchPeriod": "match_period"},
        inplace=True,
    )
    df["event_time"] = df["event_time_period"]
    df.loc[df["match_period"] == "2H", "event_time"] = (
        df.loc[df["match_period"] == "2H", "event_time_period"] + 60 * 60
    )  # start second half time at 60 minutes
    df.loc[df["match_period"] == "E1", "event_time"] = (
        df.loc[df["match_period"] == "E1", "event_time_period"] + 120 * 60
    )  # start ET1 at 120 minutes
    df.loc[df["match_period"] == "E2", "event_time"] = (
        df.loc[df["match_period"] == "E2", "event_time_period"] + 150 * 60
    )  # start ET2 at 150 minutes
    df.loc[df["match_period"] == "P", "event_time"] = (
        df.loc[df["match_period"] == "P", "event_time_period"] + 180 * 60
    )  # start Pens at 180 minutes

    # {start, end}_pos_{x, y}
    df["positions"] = df["positions"].astype(str).str.replace("[{'y': ", "")
    df["positions"] = df["positions"].astype(str).str.replace(" 'x': ", "")
    df["positions"] = df["positions"].astype(str).str.replace("}, {'y': ", ",")
    df["positions"] = df["positions"].astype(str).str.replace("}]", "")

    dfpos = df["positions"].str.split(",", expand=True)
    dfpos.columns = ["start_pos_y", "start_pos_x", "end_pos_y", "end_pos_x"]
    dfpos = dfpos.loc[:, ["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]]

    df = pd.concat([df, dfpos], axis=1)
    df = df.drop("positions", axis=1)

    # fill none and -1
    for i in tqdm(range(len(df)), total=len(df)):
        if df.loc[i, "end_pos_y"] is None or df.loc[i, "end_pos_y"] == "-1":
            df.loc[i, "end_pos_y"] = (
                df.loc[i, "start_pos_y"]
                if df.loc[i, "eventName"] == "Foul"
                else df.loc[i + 1, "start_pos_y"]
            )

        if df.loc[i, "end_pos_x"] is None or df.loc[i, "end_pos_x"] == "-1":
            df.loc[i, "end_pos_x"] = (
                df.loc[i, "start_pos_x"]
                if df.loc[i, "eventName"] == "Foul"
                else df.loc[i + 1, "start_pos_x"]
            )

        if df.loc[i, "start_pos_y"] is None or df.loc[i, "start_pos_y"] == "-1":
            df.loc[i, "start_pos_y"] = (
                df.loc[i, "end_pos_y"]
                if df.loc[i, "eventName"] == "Foul"
                else df.loc[i - 1, "end_pos_y"]
            )

        if df.loc[i, "start_pos_x"] is None or df.loc[i, "start_pos_x"] == "-1":
            df.loc[i, "start_pos_x"] = (
                df.loc[i, "end_pos_x"]
                if df.loc[i, "eventName"] == "Foul"
                else df.loc[i - 1, "end_pos_x"]
            )
    df[["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]] = df[
        ["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]
    ].astype(int)

    # clip positions
    df[["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]] = df[
        ["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]
    ].clip(0, 100)
    assert (
        (0 > df[["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]])
        | (df[["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]] > 100)
    ).sum().sum() == 0

    # player_id, player_name
    id2player_path = mappings_dir / "id2player.json"
    id2player = load_json(id2player_path)
    df.rename(columns={"playerId": "wyscout_player_id"}, inplace=True)
    df["player_name"] = df["wyscout_player_id"].map(
        lambda x: id2player[str(x)]["fullName"] if str(x) in id2player else "UNK"
    )  # some unknown player existed (==0)

    tagid2name_path = mappings_dir / "tagid2name.json"
    tagid2name = load_json(tagid2name_path)
    df["tags"] = df["tags"].map(
        lambda items: [tagid2name[str(item["id"])]["label"] for item in items]
    )

    # goal, own-goal
    goal_indices = df[
        (df["tags"].map(lambda x: "Goal" in x)) & (df["eventName"] == "Save attempt")
    ].index
    df.loc[goal_indices, "subEventName"] = df.loc[goal_indices, "eventName"] = "Goal"
    own_goal_indices = df[(df["tags"].map(lambda x: "own_goal" in x))].index
    df.loc[own_goal_indices, "subEventName"] = "Own goal"

    # delete some events after 'own-goal' events
    delete_indices = []
    for idx in own_goal_indices:
        delete_idx = idx + 1
        while (
            not df.loc[delete_idx, "subEventName"]
            == df.loc[delete_idx, "eventName"]
            == "Goal"
        ):
            delete_indices.append(delete_idx)
            delete_idx += 1
        delete_indices.append(delete_idx)
        assert len(df.iloc[idx + 1 : delete_idx + 1]) <= 2

    df = df.drop(delete_indices).reset_index(drop=True)

    # the event's coordinates depends on the subject.
    # -> convert pos to (100 - pos) if the subject belongs to the opponent team.
    match_ids = list(df["wyscout_match_id"].unique())
    opponent_team_indices = pd.Index([])
    for match_id in tqdm(match_ids, total=len(match_ids)):
        match_indices = df.query("wyscout_match_id == @match_id").index
        opponent_team = list(df.loc[match_indices, "team_name"].unique())[  # noqa: F841
            1
        ]
        opponent_team_indices_in_match = df.query("team_name == @opponent_team").index
        opponent_team_indices = opponent_team_indices.append(
            match_indices.intersection(opponent_team_indices_in_match)
        )
    df.loc[
        opponent_team_indices, ["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"]
    ] = (
        100
        - df.loc[
            opponent_team_indices,
            ["start_pos_x", "start_pos_y", "end_pos_x", "end_pos_y"],
        ]
    )

    goal_indices = df.query('eventName == "Goal" and subEventName == "Goal"').index
    for idx in tqdm(goal_indices, total=len(goal_indices)):
        shot_index = idx
        team = df.iloc[idx]["team_name"]
        while df.iloc[shot_index]["team_name"] == team or df.iloc[shot_index][
            "eventName"
        ] not in ["Shot", "Free Kick"]:
            shot_index -= 1

        df.loc[
            idx, ["wyscout_player_id", "wyscout_team_id", "team_name", "player_name"]
        ] = df.loc[
            shot_index,
            ["wyscout_player_id", "wyscout_team_id", "team_name", "player_name"],
        ]

    # scale event_time into integers(0-120)
    scaler = MinMaxScaler(feature_range=(0, 120))
    scaled_event_time = (
        scaler.fit_transform(df["event_time"].values.reshape(-1, 1))
        .reshape(
            -1,
        )
        .astype(int)
    )
    df["scaled_event_time"] = scaled_event_time

    df.loc[:, "comb_event_name"] = df["eventName"] + "_" + df["subEventName"]
    df = df.drop(["eventId", "subEventName", "eventName", "subEventId", "id"], axis=1)[
        [
            "competition",
            "wyscout_match_id",
            "match_period",
            "event_time_period",
            "event_time",
            "scaled_event_time",
            "wyscout_team_id",
            "team_name",
            "comb_event_name",
            "start_pos_x",
            "start_pos_y",
            "end_pos_x",
            "end_pos_y",
            "wyscout_player_id",
            "player_name",
            "tags",
        ]
    ]
    df = df.query('match_period != "P"')
    if offense_only:
        df = df.query("comb_event_name not in @DEFENSIVE_COMB_EVENTS")
    df.to_pickle(output_dir / "all_preprocessed.pkl")
    return None


def split_wyscout_data(
    df_pickle_path: Path
    | str = DATA_DIR / "wyscout/preprocessed/events/all_preprocessed.pkl",
    output_dir: Path | str = DATA_DIR / "wyscout/preprocessed/events",
    random_state: int = 42,
) -> None:
    output_dir = Path(output_dir)
    df = pd.read_pickle(df_pickle_path)
    match2teams = df.groupby("wyscout_match_id")["team_name"].unique().reset_index()
    match2teams["team_name"] = match2teams["team_name"].map(lambda x: x[0])

    train_match, dev_test_match = train_test_split(
        match2teams,
        test_size=0.2,
        random_state=random_state,
        stratify=match2teams["team_name"],
    )
    dev_match, test_match = train_test_split(
        dev_test_match,
        test_size=0.5,
        random_state=random_state,
        stratify=dev_test_match["team_name"],
    )

    train_match_ids = list(train_match["wyscout_match_id"].unique())
    dev_match_ids = list(dev_match["wyscout_match_id"].unique())
    test_match_ids = list(test_match["wyscout_match_id"].unique())

    train = df.query("wyscout_match_id in @train_match_ids")
    dev = df.query("wyscout_match_id in @dev_match_ids")
    test = df.query("wyscout_match_id in @test_match_ids")

    logger.info(f"{len(train)=}\t{len(dev)=}\t{len(test)=}")
    logger.info(
        f"""
        {train["wyscout_match_id"].nunique()=}
        \t{dev["wyscout_match_id"].nunique()=}
        \t{test["wyscout_match_id"].nunique()=}
        """
    )
    logger.info(
        f"""
        {len(set(test_match_ids) - set(dev_match_ids))=}
        \t{len(set(test_match_ids) - set(train_match_ids))=}
        \t{len(set(dev_match_ids) - set(train_match_ids))=}
        """
    )

    # one match -> one dict
    all_events = []
    for data, label in zip((train, dev, test), ("train", "dev", "test")):
        match_dict_list = []
        match_ids = list(data["wyscout_match_id"].unique())
        for match_id in match_ids:
            match = data.query("wyscout_match_id == @match_id")
            competition = match["competition"].unique()[0]
            team_ids = list(match["wyscout_team_id"].unique())
            match_id = match["wyscout_match_id"].unique()[0]
            team_id1 = team_ids[0]
            team_id2 = team_ids[1]
            player_list1 = list(
                match.query("wyscout_team_id == @team_id1")["player_name"].unique()
            )
            player_list2 = list(
                match.query("wyscout_team_id == @team_id2")["player_name"].unique()
            )
            events = match.drop(
                [
                    "competition",
                    "wyscout_match_id",
                    "wyscout_team_id",
                    "wyscout_player_id",
                ],
                axis=1,
            ).to_dict(orient="records")
            match_dict = {
                "competition": competition,
                "wyscout_match_id": int(match_id),
                "wyscout_team_id_1": int(team_id1),
                "wyscout_team_id_2": int(team_id2),
                "player_list_1": player_list1,
                "player_list_2": player_list2,
                "events": events,
            }
            match_dict_list.append(match_dict)
        all_events.extend(match_dict_list)
        save_as_jsonlines(match_dict_list, output_dir / f"{label}.jsonl")

    save_as_jsonlines(all_events, output_dir / "events_all.jsonl")


def load_wyscout_data(
    input_dir: Path | str = DATA_DIR / "wyscout/raw/events",
    targets: List[str] = ["events_Spain.json"],
) -> pd.DataFrame:
    data_paths = sorted(Path(input_dir).glob("*.json"))
    df_list = []
    for path in data_paths:
        if path.name not in targets:
            continue
        logger.info(f"Loading {path.name} ...")
        df = pd.read_json(path)
        competition_name = "_".join(path.stem.split("_")[1:])
        df.insert(0, "competition", competition_name)
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    return df
