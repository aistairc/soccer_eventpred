import fire

from soccer_eventpred.data.preprocess.preprocess_wyscout import (
    preprocess_wyscout_events_data, preprocess_wyscout_players_data,
    preprocess_wyscout_tags_data, preprocess_wyscout_teams_data,
    split_wyscout_data)

if __name__ == "__main__":
    fire.Fire(
        {
            "teams": preprocess_wyscout_teams_data,
            "players": preprocess_wyscout_players_data,
            "tags": preprocess_wyscout_tags_data,
            "events": preprocess_wyscout_events_data,
            "split": split_wyscout_data,
        }
    )
