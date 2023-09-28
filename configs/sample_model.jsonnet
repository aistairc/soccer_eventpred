local time_encoder_dim = 10;
local team_encoder_dim = 10;
local event_encoder_dim = 10;
local player_encoder_dim = 10;
local x_axis_encoder_dim = 10;
local y_axis_encoder_dim = 10;
local seq2vec_input_dim = time_encoder_dim + team_encoder_dim + event_encoder_dim + player_encoder_dim + x_axis_encoder_dim * 2 + y_axis_encoder_dim * 2;



{
    "seq2vec_encoder": {
        "type": "gru",
        "input_size": seq2vec_input_dim,
        "hidden_size": 32,
        "num_layers": 1,
        "bidirectional": true,
    },
    "time_encoder": {
        "type": "embedding",
        "embedding_dim": time_encoder_dim,
        "num_embeddings": 121,
        "padding_idx": 120,
    },
    "team_encoder": {
        "type": "embedding",
        "embedding_dim": team_encoder_dim,
    },
    "event_encoder": {
        "type": "embedding",
        "embedding_dim": event_encoder_dim,
    },
    "player_encoder": {
        "type": "embedding",
        "embedding_dim": player_encoder_dim,
    },
    "x_axis_encoder": {
        "type": "embedding",
        "embedding_dim": x_axis_encoder_dim,
        "num_embeddings": 102,
        "padding_idx": 101,
    },
    "y_axis_encoder": {
        "type": "embedding",
        "embedding_dim": y_axis_encoder_dim,
        "num_embeddings": 102,
        "padding_idx": 101,
    },
    "optimizer": {
        "type": "torch::AdamW",
        "lr": 1e-5,
    },
    "scheduler": {
        "type": "torch::LinearLR",
    },
    "batch_size": 128
}
