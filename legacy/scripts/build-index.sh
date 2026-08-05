#!/bin/bash

PYTHON=/home/yacine/anaconda3/envs/wemap-vision/bin/python

# python -m pipeline.offline.build_index \
#   --map_path "$VPS_DATA_DIR/maps/vinci-st-domingue" \
#   --config pipeline/config/config_hybrid_airport.yaml \
#   --limit 50

# python -m pipeline.offline.build_index \
#   --map_path "$VPS_DATA_DIR/maps/bbhotel-choisy" \
#   --config pipeline/config/config_hybrid_hotel.yaml

$PYTHON -m pipeline.offline.build_index \
    --map_path "$VPS_DATA_DIR/maps/sncf-paris-gare-du-nord" \
    --config pipeline/config/config_hybrid_train_station.yaml

# $PYTHON -m pipeline.offline.build_index \
#     --map_path "$VPS_DATA_DIR/maps/galeries-lafayette-saint-laurent-du-var" \
#     --config pipeline/config/config_retail.yaml