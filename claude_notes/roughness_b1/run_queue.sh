#!/bin/bash
# wait for the first driver (4 exits) then run the follow-ups sequentially
cd "$(dirname "$0")/../.."
until [ "$(grep -c exit claude_notes/roughness_b1/logs/driver.log)" -ge 4 ]; do sleep 30; done
bash claude_notes/roughness_b1/run_pilots.sh wc_hd_b1 greenland_westcoast
bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke_b1_th20 greenland_westcoast
bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke_b1_th40 greenland_westcoast
bash claude_notes/roughness_b1/run_pilots.sh wc_hd_b1fix greenland_westcoast
bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke_b1_th20 greenland_geikie01_transit
bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke_b1_th40 greenland_geikie01_transit
