#!/bin/bash

# This script runs the ablation study for BAARD.

source ./.venv/bin/activate
pip install --upgrade .

# A seed starts with 6 to indicate this is for ablation study. 
# 1-5 are reserved for repeated experiment for grey-box evaluation.
SEED=643896
SIZE=1000

# 2. Reliability: Tuning K on entire set.
# 3. Reliability: Tuning scale on optimal k.
# 4. Decidability: Tuning K on entire set.
# 5. Decidability: Tuning scale on optimal k.
