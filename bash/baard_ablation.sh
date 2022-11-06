#!/bin/bash

# This script runs the ablation study for BAARD.

source ./.venv/bin/activate
pip install --upgrade .

# TODO: 
# 1. More espilon on both L2 and Linf
# 2. Reliability: Tuning K on entire set.
# 3. Reliability: Tuning scale on optimal k.
# 4. Decidability: Tuning K on entire set.
# 5. Decidability: Tuning scale on optimal k.
