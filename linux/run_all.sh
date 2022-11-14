#!/bin/bash

# This script run all experiements. However, it can read existing results and skip the one that has been computed.

# echo "Generating adversarial examples for BAARD ablation study ######################"
# bash ./linux/baard_generate_adv.sh

# echo "Tuning K for BAARD ############################################################"
# bash ./linux/baard_tune_k.sh

# echo "[NOTE] Update K value and then run the next! ##################################"

# echo "Tuning Scale for BAARD ########################################################"
# bash ./linux/baard_tune_scale.sh

# echo "[NOTE] Update K and Scale and then run the next! ##############################"
# bash ./linux/baard_ablation.sh
# echo "BAARD ablation study has completed! ###########################################"

echo "[NOTE] K and Scale for BAARD need to be the optimal value! ####################"
echo "Generating adversarial examples for grey-box benchmark ########################"
bash ./linux/grey_box_generate_adv.sh

echo "Runing rey-box benchmark ######################################################"
bash ./linux/grey_box_detect.sh
echo "Grey-box benchmark has completed for 1 repeataion! ############################"
