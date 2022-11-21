# Bash script for Linux

All bash script and terminal commands are prepared for Linux only. Changing the script according when running on a Windows machine.

## Experiment: Find epsilon with min. 95% success rate

1. Run `search_attack_eps.sh` to find minimal epsilon for each dataset.

## Experiment: Grey-box benchmark

1. Run `grey_box_generate_adv.sh` to generate adversarial examples.
2. Run `grey_box_detect.sh` to extract features.

## Experiment: BAARD ablation study

1. Run `baard_generate_adv.sh` to generate adversarial examples.
2. Run `baard_tune_k.sh` to tune the parameter `k_neighbors` for Stage 2 and 3.
3. Run `baard_tune_sample_size.sh` to tune the parameter `sample_size` for Stage 2 and 3 (Use the optimal `k` value from previous step).
4. Run `baard_ablation.sh` to evaluate BAARD on each stage separately.

## Running all experiments

Run `run_all.sh` to run all experiments. The code can read existing results and skip the one that has been computed.
