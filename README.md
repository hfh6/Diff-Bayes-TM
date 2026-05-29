# Network Traffic Matrix Estimation via Diffusion-Based Bayesian Inference

<!-- ## Abstract -->

The traffic matrix (TM) is essential for network management, yet complete measurement is extremely costly. Existing methods infer the full TM from low-cost link loads or a few direct measurements. However, they require a TM prior, which is difficult to obtain given the complexity of real traffic. Moreover, forcing the TM toward the observed posterior often distorts this prior. This paper proposes Diff-Bayes-TM, a Bayesian inference framework that combines diffusion models with Monte Carlo (MC) sampling. Our method first uses a diffusion model to capture the complex TM prior from only a few traffic samples. Then, using MC sampling during the denoising process, we force the generated TM to be unbiasedly close to the posterior distribution. Experiments on two real-world datasets show that Diff-Bayes-TM significantly outperforms existing methods in both inference accuracy and distribution similarity. Notably, our method accurately captures the TM distribution with as few as 5% of the traffic samples for training.

## Requirements

We recommend using Python 3.8 and `virtualenv` or `conda`.

```bash
pip install -r requirements.txt
```

## Training & Evaluation

```bash
python scripts/run_main.py --flow_known_rate <sampling_rate> --dataset <dataset_name> --mode <mode_name> 
```

## Datasets

We use publicly available datasets (e.g., [Abilene](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/) and [G´EANT](https://totem.info.ucl.ac.be/dataset.html)) for experiments. Please download pre-processed traffic matrices and routing matrices from [this URL](https://github.com/duoduoqiao/AutoTomo/tree/main/Data).

## File Structure

```
.
├── Datasets/               # Data .csv files
├── Diffusions/             # Diffusion model definitions
├── Net_params/             # Neural networks
├── scripts/                # Scripts for training/evaluation
├── TME_train_test/         # Helper for training/evaluation
├── Utils/                  # Utility functions
└── requirements.txt        # Required Python packages
```

