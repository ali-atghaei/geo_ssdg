# Geometry-Aware Representation Learning for Semi-Supervised Domain Generalization

Official implementation of the manuscript:

**“Geometry-Aware Representation Learning for Semi-Supervised Domain Generalization”**  
submitted to *The Visual Computer*.

This repository contains the source code, dataset preprocessing procedures, training and evaluation scripts, and experimental configurations required to reproduce the results reported in the manuscript.

---

# Repository Overview

This project is implemented based on the following repositories:

- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
- [ssdg-benchmark](https://github.com/KaiyangZhou/ssdg-benchmark)

We sincerely thank the authors for publicly releasing their implementations.

---

# Environment Setup

Please first follow the installation and dataset preparation instructions provided in the following repositories:

- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
- [ssdg-benchmark](https://github.com/KaiyangZhou/ssdg-benchmark)

After installing the required dependencies, activate the environment:

```bash
conda activate dassl
```

---

# Dataset Preparation

Prepare the datasets according to the official SSDG benchmark structure described in the referenced repositories.

Ensure that the dataset path is correctly assigned to the `DATA` variable inside:

```bash
/scripts/StyleMatch/run_ssdg.sh
```

---

# Reproducing the Experiments

The main training script is:

```bash
/scripts/StyleMatch/run_ssdg.sh
```

The script accepts two arguments:

- `DATASET`: dataset name
- `NLAB`: total number of labeled samples

## Example

To reproduce the experiments on the OfficeHome dataset under the 10-labels-per-class setting (`1950` labeled samples), run:

```bash
cd scripts/stylematch
bash run_ssdg.sh ssdg_officehome 1950
```

The implementation automatically evaluates all target domains using multiple random seeds, following the experimental protocol described in the manuscript.

If multiple GPUs are available, the script may execute several experiments simultaneously. Users may modify the script to run experiments sequentially if desired.

---

# Evaluation

To parse and summarize the experimental results, run:

```bash
python parse_test_res.py output/ssdg_officehome/nlab_1950/FBCSA/resnet18 --multi-exp
```

---

# Reproducibility Notes

This repository includes:

- source code of the proposed method,
- dataset preprocessing scripts,
- training and evaluation pipelines,
- experiment configurations,
- random-seed-based evaluation settings.

These resources are provided to facilitate transparent and reproducible evaluation of the proposed method.

---

# Citation

If you find this repository useful in your research, please cite the corresponding manuscript:

```bibtex
@article{atghaei2026geometry,
  title={Geometry-Aware Representation Learning for Semi-Supervised Domain Generalization},
  author={Ali Atghaei and Mohammad Rahmati},
  journal={The Visual Computer},
  note={Under review},
  year={2026}
}
```

---

# Acknowledgements

This implementation is partially based on the following works:

- **StyleMatch**  
  *Semi-Supervised Domain Generalization with Stochastic StyleMatch*,  
  International Journal of Computer Vision (IJCV), 2023.

- **FBC-SA**  
  *Towards Generalizing to Unseen Domains with Few Labels*,  
  CVPR 2024.

We gratefully acknowledge the authors for making their code publicly available.
