# DeepGSEA: Explainable Deep Gene Set Enrichment Analysis for Single-cell Transcriptomic Data

DeepGSEA is a deep learning-enhanced gene set enrichment (GSE) analysis method which leverages the expressiveness of interpretable, prototype-based neural networks to provide an in-depth analysis of GSE.

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)]()

## Prerequisite
Install all required packages in `./requirements.txt`

## Quick start

Reproduce results on the glioblastoma data

1. Downlaod gene set databases from MSigDB following `./data/msigdb/download.txt`

1. Download the scRNA-seq data following `./data/GSE132172/download.txt`

2. Move to the directory `./src`
    ```
    cd ./src
    ```
2. Run DeepGSEA on the dataset with interpretations
    ```
    sh ./scripts/run_glioblastoma.sh
    ```
## Citation
If you find our research useful, please consider citing:
```

```