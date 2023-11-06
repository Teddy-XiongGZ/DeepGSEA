# DeepGSEA: Explainable Deep Gene Set Enrichment Analysis for Single-cell Transcriptomic Data

DeepGSEA is a deep learning-enhanced gene set enrichment (GSE) analysis method which leverages the expressiveness of interpretable, prototype-based neural networks to provide an in-depth analysis of GSE.

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2023.11.03.565235)

## Prerequisite
Install all required packages in `./requirements.txt`

## Quick start

Reproduce results on the glioblastoma data

1. Downlaod gene set databases from MSigDB following `./data/msigdb/download.txt`

2. Download the scRNA-seq data following `./data/GSE132172/download.txt`

3. Move to the directory `./src`
    ```
    cd ./src
    ```
4. Run DeepGSEA on the dataset with interpretations
    ```
    sh ./scripts/run_glioblastoma.sh
    ```
5. Run the sigificance test on the results
    ```
    python pvalue_real.py --data glioblastoma
    ```
## Citation
If you find our research useful, please consider citing:
```
@article {xiong2023deepgsea,
	title = {DeepGSEA: Explainable Deep Gene Set Enrichment Analysis for Single-cell Transcriptomic Data},
	author = {Guangzhi Xiong and Nathan John LeRoy and Stefan Bekiranov and Aidong Zhang},
	journal = {bioRxiv},
	year = {2023},
	publisher = {Cold Spring Harbor Laboratory}
}
```