# DeepGSEA: Explainable Deep Gene Set Enrichment Analysis for Single-cell Transcriptomic Data

DeepGSEA is a deep learning-enhanced gene set enrichment (GSE) analysis method which leverages the expressiveness of interpretable, prototype-based neural networks to provide an in-depth analysis of GSE.

[![Paper](https://img.shields.io/badge/paper-available-brightgreen)](https://academic.oup.com/bioinformatics/article/40/7/btae434/7702331)

## Prerequisite
Install all required packages in `./requirements.txt` (tested on Python 3.9.6)
```script
pip install -r requirements.txt
```

## Quick start

Reproduce results on the glioblastoma data

1. Downlaod gene set databases from MSigDB following `./data/msigdb/download.txt`

2. Download the scRNA-seq data following `./data/GSE132172/download.txt`

3. Move to the directory `./deepgsea`
    ```
    cd ./deepgsea
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
@article{xiong2024deepgsea,
  title={DeepGSEA: Explainable Deep Gene Set Enrichment Analysis for Single-cell Transcriptomic Data},
  author={Xiong, Guangzhi and John LeRoy, Nathaniel and Bekiranov, Stefan and Sheffield, Nathan and Zhang, Aidong},
  journal={Bioinformatics},
  pages={btae434},
  year={2024},
  publisher={Oxford University Press}
}
```
