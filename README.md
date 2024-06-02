# **Snoopy: Effective and Efficient Semantic Join Discovery via Proxy Columns**
(https://github.com/ZJU-DAILY/Snoopy/blob/main/snoopy.jpg)
Snoopy: an effective and efficient semantic join discovery framework powered by proxy-column-based column embeddings. The proposed column embeddings are obtained from the column-to-proxy-column relationships captured by a lightweight approximate-graph-matching-based column projection function. To acquire good pivot columns for guiding the column projection process, a rank-aware contrastive learning paradigm is introduced.
## Requirements

* Python 3.7
* PyTorch 1.10.1
* CUDA 11.5
* NVIDIA 3090 GPU

Please refer to the source code to install all required packages in Python.

## Datasets
We use WikiTable, Opendata, and WDC. We provide our [experimental datasets](https://drive.google.com/drive/folders/19vwb45WCayF2j8oPOFf2QVHVopIrgFva?usp=sharing). 

## Run Experimental Case
To construct training data:

```
python DataGen.py --datasets "WikiTable" --type mat --tau 0.2 --list_size 3
```

To learn proxy columns using the generated data:

```
python train.py --datasets "WikiTable" --type mat --tau 0.2 --list_size 3 --version Your_Model_Version
```

To perform semantically join search via learned proxy columns:

```
python search.py --datasets "WikiTable" --version Your_Model_Version --topk 25
```

## Parameters
- `--datasets`: the dataset used (e.g., "WikiTable")

- `--type`: which data generation strategy to be used ("mat" means embedding-level, and "text" means text-level)

- `--tau`: the threshold of cell matching

- `--list_size`: the size of the positive ranking list

- `--version`: the model version you saved during the training phase and used for online search

- `--topk`: top-k joinable columns will be returned


## Acknowledgementt
The original datasets are form [WikiTable](http://websail-fe.cs.northwestern.edu/TabEL/), [opendata](https://arxiv.org/pdf/2209.13589.pdf), and [WDC Web Table Corpus](http://webdatacommons.org/webtables/2015/downloadInstructions.html).

The baseline [Deepjoin](https://www.vldb.org/pvldb/vol16/p2458-dong.pdf) is implemented with the details provided by the authors after contacting them.
