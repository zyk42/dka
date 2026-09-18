# DKA Datasets

This directory contains corpus data for the DKA project. The data files are
excluded from Git by `.gitignore`; this README is tracked separately.

## Linguistic Corpus

The complete linguistic corpus is available on ModelScope:

**https://www.modelscope.cn/datasets/vivalavida21k/linguistic_data**

The dataset contains only linguistic-domain corpus data, with **31,954 records**
in total:

- Original corpus: 6,145 records
- Supplementary corpus: 25,809 records

Download it with:

```bash
pip install modelscope
modelscope download \
    --dataset vivalavida21k/linguistic_data \
    --local_dir data/linguistic_modelscope
```

The downloaded `linguistic_merged.jsonl` can be passed directly to
`dka-build-kg --input`. Each JSONL record contains `text`, `score`, and
`linguistics_score` fields.

The local project corpus is stored at:

```text
data/linguistic/corpus/linguistic.jsonl
```

## Other Datasets

### Law Datasets (ModelScope)

| Dataset | Link | Description |
|---|---|---|
| Legal | https://www.modelscope.cn/datasets/peopletech/Legal | Legal-domain corpus |
| wenshu_dataset | https://www.modelscope.cn/datasets/KLGR123/wenshu_dataset | Chinese court judgment JSON data |
| Chinese_Law | https://www.modelscope.cn/datasets/KuugoRen/Chinese_Law | Chinese legal text files |

Download commands:

```bash
pip install modelscope
modelscope download --dataset peopletech/Legal --local_dir data/law/Legal
modelscope download --dataset KLGR123/wenshu_dataset --local_dir data/law/wenshu_dataset
modelscope download --dataset KuugoRen/Chinese_Law --local_dir data/law/Chinese_Law
```

### MuSiQue (Multi-hop QA Benchmark)

Official repository and download page:

https://github.com/StonyBrookNLP/musique
