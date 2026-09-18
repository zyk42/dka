# DKA 数据集说明

本目录存放 DKA 项目的语料数据（**不入 git**，`.gitignore` 已排除，`data/README.md` 除外）。

## Linguistic 语言学语料

完整语言学语料已上传到 ModelScope：

**https://www.modelscope.cn/datasets/vivalavida21k/linguistic_data**

该数据集只包含语言学语料，合计 **31,954 条**：

- 原始语料：6,145 条
- 补充语料：25,809 条

从 ModelScope 下载：

```bash
pip install modelscope
modelscope download \
    --dataset vivalavida21k/linguistic_data \
    --local_dir data/linguistic_modelscope
```

下载后可直接将其中的 `linguistic_merged.jsonl` 作为 `dka-build-kg --input` 的输入。该 JSONL 每行是一条语料记录，包含 `text`、`score`、`linguistics_score` 字段。

本地项目中保留的语料：

```text
data/linguistic/corpus/linguistic.jsonl
```

## 其他数据集（请自行下载）

### Law 法律领域（ModelScope）

| 数据集 | 链接 | 说明 |
|---|---|---|
| Legal | https://www.modelscope.cn/datasets/peopletech/Legal | 法律领域语料库 |
| wenshu_dataset | https://www.modelscope.cn/datasets/KLGR123/wenshu_dataset | 裁判文书 JSON |
| Chinese_Law | https://www.modelscope.cn/datasets/KuugoRen/Chinese_Law | 中国法律条文 txt |

下载方式：

```bash
pip install modelscope
modelscope download --dataset peopletech/Legal --local_dir data/law/Legal
modelscope download --dataset KLGR123/wenshu_dataset --local_dir data/law/wenshu_dataset
modelscope download --dataset KuugoRen/Chinese_Law --local_dir data/law/Chinese_Law
```

### MuSiQue（多跳 QA 基准）

官方仓库与下载地址：

https://github.com/StonyBrookNLP/musique

MuSiQue 数据下载后可放置到 `data/musique/`。官方 test 集不公开答案，评估建议使用带答案的 dev 集。
