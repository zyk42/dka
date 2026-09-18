# DKA 数据集说明

本目录存放 DKA 项目的数据（**不入 git**,`.gitignore` 已排除，`data/README.md` 除外）。

## Linguistic 语言学数据（论文主实验）

完整语言学语料已上传到 ModelScope：

**https://www.modelscope.cn/datasets/vivalavida21k/linguistic_data**

该数据集包含原始语料与补充语料，合计 **31,954 条**（原语料 6,145 条 + 补充语料 25,809 条）。

```
data/
└── linguistic/
    ├── corpus/linguistic.jsonl        # 本地语料:6,145 条,{"text": ...}
    ├── sft/step1_sft.jsonl            # Stage 1 训练数据:100,000 条,{"prompt", "response"}
    ├── sft/step2_sft.jsonl            # Stage 2 训练数据:44,006 条,{"prompt", "response"}
    └── benchmark.jsonl                # 评测集:1,801 条,{"question", "ground_truth", "type", "difficulty"}
```

从 ModelScope 下载：

```bash
pip install modelscope
modelscope download --dataset vivalavida21k/linguistic_data \\
    --local_dir data/linguistic_modelscope
```

下载后可将语料 JSONL 作为 `dka-build-kg --input` 的输入。

## 其他数据集（请自行下载后放入本目录）

### Law 法律领域（ModelScope)

| 数据集 | 链接 | 说明 |
|---|---|---|
| Legal | https://www.modelscope.cn/datasets/peopletech/Legal | 法律领域语料库（~424MB zip) |
| wenshu_dataset | https://www.modelscope.cn/datasets/KLGR123/wenshu_dataset | 裁判文书 JSON |
| Chinese_Law | https://www.modelscope.cn/datasets/KuugoRen/Chinese_Law | 中国法律条文 txt |

下载方式（任选其一）:

```bash
# ModelScope CLI
pip install modelscope
modelscope download --dataset peopletech/Legal --local_dir data/law/Legal
modelscope download --dataset KLGR123/wenshu_dataset --local_dir data/law/wenshu_dataset
modelscope download --dataset KuugoRen/Chinese_Law --local_dir data/law/Chinese_Law

# 或直接网页下载后解压到 data/law/ 下
```

### MuSiQue(多跳 QA 基准）

| 项目 | 链接 |
|---|---|
| 官方仓库与下载 | https://github.com/StonyBrookNLP/musique (TACL 2022) |

下载后放置于 `data/musique/`，预期文件：

```
data/musique/
├── musique_ans_v1.0_{train,dev,test}.jsonl    # answerable: 19,938 / 2,417 / 2,459 条
├── musique_full_v1.0_{train,dev,test}.jsonl   # full(含不可回答): 39,876 / 4,834 / 4,918 条
└── dev_test_singlehop_questions_v1.0.json
```

## 注意事项

- **MuSiQue test 集无答案**（官方 leaderboard 盲测），评估请用 dev(answerable 2,417 条）
- linguistic benchmark 含 `type`(tf 判断题等）与 `difficulty` 字段，评估时需按类型分别处理
- `step1/step2_sft.jsonl` 是 `prompt`/`response` 格式，用于 `dka-train` 前需转换为 ShareGPT `conversations` 格式
