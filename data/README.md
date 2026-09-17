# DKA 数据集说明

本目录存放 DKA 项目的各领域数据（**不入 git**,`.gitignore` 已排除）。

## 目录结构与统计

```
data/
├── linguistic/                        # 语言学领域(论文主实验)
│   ├── corpus/linguistic.jsonl        # 语料:6,145 条,{"text": ...},用于 dka-build-kg
│   ├── sft/step1_sft.jsonl            # Stage 1 训练数据:100,000 条,{"prompt", "response"}
│   ├── sft/step2_sft.jsonl            # Stage 2 训练数据:44,006 条,{"prompt", "response"}
│   └── benchmark.jsonl                # 评测集:1,801 条,{"question", "ground_truth", "type", "difficulty"}
├── law/                               # 法律领域(ModelScope 下载)
│   ├── Legal/                         # peopletech/Legal — 法律领域语料库.zip (~424MB)
│   ├── wenshu_dataset/                # KLGR123/wenshu_dataset — 裁判文书 JSON (~42MB)
│   └── Chinese_Law/                   # KuugoRen/Chinese_Law — 法律条文 txt (~3MB)
└── musique/                           # MuSiQue 官方 v1.0 原始数据
    ├── musique_ans_v1.0_{train,dev,test}.jsonl   # answerable: 19,938 / 2,417 / 2,459
    ├── musique_full_v1.0_{train,dev,test}.jsonl  # full(含不可回答): 39,876 / 4,834 / 4,918
    └── dev_test_singlehop_questions_v1.0.json
```

## 数据来源

| 数据集 | 来源 |
|---|---|
| linguistic | 内部整理(to_open.zip) |
| Law: Legal | https://www.modelscope.cn/datasets/peopletech/Legal |
| Law: wenshu_dataset | https://www.modelscope.cn/datasets/KLGR123/wenshu_dataset |
| Law: Chinese_Law | https://www.modelscope.cn/datasets/KuugoRen/Chinese_Law |
| MuSiQue | https://github.com/StonyBrookNLP/musique (TACL 2022) |

## 注意事项

- **MuSiQue test 集无答案**(官方 leaderboard 盲测),评估请用 dev(answerable 2,417 条)
- linguistic benchmark 含 `type`(tf 判断题等)与 `difficulty` 字段,评估时需按类型分别处理
- `linguistic/corpus/linguistic.jsonl` 可直接作为 `dka-build-kg --input` 的输入(含 `text` 字段)
- `step1/step2_sft.jsonl` 是 `prompt`/`response` 格式,用于 `dka-train` 前需转换为 ShareGPT `conversations` 格式
