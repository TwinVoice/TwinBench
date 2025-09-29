# TwinVoice

TwinVoice 是一个对话系统评测框架，支持多维度的对话能力评估。

## 项目结构

```
TwinVoice/
├── dataset/                    # 评测数据集
│   ├── dimension_2/           
│   │   └── data.jsonl         
│   └── dimension_3/           
│       ├── choices.jsonl      # 多选题数据
│       └── profiles.jsonl     # 角色资料
├── twinvoice/                 # 主代码包
│   ├── api_config.py         # API配置
│   ├── discriminative/       # 判别式评测
│   │   └── dimension_3/     # 维度3评测模块
│   │       └── evaluate.py  # 评测主脚本
│   └── generative/          # 生成式评测
└── result/                   # 评测结果
    └── discriminative/       # 判别式评测结果
        └── dimension_3/     # 维度3评测结果
            ├── results.jsonl       # 评测详细结果
            ├── wrong_cases.jsonl   # 错误案例
            └── capability_report.csv # 能力维度分析
```

## 使用方法

### 维度3评测（多选题形式）

在项目根目录下运行：

```bash
# 基础评测（使用默认路径）
python -m twinvoice.discriminative.dimension_3.evaluate \
    --model qwen-max

# 指定数据路径
python -m twinvoice.discriminative.dimension_3.evaluate \
    dataset/dimension_3/choices.jsonl \
    dataset/dimension_3/profiles.jsonl \
    --model qwen-max

# 带能力维度分析的评测
python -m twinvoice.discriminative.dimension_3.evaluate \
    --model qwen-max \
    --annotations annotated.jsonl
```

### 主要参数说明

- `choices_jsonl`: 多选题数据文件（默认：dataset/dimension_3/choices.jsonl）
- `profile_json`: 角色资料文件（默认：dataset/dimension_3/profiles.jsonl）
- `--model`: 使用的模型名称（默认：gpt-4o-mini）
- `--sample`: 采样数量（可选）
- `--report`: 评测结果文件（默认：result/discriminative/dimension_3/results.jsonl）
- `--wrong-report`: 错误案例文件（默认：result/discriminative/dimension_3/wrong_cases.jsonl）
- `--temperature`: 采样温度（默认：0.0）
- `--history-max`: 历史对话最大条数（默认：30）
- `--annotations`: 能力标注文件（可选）
- `--cap-report-csv`: 能力维度报告（默认：result/discriminative/dimension_3/capability_report.csv）

### 评测能力维度

- Opinion_Consistency（观点一致性）
- Memory_Recall（记忆回溯）
- Logical_Reasoning（逻辑推理）
- Lexical_Fidelity（词汇忠实度）
- Persona_Tone（人物语气）
- Syntactic_Style（句法风格）

### 配置说明

1. 复制示例配置文件：
```bash
cp twinvoice/api_config.example.py twinvoice/api_config.py
```

2. 编辑 `twinvoice/api_config.py` 配置您的API密钥：
```python
# SVIP API配置
api_key = 'your-api-key-here'  # 替换为您的API密钥
svip_base_url = 'https://svip.xty.app/v1'

# 本地API配置
local_base_url = 'http://localhost:8005/v1'  # 如果使用其他端口，请修改
local_api_key = 'EMPTY'
```

### 注意事项

1. 确保在项目根目录下运行评测命令
2. 运行前确保已正确配置 API 密钥（复制并修改 `api_config.example.py`）
3. 本地模型评测需要启动本地服务器（默认端口8005）
4. 评测过程支持优雅退出（Ctrl+C）
5. 建议先使用小样本测试配置是否正确
6. 不要提交 `api_config.py` 到版本控制系统

## 输出说明

1. 基础评测输出：
   - 总体准确率
   - 错误案例分析（最多显示20条）
   - 详细结果保存在report文件中

2. 能力维度报告：
   - 各能力维度的样本数量
   - 各能力维度的准确率
   - 按样本数量和准确率排序的统计表