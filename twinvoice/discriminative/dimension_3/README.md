# TwinVoice 评测模块 - Dimension 3

本模块实现了基于多选题的对话系统评测功能，主要用于评估模型在多个维度上的对话能力。

## 功能特点

- 基于多选题(MCQ)的评测方式
- 支持多个维度的能力评估：
  - Opinion_Consistency（观点一致性）
  - Memory_Recall（记忆回溯）
  - Logical_Reasoning（逻辑推理）
  - Lexical_Fidelity（词汇忠实度）
  - Persona_Tone（人物语气）
  - Syntactic_Style（句法风格）
- 支持本地和远程API模型评测
- 提供详细的评测报告和错误分析

## 文件结构

- `evaluate.py`: 主评测脚本
- `utils.py`: 工具函数模块
- `README.md`: 本说明文档

## 使用方法

### 基础评测

```bash
python twinvoice/discriminative/dimension_3/evaluate.py dataset/dimension_3/choices.jsonl dataset/dimension_3/profiles.jsonl --model '/common/users/mg1998/models/Qwen2.5-14B-Instruct' --sample 5 --report result/discriminative/dimension_3/results_test.jsonl --temperature 0.0
```

### 主要参数说明

- `choices.jsonl`: 多选题数据文件（位于 Dataset/dimension_3/choices.jsonl）
- `profiles.json`: 角色资料文件（位于 Dataset/dimension_3/profiles.json）
- `--model`: 使用的模型名称
- `--sample`: 采样数量（可选）
- `--report`: 输出评测结果的JSONL文件
- `--temperature`: 采样温度（默认0.0）
- `--history-max`: 历史对话最大条数（默认30）
- `--annotations`: 能力标注文件
- `--cap-report-csv`: 能力维度报告输出文件

### 评测数据格式

choices.jsonl 示例：
```json
{
    "chunk_id": "123",
    "speaker": "角色名",
    "context": "对话上下文",
    "mcq": {
        "options": {
            "A": "选项A内容",
            "B": "选项B内容",
            "C": "选项C内容",
            "D": "选项D内容"
        },
        "answer": "A"
    }
}
```

### 注意事项

1. 确保已正确配置API密钥（在 `api_config.py` 中）
2. 本地模型评测需要启动本地服务器（默认端口8005）
3. 评测过程支持优雅退出（Ctrl+C）
4. 建议先使用小样本测试配置是否正确

## 输出说明

1. 基础评测输出：
   - 总体准确率
   - 错误案例分析（最多显示20条）
   - 详细结果保存在report文件中

2. 能力维度报告：
   - 各能力维度的样本数量
   - 各能力维度的准确率
   - 按样本数量和准确率排序的统计表
