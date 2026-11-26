# Qwen3.5-9B 的 agentic SFT 工具调用 loss mask

已处理 `sft_stage1_hx.jsonl`，输出在 `data/sft_stage1_hx_loss_mask/`。原始文件未修改，全部消息、工具返回、轨迹顺序和元数据均保留，只给明确错误的工具调用增加了 `"loss": false, "loss_scale": 0.0`。

## 为什么要屏蔽

SFT 的交叉熵会鼓励模型复现被监督的 token。若训练目标包含错误工具名、缺失参数、无效计算表达式或不存在的存储 key，训练也会鼓励模型复现这些调用。最终答对并不代表中间每一步都适合模仿。

mask 表示这些调用 token 的目标标签设为 `-100`，不对其直接计算交叉熵；调用、错误返回仍留在输入中，后续成功调用和纠正回答继续参与训练，因此模型可以学习如何从错误状态恢复。不要删除报错上下文，也不要把对应的 `attention_mask` 设为 0。被屏蔽的调用仍可通过上下文影响后续监督；mask 不等于对错误行为施加负奖励。

ms-swift 的 `default` loss 策略会屏蔽 system/user/tool 返回，监督 assistant 和工具调用。逐消息 `loss` 字段控制额外屏蔽。[官方数据格式](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)、[官方参数说明](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)。

## 实际结果

| 项目 | 数量 |
|---|---:|
| 保留轨迹 | 6,397 |
| 工具调用总数 | 231,332 |
| 包含被屏蔽调用的轨迹 | 62 |
| 被屏蔽调用 | 90 |
| 其中 retrieve_information | 45 |
| 其中 calculator | 35 |
| 其中 edgar_search | 3 |
| 其中 submit_final_result | 7 |
| 含成功和错误调用的混合并行批次 | 29 |
| 执行失败返回，默认保留调用监督 | 807 |
| 待复核返回，默认保留调用监督 | 3,513 |

默认屏蔽明确的调用错误：按本条记录中的工具 schema 验证必要参数、类型、枚举、数值上下限；验证日期格式和顺序；识别工具实际返回的无效表达式、空存储、错误 key/引用、输入超过工具上下文窗口等诊断。

超时、网络、鉴权、限流或服务额度问题不直接证明调用决策有错。HTTP 400/404/422 原因不清、搜索为空、价格数据为空、检索为空等写入审计文件，默认保留调用的监督。只匹配返回开头的明确诊断，不因为文档正文出现 `error`、`failed` 或 `not found` 就屏蔽调用。最终提交工具正常的空返回也不算错误。

本次处理没有自动判断“工具执行成功但查错公司/年份”“无关搜索”“最终财务结论有误”等语义质量，也没有屏蔽错误调用前的 assistant 思考内容。这些需要另外审查，不能仅凭返回里是否报错判断。

## ms-swift 的并行调用问题

原生处理会合并连续 `tool_call` 消息，合并后只保留一组 loss 设置。因此直接在单个并行调用上写 `loss: false`，可能漏屏蔽错误调用或连带屏蔽成功调用。[官方说明](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#supervised-fine-tuning)。

随附 `scripts/swift_per_call_loss.py` 外部插件，使用安装环境自己的 agent formatter 渲染调用，把混合并行批次拆成具有独立 loss 设置的 assistant 片段，再交给原生模板合并。插件核对拼接后的调用文本与原生渲染一致，保留并行批次中的成功调用监督。工具返回仍由原生模板处理。请使用 `swift` 模板后端，加载此插件；不要只使用数据文件而省略插件。

## 使用方法

在项目根目录、安装有近期 ms-swift 4.x 且支持 Qwen3.5 的训练环境中，先运行实际 tokenizer 检查；也可以把 `--model` 替换为你的本地模型目录：

```bash
python scripts/check_swift_loss_mask.py \
  --model Qwen/Qwen3.5-9B \
  --dataset data/sft_stage1_hx_loss_mask/sft_stage1_hx.masked.swift.jsonl
```

检查脚本只加载 processor/template，不加载模型权重。它验证错误调用在并行批次首位、末位、整批错误和整批成功四种情况；确认错误调用仍在输入中但不出现在监督标签中，成功调用、推理和纠正回答仍受监督，工具返回不受监督，并抽检真实轨迹的输入 token 未改变。若检查失败，先解决框架版本/模板兼容性。

在你现有的 `swift sft` 命令中使用以下参数，其余 LoRA、GPU、学习率和保存参数沿用你的训练配置：

```bash
swift sft \
  --model Qwen/Qwen3.5-9B \
  --dataset data/sft_stage1_hx_loss_mask/sft_stage1_hx.masked.swift.jsonl \
  --agent_template qwen3_5 \
  --template_backend swift \
  --external_plugins scripts/swift_per_call_loss.py \
  --loss_scale default \
  --is_binary_loss_scale true
```

`default` 用于监督所有 assistant 步骤。实际序列长度须在你的 tokenizer 和训练配置中测量，再设置 `max_length`、截断或切分；过短的截断可能丢掉后续纠正和最终回答。检查脚本会显示抽检轨迹的 token 数，但没有替你选择训练长度或超参数。

## 重现与审计

预处理只依赖 Python 标准库，直接运行文件即可，不需要安装训练库：

```bash
python finance_agent/mask_sft_tool_calls.py \
  --input sft_stage1_hx.jsonl \
  --output-dir data/sft_stage1_hx_loss_mask_new
```

输出目录必须新建或为空，避免覆盖已有结果。若实验确实要求屏蔽所有明确执行失败和原因不明的 HTTP 错误，可另外加 `--mask-execution-failures`，输出到另一个目录；空搜索/空价格结果仍不据此判错。

输出文件：

- `sft_stage1_hx.masked.swift.jsonl`：训练数据。
- `tool_call_audit.jsonl`：逐调用审计，含轨迹 ID、原文件行号、0-based 消息索引、原因、是否屏蔽、调用内容和返回摘录。
- `summary.json`：统计、策略、文件 SHA-256 和已知边界。
- `verification.json`：全量 6,397 条记录的内容/顺序及哈希校验结果。
- `source_integration_check.json`：官方源码方法的并行片段及标签检查结果。

已通过 13 项自动测试、ruff 检查、全量输出一致性检查，以及四种并行情况的官方源码方法检查。源码方法检查使用隔离的字节 tokenizer，不等价于真实 Qwen tokenizer 验证。当前机器没有 ms-swift/Transformers/PyTorch，真实 tokenizer 检查与模型训练尚未运行；应在训练环境执行上面的检查脚本。
