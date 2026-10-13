# 数据目录

此目录只包含可公开内容：

- `qwbot_v2_alpha_report.json`：Alpha 数据集的聚合统计报告。发布副本移除了逐条聊天预览和本机绝对路径，原始统计值保持不变。
- `data_example.jsonl`：5 条纯虚构、脱敏的 SFT 格式示例。

完整 Alpha 数据规模为训练集 1350 条、验证集 150 条。真实聊天原文、完整训练数据及其元数据不随公开仓库发布。

## 示例格式

JSONL 每行是一个独立 JSON 对象：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

公开样例不含群标识、用户标识、QQ 号、昵称、链接、联系方式或真实聊天原文。
