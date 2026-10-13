# 四模型实验分析：test5_20261009

- 盲映射 seed：`20261009`
- 盲评行数：5

## 对齐与数据质量

- 对齐测试键数：5
- 重复记录数（合计）：0
- 缺失记录数（合计）：0
- 错误/格式问题：0

## 模型统计

| 模型 |总测试|有效|未完成|错误|命中上限|平均秒|平均生成Token|平均峰值显存GB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_1p7b_non_thinking | 5 | 5 | 0 | 0 | 0 | 2.092 | 25.0 | 1.354 |
| 02_1p7b_thinking | 5 | 5 | 0 | 0 | 0 | 23.381 | 379.8 | 1.403 |
| 03_4b_non_thinking | 5 | 5 | 0 | 0 | 0 | 4.592 | 45.6 | 3.376 |
| 04_4b_thinking | 5 | 4 | 1 | 0 | 1 | 125.398 | 1691.4 | 3.624 |

## 历史上下文分组

### with_history
- 01_1p7b_non_thinking: {'total_test_count': 5, 'unique_test_count': 5, 'valid_response_count': 5, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': 2.092, 'average_generated_tokens': 25.0, 'average_peak_vram_gb': 1.354, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
- 02_1p7b_thinking: {'total_test_count': 5, 'unique_test_count': 5, 'valid_response_count': 5, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': 23.381, 'average_generated_tokens': 379.8, 'average_peak_vram_gb': 1.403, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
- 03_4b_non_thinking: {'total_test_count': 5, 'unique_test_count': 5, 'valid_response_count': 5, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': 4.592, 'average_generated_tokens': 45.6, 'average_peak_vram_gb': 3.376, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
- 04_4b_thinking: {'total_test_count': 5, 'unique_test_count': 5, 'valid_response_count': 4, 'incomplete_count': 1, 'error_count': 0, 'hit_token_limit_count': 1, 'average_generate_seconds': 125.398, 'average_generated_tokens': 1691.4, 'average_peak_vram_gb': 3.624, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
### without_history
- 01_1p7b_non_thinking: {'total_test_count': 0, 'unique_test_count': 0, 'valid_response_count': 0, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': None, 'average_generated_tokens': None, 'average_peak_vram_gb': None, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
- 02_1p7b_thinking: {'total_test_count': 0, 'unique_test_count': 0, 'valid_response_count': 0, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': None, 'average_generated_tokens': None, 'average_peak_vram_gb': None, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
- 03_4b_non_thinking: {'total_test_count': 0, 'unique_test_count': 0, 'valid_response_count': 0, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': None, 'average_generated_tokens': None, 'average_peak_vram_gb': None, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
- 04_4b_thinking: {'total_test_count': 0, 'unique_test_count': 0, 'valid_response_count': 0, 'incomplete_count': 0, 'error_count': 0, 'hit_token_limit_count': 0, 'average_generate_seconds': None, 'average_generated_tokens': None, 'average_peak_vram_gb': None, 'duplicate_records': 0, 'duplicate_keys': [], 'missing_keys': [], 'invalid_key_records': 0}
