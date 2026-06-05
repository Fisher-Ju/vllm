# vllm.v1.request.Request 属性分析文档

## 概述

`Request` 类是 vLLM 中用于表示单个推理请求的核心数据结构。它在请求的整个生命周期中扮演着重要角色，从接收到完成的全过程都被跟踪和管理。

## 请求属性详解

### 基础属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `request_id` | str | 请求唯一标识符，用于区分不同请求 |
| `client_index` | int | 客户端索引，用于确保输出返回给正确的客户端 |
| `priority` | int | 请求优先级，用于调度决策 |
| `sampling_params` | SamplingParams | 采样参数，控制生成行为（如温度、最大长度等） |
| `pooling_params` | PoolingParams | 池化参数，用于池化模型 |
| `lora_request` | LoRARequest | LoRA微调参数，指定使用的LoRA适配器 |
| `structured_output_request` | StructuredOutputRequest | 结构化输出请求，处理特定格式的输出 |
| `arrival_time` | float | 请求到达时间戳，用于调度和统计 |

### 状态相关属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `status` | RequestStatus | 当前请求状态，枚举类型 |
| `events` | list[EngineCoreEvent] | 请求经历的事件记录列表 |
| `stop_reason` | int \| str \| None | 停止原因 |

### 调度与缓存属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `max_tokens` | int | 请求的最大token数 |
| `kv_transfer_params` | dict[str, Any] | KV缓存传输参数 |
| `num_prompt_tokens` | int | 提示token数量 |
| `prompt_token_ids` | list[int] | 提示token ID列表 |
| `prompt_embeds` | torch.Tensor | 提示嵌入向量 |
| `_prompt_embeds_per_block_hashes` | dict[tuple[int, int], bytes] | 每个块提示嵌入的哈希值 |
| `_output_token_ids` | list[int] | 已生成的输出token ID列表 |
| `_all_token_ids` | list[int] | 所有token ID列表（包含提示和输出） |
| `output_token_ids` | ConstantList | 只读输出token ID视图 |
| `all_token_ids` | ConstantList | 只读所有token ID视图 |
| `num_output_placeholders` | int | 输出占位符数量 |
| `discard_latest_async_tokens` | bool | 是否丢弃最新的异步token |
| `spec_token_ids` | list[int] | 规范化token ID列表 |
| `num_computed_tokens` | int | 已计算的token数量 |
| `cache_salt` | str | 缓存盐值，用于缓存键生成 |

### 多模态属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `mm_features` | list[MultiModalFeatureSpec] | 多模态特征列表 |
| `num_encoder_inputs` | int | 编码器输入数量 |
| `has_encoder_inputs` | bool | 是否具有编码器输入 |

### 跟踪与调试属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `trace_headers` | Mapping[str, str] | 追踪HTTP头信息 |
| `is_prefill_chunk` | bool | 是否为非最终预填充块 |
| `num_nans_in_logits` | int | logits中的NaN数量 |
| `num_preemptions` | int | 被抢占次数 |
| `prefill_stats` | PrefillStats | 预填充统计信息 |
| `block_hashes` | list[BlockHash] | 块哈希值列表 |
| `_block_hasher` | Callable[[Request], list[BlockHash]] | 块哈希函数 |

### 流式处理属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `resumable` | bool | 是否可恢复请求 |
| `streaming_queue` | deque[StreamingUpdate \| None] | 流式更新队列 |

## RequestStatus 枚举状态

请求状态用于描述请求在其生命周期中的不同阶段：

1. `WAITING` - 等待处理
2. `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` - 等待结构化输出语法
3. `WAITING_FOR_REMOTE_KVS` - 等待远程KV存储
4. `WAITING_FOR_STREAMING_REQ` - 等待流式请求
5. `RUNNING` - 正在运行
6. `PREEMPTED` - 被抢占
7. `FINISHED_STOPPED` - 因停止而完成
8. `FINISHED_LENGTH_CAPPED` - 因长度限制而完成
9. `FINISHED_ABORTED` - 因中断而完成
10. `FINISHED_IGNORED` - 因忽略而完成
11. `FINISHED_ERROR` - 因错误而完成
12. `FINISHED_REPETITION` - 因重复而完成

## 生命周期流程

### 1. 请求接收阶段
- 接收来自前端的请求（通常通过EngineCoreRequest）
- 创建新的Request对象实例
- 初始化所有基本属性（request_id, sampling_params, pooling_params等）

### 2. 预处理阶段
- 设置请求状态为`WAITING`
- 根据参数设置`max_tokens`值
- 计算并保存提示token的数量和ID
- 初始化输出token列表
- 验证参数有效性（如sampling_params不能同时为空）

### 3. 调度阶段
- 在调度器中根据优先级、到达时间和request_id进行排序
- 状态可能变为`RUNNING`或`PREEMPTED`
- 记录调度相关的事件（如QUEUED、SCHEDULED）

### 4. 执行阶段
- 请求被分配到GPU工作进程执行
- 在Worker中处理模型推理
- 实时更新token输出列表
- 更新块哈希值用于KV缓存

### 5. 完成阶段
- 当达到停止条件（最大token数、停止字符串、错误等）时
- 状态变为相应的完成状态（如FINISHED_STOPPED等）
- 记录完成事件和统计数据
- 清理资源并通知前端

### 6. 流式处理阶段
- 对于支持流式的请求，使用`streaming_queue`处理增量输出
- 支持断点续传功能（通过`resumable`属性）

## 关键方法

| 方法名 | 作用说明 |
|--------|----------|
| `append_output_token_ids` | 添加新的输出token ID |
| `update_block_hashes` | 更新块哈希值 |
| `is_finished` | 判断请求是否已完成 |
| `get_finished_reason` | 获取完成原因 |
| `record_event` | 记录事件 |
| `take_events` | 获取并清空事件列表 |
| `take_prefill_stats` | 获取并清空预填充统计信息 |
| `__lt__` | 比较两个请求的优先级 |