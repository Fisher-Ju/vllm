# vllm.v1.core.sched.output.SchedulerOutput 类分析文档

## 概述

`SchedulerOutput` 是 vLLM 中用于描述调度器输出结果的核心数据结构。它封装了调度过程中产生的所有必要信息，包括新请求和已缓存请求的数据、调度的token数量、完成请求ID等，是调度器与执行器之间通信的重要桥梁。

## 核心职责

1. **调度结果封装**：将调度决策的结果打包为结构化数据
2. **跨组件通信**：作为调度器与模型执行器之间的数据交换格式
3. **状态同步**：维护请求的调度状态和生命周期信息
4. **性能优化**：通过数据分组减少通信开销

## 主要组成部分

### 1. NewRequestData
用于表示需要首次调度的新请求数据。

### 2. CachedRequestData  
用于表示已经缓存在工作节点上的已调度请求数据。

### 3. SchedulerOutput
主数据结构，封装了完整的调度输出信息。

## SchedulerOutput 类属性详解

### 新请求相关属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `scheduled_new_reqs` | list[NewRequestData] | 首次调度的新请求列表 |
| `scheduled_cached_reqs` | CachedRequestData | 已缓存请求的数据 |

### 调度统计属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `num_scheduled_tokens` | dict[str, int] | 每个请求调度的token数量 |
| `total_num_scheduled_tokens` | int | 所有请求调度的token总数 |
| `scheduled_spec_decode_tokens` | dict[str, list[int]] | 规范化解码token列表 |
| `scheduled_encoder_inputs` | dict[str, list[int]] | 需要处理的编码器输入索引 |

### 状态管理属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `num_common_prefix_blocks` | list[int] | 每个KV缓存组的公共前缀块数量 |
| `finished_req_ids` | set[str] | 本步骤中完成的请求ID集合 |
| `free_encoder_mm_hashes` | list[str] | 需要从编码器缓存中释放的mm_hash列表 |

### 特殊功能属性

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `preempted_req_ids` | set[str] | 本步骤中被抢占的请求ID集合 |
| `has_structured_output_requests` | bool | 是否包含结构化输出请求 |
| `pending_structured_output_tokens` | bool | 是否有待处理的结构化输出token |
| `num_invalid_spec_tokens` | dict[str, int] | 无效规范解码token数量 |
| `kv_connector_metadata` | KVConnectorMetadata | KV缓存连接器元数据 |
| `ec_connector_metadata` | ECConnectorMetadata | EC缓存连接器元数据 |
| `new_block_ids_to_zero` | list[int] | 本步骤中需要清零的新块ID列表 |

## 辅助数据结构

### NewRequestData 类

用于封装新请求的必要信息：

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `req_id` | str | 请求ID |
| `prompt_token_ids` | list[int] | 提示token ID |
| `mm_features` | list[MultiModalFeatureSpec] | 多模态特征 |
| `sampling_params` | SamplingParams | 采样参数 |
| `pooling_params` | PoolingParams | 池化参数 |
| `block_ids` | tuple[list[int], ...] | 块ID列表 |
| `num_computed_tokens` | int | 已计算的token数量 |
| `lora_request` | LoRARequest | LoRA请求 |
| `prompt_embeds` | torch.Tensor | 提示嵌入向量 |
| `prefill_token_ids` | list[int] | 预填充token ID |

### CachedRequestData 类

用于封装已缓存请求的信息：

| 属性名 | 类型 | 作用说明 |
|--------|------|----------|
| `req_ids` | list[str] | 请求ID列表 |
| `resumed_req_ids` | set[str] | 恢复的请求ID集合 |
| `new_token_ids` | list[list[int]] | 新的token ID列表 |
| `all_token_ids` | dict[str, list[int]] | 所有token ID字典 |
| `new_block_ids` | list[tuple[list[int], ...] | 新块ID列表 |
| `num_computed_tokens` | list[int] | 已计算token数量列表 |
| `num_output_tokens` | list[int] | 输出token数量列表 |

## 生命周期流程

### 1. 调度准备阶段
- 调度器根据策略选择要调度的请求
- 构建新请求和缓存请求的数据结构
- 计算各请求的调度token数量

### 2. 数据组织阶段
- 将新请求数据包装为 NewRequestData 对象
- 将已缓存请求数据包装为 CachedRequestData 对象
- 构建完整的 SchedulerOutput 对象

### 3. 传输阶段
- SchedulerOutput 通过IPC或网络传输到执行器
- 执行器根据SchedulerOutput中的信息进行相应的处理

### 4. 执行阶段
- 工作进程使用SchedulerOutput中的数据加载请求
- 根据调度结果执行模型推理
- 处理预填充、解码等不同阶段的请求

### 5. 状态更新阶段
- 记录已完成的请求
- 更新块ID和token数量信息
- 处理被抢占的请求

## 在推理过程中的作用

### 1. 调度决策的载体
SchedulerOutput 是调度算法决策结果的直接体现，包含了：
- 哪些请求被选中进行调度
- 每个请求需要处理多少token
- 请求的初始状态信息

### 2. 执行器的指令集
执行器通过解析 SchedulerOutput 来知道如何处理请求：
- 加载新请求的数据
- 继续处理已有请求
- 管理缓存和块分配

### 3. 状态同步工具
- 传递完成请求信息，让执行器可以清理资源
- 传递需要释放的缓存项
- 同步请求的状态变更

### 4. 性能优化机制
- 通过区分新请求和缓存请求，减少重复数据传输
- 仅传输必要的token ID和块信息
- 支持管道并行和分布式调度

## 核心方法详解

### `make_empty()` 类方法
```python
@classmethod
def make_empty(cls) -> "SchedulerOutput":
```
用途：创建一个空的 SchedulerOutput 实例，用于初始化或错误情况

### CachedRequestData 的辅助方法

#### `is_context_phase()` 方法
```python
def is_context_phase(self, req_id: str) -> bool:
```
用途：判断请求是否处于上下文阶段（即还没有输出token）

#### `num_reqs` 属性
```python
@property
def num_reqs(self) -> int:
```
用途：获取请求数量的便捷属性

#### `_req_id_to_num_output_tokens` 缓存属性
```python
@cached_property
def _req_id_to_num_output_tokens(self) -> dict[str, int]:
```
用途：缓存请求ID到输出token数的映射，提高查找效率

## 使用场景

### 1. 调度器输出
在调度器完成一轮调度后，生成 SchedulerOutput 对象返回给执行器

### 2. 执行器输入
执行器接收 SchedulerOutput 后，根据其中的数据启动相应的推理过程

### 3. 分布式协调
在分布式环境中，SchedulerOutput 作为不同节点间通信的标准格式

### 4. 资源管理
通过 `finished_req_ids` 和 `free_encoder_mm_hashes` 等字段实现资源的及时回收

## 性能考虑

### 1. 数据最小化
- 只传输必要的数据
- 使用 `anon_repr` 方法避免敏感数据泄露
- 区分新请求和缓存请求，减少重复数据传输

### 2. 内存效率
- 使用 `@cached_property` 缓存常用计算结果
- 合理的数据结构选择，如使用 `set` 优化查找性能

### 3. 扩展性
- 支持结构化输出、规范解码等高级特性
- 便于扩展新的调度功能和监控指标

## 与其他组件的交互

### 与调度器的关系
- SchedulerOutput 是调度器的主要输出
- 调度器通过调度算法决定哪些请求应该被包含在输出中

### 与执行器的关系
- 执行器是 SchedulerOutput 的消费者
- 执行器依据 SchedulerOutput 中的信息来安排实际的推理工作

### 与缓存系统的交互
- 通过 `num_common_prefix_blocks` 优化缓存访问
- 通过 `free_encoder_mm_hashes` 管理多模态缓存

### 与监控系统的集成
- 通过各种计数和统计信息提供性能监控数据
- 为调试和优化提供详细的调度行为记录