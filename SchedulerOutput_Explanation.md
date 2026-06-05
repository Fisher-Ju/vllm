# vllm.v1.core.sched.output.SchedulerOutput 类详解

## 概述

`SchedulerOutput` 是 vLLM 引擎中调度器的核心输出数据结构，用于描述在一次调度周期内需要执行的任务和状态变化。它包含了调度过程中所有必要的信息，以便将请求分发到工作进程和更新系统状态。

## 主要功能

该类的主要职责是：
1. **请求调度信息**：定义哪些请求需要被调度执行
2. **状态管理**：跟踪请求的状态变更和完成情况
3. **资源分配**：管理内存块和其他资源的分配与回收
4. **特殊处理**：支持多步预测、结构化输出等高级特性

## 核心属性说明

### 1. 调度请求数据

```python
scheduled_new_reqs: list[NewRequestData]
```
- 新加入的请求列表，这些请求在本次调度周期首次被处理
- 包含请求的提示词、采样参数、块ID等必要信息

```python
scheduled_cached_reqs: CachedRequestData  
```
- 已缓存的请求列表，这些请求之前已经被处理过
- 为了减少通信开销，只传输差异部分的数据

### 2. 调度统计信息

```python
num_scheduled_tokens: dict[str, int]
```
- 每个请求安排的token数量映射

```python
total_num_scheduled_tokens: int
```
- 所有请求安排的token总数

### 3. 特殊调度特性

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
```
- 用于多步预测（Speculative Decoding）的token列表

```python
scheduled_encoder_inputs: dict[str, list[int]]
```
- 需要进行编码器输入处理的请求及其索引

### 4. 缓存和内存管理

```python
num_common_prefix_blocks: list[int]
```
- 每个KV缓存组中所有请求的公共前缀块数

```python
finished_req_ids: set[str]
```
- 在当前调度周期内完成的请求ID集合

```python
free_encoder_mm_hashes: list[str]
```
- 需要从编码器缓存中释放的多媒体哈希值列表

### 5. 预占和特殊状态

```python
preempted_req_ids: set[str] | None
```
- 在当前步骤中被抢占的请求ID集合（仅用于v2模型运行时）

```python
has_structured_output_requests: bool
```
- 是否存在使用结构化输出的请求

```python
pending_structured_output_tokens: bool
```
- 请求是否已经获得了进行语法位掩码计算所需的所有输出token

### 6. 调试和监控

```python
num_invalid_spec_tokens: dict[str, int] | None
```
- 无效的spec decode token计数

```python
new_block_ids_to_zero: list[int] | None
```
- 在本次调度步骤中新分配的块ID，需要在GPU上清零以防止旧数据干扰

## 使用场景

### 1. 调度器工作流程
- 调度器根据各种策略选择待处理的请求
- 创建SchedulerOutput对象来包装这些请求
- 将SchedulerOutput发送给工作进程执行

### 2. 请求生命周期管理
- 跟踪新请求的添加和现有请求的继续执行
- 管理请求完成状态的更新
- 清理已完成请求占用的资源

### 3. 高级特性支持
- 多步预测：处理speculative decoding需要的特殊token
- 结构化输出：处理需要语法约束的输出请求
- 多模态：管理编码器输入和缓存

## 示例用法

在调度循环中，SchedulerOutput通常这样生成：

```python
# 调度请求后创建输出
output = SchedulerOutput(
    scheduled_new_reqs=new_requests,
    scheduled_cached_reqs=cached_requests,
    num_scheduled_tokens=token_counts,
    total_num_scheduled_tokens=total_tokens,
    scheduled_spec_decode_tokens=spec_tokens,
    scheduled_encoder_inputs=encoder_inputs,
    num_common_prefix_blocks=common_blocks,
    finished_req_ids=finished_ids,
    free_encoder_mm_hashes=free_hashes
)
```

## 设计特点

1. **数据类设计**：使用`@dataclass`简化了对象的创建和字段访问
2. **类型注解**：完整的类型提示增强了代码可读性和IDE支持
3. **模块化结构**：将不同类型的请求数据分离，便于维护和扩展
4. **内存效率**：通过缓存机制减少重复数据传输
5. **可扩展性**：预留了多种扩展点，支持未来新增功能

该类的设计体现了vLLM系统的模块化和高效率设计理念，能够有效管理大规模并发请求的调度和执行。