# vLLM V1 总体架构与关键模块详解

## 一、总体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           vLLM V1 多进程架构                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   API Server     │     │   API Server     │     │   API Server     │  (数量 = DP)
│   (进程 P0)      │     │   (进程 P1)      │     │   (进程 P2)      │
│                  │     │                  │     │                  │
│ - HTTP 请求处理   │     │                  │     │                  │
│ - Tokenization   │     │                  │     │                  │
│ - 输入预处理      │     │                  │     │                  │
│ - 输出后处理      │     │                  │     │                  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         │ ZMQ Socket             │                        │
         │ (多对多拓扑)            │                        │
         ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Engine Core (进程)                                     │
│                          数量 = Data Parallel Size                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Scheduler                                        │   │
│  │                                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ │   │
│  │  │ Waiting Queue│  │Running Queue │  │   KV Cache Manager          │ │   │
│  │  │              │  │              │  │                              │ │   │
│  │  │ - 新请求排队  │  │ - 正在执行   │  │ - PagedAttention 块管理     │ │   │
│  │  │ - 优先级排序  │  │ - 状态追踪   │  │ - 前缀缓存                   │ │   │
│  │  └──────────────┘  └──────────────┘  │ - 块分配/释放               │ │   │
│  │                                      └──────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  核心 Busy Loop:                                                                │
│  1. schedule() → SchedulerOutput                                               │
│  2. execute_model(scheduler_output) → ModelRunnerOutput                        │
│  3. update_from_output() → EngineCoreOutputs                                   │
│                                                                                 │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                                │ MessageQueue (共享内存)
                                │ SchedulerOutput 广播
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GPU Worker (进程)                                       │
│                         数量 = TP × PP × DP                                      │
│                         每个 GPU 一个 Worker 进程                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     ModelRunner                                          │   │
│  │                                                                         │   │
│  │  - 准备输入张量 (attention metadata, positions)                          │   │
│  │  - 管理 CUDA Graphs                                                      │   │
│  │  - 执行 forward pass                                                     │   │
│  │  - Sample tokens (采样)                                                  │   │
│  └────────────────────────────────────┬────────────────────────────────────┘   │
│                                       │                                        │
│                                       ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      Model (torch.nn.Module)                             │   │
│  │                                                                         │   │
│  │  例如: LlamaForCausalLM                                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ embed_tokens│  │ Layer 0-N   │  │   norm      │  │   lm_head   │    │   │
│  │  │             │  │             │  │             │  │             │    │   │
│  │  │ 词嵌入层     │  │ DecoderLayer│  │ RMSNorm     │  │ 输出层      │    │   │
│  │  └─────────────┘  │ ┌─────────┐ │  └─────────────┘  └─────────────┘    │   │
│  │                   │ │Self Attn│ │                                        │   │
│  │                   │ │         │ │                                        │   │
│  │                   │ │ QKV proj│ │                                        │   │
│  │                   │ │ Rotary  │ │                                        │   │
│  │                   │ │ Attention│ │                                       │   │
│  │                   │ └─────────┘ │                                        │   │
│  │                   │ ┌─────────┐ │                                        │   │
│  │                   │ │  MLP    │ │                                        │   │
│  │                   │ │gate_up  │ │                                        │   │
│  │                   │ │down_proj│ │                                        │   │
│  │                   │ └─────────┘ │                                        │   │
│  │                   └─────────────┘                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

        当 DP > 1 时存在:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       DP Coordinator (进程)                                     │
│                       数量 = 1 (仅当 DP > 1)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  - 跨 DP ranks 负载均衡                                                         │
│  - MoE 模型同步前向传播                                                          │
│  - Wave 请求批次协调                                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 二、数据流向图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           请求处理数据流                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

用户请求 (HTTP)
      │
      ▼
┌──────────────────┐
│   API Server     │
│                  │
│ 1. InputProcessor│  ← process_inputs(): EngineInput → EngineCoreRequest
│    - Tokenize    │
│    - 多模态处理   │
│                  │
│ 2. OutputProcessor│ ← process_outputs(): EngineCoreOutput → RequestOutput
│    - Detokenize  │
│    - 流式输出     │
│                  │
└───────┬──────────┘
        │ add_request_async()
        │ (ZMQ DEALER socket)
        ▼
┌──────────────────┐
│   Engine Core    │
│                  │
│ Scheduler Loop:  │
│                  │
│ ┌────────────────────────────────────────────────────────────────┐            │
│ │                    schedule() 方法                              │            │
│ │                                                                │            │
│ │  for each request in running:                                  │            │
│ │      allocate tokens → update num_scheduled_tokens             │            │
│ │                                                                │            │
│ │  for each request in waiting:                                  │            │
│ │      check constraints (KV blocks, token budget)               │            │
│ │      allocate KV cache blocks                                   │            │
│ │      move to running                                            │            │
│ │                                                                │            │
│ │  return SchedulerOutput(                                       │            │
│ │      num_scheduled_tokens,                                     │            │
│ │      new_blocks,                                               │            │
│ │      cached_blocks,                                            │            │
│ │      ...)                                                      │            │
│ └────────────────────────────────────────────────────────────────┘            │
│                  │
│                  │ execute_model(scheduler_output)
│                  │ (MessageQueue 广播)
│                  ▼
┌──────────────────┐
│   GPU Worker     │
│                  │
│ ModelRunner:     │
│                  │
│ ┌────────────────────────────────────────────────────────────────┐            │
│ │                execute_model() 方法                             │            │
│ │                                                                │            │
│ │ 1. prepare_inputs(scheduler_output):                           │            │
│ │    - 构建 attention metadata                                   │            │
│ │    - 组织 input_ids, positions                                 │            │
│ │                                                                │            │
│ │ 2. forward_pass():                                             │            │
│ │    model(input_ids, positions, intermediate_tensors)           │            │
│ │                                                                │            │
│ │ 3. sample_tokens() (仅 PP rank=0):                             │            │
│ │    - logits → sampling → token_ids                             │            │
│ │                                                                │            │
│ │ return ModelRunnerOutput                                       │            │
│ └────────────────────────────────────────────────────────────────┘            │
│                  │
│                  │ return ModelRunnerOutput
│                  ▼
┌──────────────────┐
│   Engine Core    │
│                  │
│ update_from_output():                                            │
│   - 更新请求状态 (num_computed_tokens)                          │
│   - 更新 KV cache 状态                                           │
│   - 生成 EngineCoreOutputs                                       │
│                  │
│                  │ get_output_async()
│                  │ (ZMQ PUSH socket)
│                  ▼
┌──────────────────┐
│   API Server     │
│                  │
│ output_handler():│  ← asyncio background loop
│   - 接收 outputs │
│   - process_outputs()                                           │
│   - 推送到 AsyncStream                                          │
│                  │
│                  │ yield RequestOutput
│                  ▼
        用户 (流式响应)
```

## 三、关键模块详解

### 3.1 AsyncLLM (异步引擎入口)

**文件**: `vllm/v1/engine/async_llm.py`

**核心职责**: 
- 异步请求处理接口
- 管理 EngineCore 后台进程
- 处理输入/输出的异步协调

**关键代码解析**:

```python
# AsyncLLM 初始化 (行 70-153)
class AsyncLLM(EngineClient):
    def __init__(self, vllm_config, executor_class, ...):
        # 1. 输入处理器: EngineInput → EngineCoreRequest
        self.input_processor = InputProcessor(vllm_config, renderer)
        
        # 2. 输出处理器: EngineCoreOutput → RequestOutput  
        self.output_processor = OutputProcessor(renderer.tokenizer, ...)
        
        # 3. EngineCore 客户端 (后台进程通信)
        self.engine_core = EngineCoreClient.make_async_mp_client(vllm_config, ...)
```

```python
# 核心生成方法 generate() (行 521-630)
async def generate(self, prompt, sampling_params, request_id, ...):
    # 步骤1: 添加请求到调度器
    q = await self.add_request(request_id, prompt, sampling_params, ...)
    
    # 步骤2: 从输出队列拉取结果 (异步循环)
    while not finished:
        out = q.get_nowait() or await q.get()  # 非阻塞优先，避免任务切换开销
        finished = out.finished
        yield out  # 流式返回给调用者
```

```python
# 输出处理后台任务 (行 632-702)
def _run_output_handler(self):
    async def output_handler():
        while True:
            # 1. 从 EngineCore 拉取输出
            outputs = await engine_core.get_output_async()
            
            # 2. 处理输出
            processed_outputs = output_processor.process_outputs(outputs)
            
            # 3. 推送到各自的请求队列
            # RequestOutputCollector 负责分发
```

---

### 3.2 EngineCore (引擎核心)

**文件**: `vllm/v1/engine/core.py`

**核心职责**:
- 调度循环 (Schedule → Execute → Update)
- KV Cache 管理
- 执行模型

**关键代码解析**:

```python
# EngineCore 初始化 (行 91-230)
class EngineCore:
    def __init__(self, vllm_config, executor_class, ...):
        # 1. 创建执行器 (管理 GPU Workers)
        self.model_executor = executor_class(vllm_config)
        
        # 2. 初始化 KV Caches (内存分析后分配)
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        
        # 3. 创建调度器
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        self.scheduler = Scheduler(vllm_config, kv_cache_config, ...)
        
        # 4. 批处理队列 (用于 Pipeline Parallelism)
        if self.batch_queue_size > 1:
            self.batch_queue = deque(maxlen=self.batch_queue_size)
```

```python
# 核心调度步骤 step() (行 402-431)
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """调度、执行、输出"""
    
    # 1. 检查是否有请求
    if not self.scheduler.has_requests():
        return {}, False
    
    # 2. 调度: 决定哪些请求获得 token budget
    scheduler_output = self.scheduler.schedule()
    
    # 3. 执行模型 (异步 Future)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    
    # 4. 获取 grammar bitmask (用于结构化输出)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    
    # 5. 等待模型输出
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
    
    # 6. 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

```python
# EngineCoreProc Busy Loop (行 1164-1172)
def run_busy_loop(self):
    """核心忙循环"""
    while self._handle_shutdown():
        # 1. 处理输入队列 (等待工作)
        self._process_input_queue()
        
        # 2. 执行引擎步骤
        self._process_engine_step()
```

---

### 3.3 Scheduler (调度器)

**文件**: `vllm/v1/core/sched/scheduler.py`

**核心职责**:
- 管理 Waiting/Running 请求队列
- 分配 Token Budget
- 管理 KV Cache Blocks

**调度算法注释** (行 351-361):
```python
# NOTE(woosuk) on the scheduling algorithm:
# 没有 "解码阶段" 或 "预填充阶段" 的概念。
# 每个请求只有 num_computed_tokens 和 num_tokens_with_spec。
# 
# num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
# 
# 每一步，调度器尝试为请求分配 tokens，
# 使其 num_computed_tokens 能追赶 num_tokens_with_spec。
# 这足够通用，覆盖:
# - chunked prefills (分块预填充)
# - prefix caching (前缀缓存)
# - speculative decoding (推测解码)
# - "jump decoding" 优化
```

**schedule() 方法核心逻辑** (行 351-399):
```python
def schedule(self) -> SchedulerOutput:
    scheduled_new_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    
    # 1. 首先调度 RUNNING 请求 (解码阶段)
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        # 分配 token budget 给正在运行的请求
        ...
    
    # 2. 然后调度 WAITING 请求 (预填充阶段)
    while len(self.waiting) > 0 and token_budget > 0:
        request = self.waiting[0]
        # 检查约束: KV blocks, token budget, max_model_len
        # 分配 KV cache blocks
        # 移动到 running 队列
        ...
    
    return SchedulerOutput(
        num_scheduled_tokens=num_scheduled_tokens,
        new_blocks=req_to_new_blocks,
        cached_blocks=cached_blocks,
        ...
    )
```

---

### 3.4 MultiprocExecutor (多进程执行器)

**文件**: `vllm/v1/executor/multiproc_executor.py`

**核心职责**:
- 创建和管理 GPU Worker 进程
- 通过 MessageQueue 广播 SchedulerOutput
- 收集 ModelRunnerOutput

**关键代码解析**:

```python
# MultiprocExecutor 初始化 (行 102-246)
class MultiprocExecutor(Executor):
    def _init_executor(self):
        # 1. 创建 MessageQueue (用于广播 SchedulerOutput)
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,        # 总 worker 数量
            self.local_world_size,  # 本节点 worker 数量
            max_chunk_bytes=...,
        )
        
        # 2. 创建 Worker 进程
        for local_rank in range(self.local_world_size):
            unready_worker_handle = WorkerProc.make_worker_process(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=global_rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,  # 共享内存句柄
                ...
            )
            unready_workers.append(unready_worker_handle)
        
        # 3. 等待所有 Worker 就绪
        self.workers = WorkerProc.wait_for_ready(unready_workers)
```

```python
# collective_rpc: 向所有 Worker 发送命令 (行 339-403)
def collective_rpc(self, method, args, kwargs, non_block=False, ...):
    # 1. 通过 MessageQueue 广播请求
    self.rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))
    
    # 2. 创建 Future 等待响应
    def get_response():
        responses = []
        for mq in response_mqs:
            status, result = mq.dequeue(timeout=timeout)
            responses.append(result)
        return responses
    
    future = FutureWrapper(self.futures_queue, get_response)
    return future if non_block else future.result()
```

```python
# WorkerProc Busy Loop (行 944-970)
def worker_busy_loop(self):
    """Worker 主循环"""
    while True:
        # 1. 从 MessageQueue 接收命令
        method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
        
        # 2. 执行命令
        if isinstance(method, str):
            func = getattr(self.worker, method)  # 如 execute_model
        elif isinstance(method, bytes):
            func = cloudpickle.loads(method)     # 序列化的函数
        
        output = func(*args, **kwargs)
        
        # 3. 返回结果
        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)  # 放入 response_mq
```

---

### 3.5 GPU Worker

**文件**: `vllm/v1/worker/gpu_worker.py`

**核心职责**:
- GPU 设备初始化
- 模型加载
- 执行 forward pass

**关键代码解析**:

```python
# Worker 初始化 (行 105-120)
class Worker(WorkerBase):
    def __init__(self, vllm_config, local_rank, rank, ...):
        super().__init__(...)
        
        # 配置 float32 matmul 精度
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)
        
        # 权重传输引擎 (可选)
        self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(...)
```

```python
# init_device(): 设备初始化 (行 219-315)
def init_device(self):
    # 1. 设置 GPU 设备
    self.device = torch.device(f"cuda:{self.local_rank}")
    torch.cuda.set_device(self.device)
    
    # 2. 初始化分布式环境
    init_worker_distributed_environment(
        self.vllm_config, self.rank, ...
    )
    
    # 3. 设置随机种子
    set_random_seed(self.model_config.seed)
    
    # 4. 创建 ModelRunner
    self.model_runner = GPUModelRunner(self.vllm_config, self.device)
    
    # 5. 内存快照 (用于 KV cache 分配)
    self.init_snapshot = MemorySnapshot(device=self.device)
```

```python
# execute_model(): 执行模型 (行 753-841)
def execute_model(self, scheduler_output):
    # 1. 处理 Pipeline Parallel 的中间张量
    if forward_pass and not get_pp_group().is_first_rank:
        tensor_dict, comm_handles = get_pp_group().irecv_tensor_dict(...)
        intermediate_tensors = AsyncIntermediateTensors(tensor_dict, comm_handles)
    
    # 2. 调用 ModelRunner 执行
    with self.annotate_profile(scheduler_output):
        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors
        )
    
    # 3. Pipeline Parallel: 发送中间张量到下一阶段
    if not get_pp_group().is_last_rank:
        self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
        return None
    
    return output
```

---

### 3.6 LlamaModel (模型示例)

**文件**: `vllm/model_executor/models/llama.py`

**核心组件**: LlamaDecoderLayer 包含 Self-Attention 和 MLP

```python
# LlamaDecoderLayer 结构 (行 253-338)
class LlamaDecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, ...):
        # 1. Self-Attention 层
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        # 2. MLP 层
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            ...
        )
        
        # 3. Layer Norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, positions, hidden_states, residual):
        # Attention branch
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        
        # MLP branch
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
```

```python
# LlamaAttention 结构 (行 124-234)
class LlamaAttention(nn.Module):
    def __init__(self, config, hidden_size, num_heads, ...):
        # 1. QKV 投影 (合并，支持 Tensor Parallel)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            ...
        )
        
        # 2. Output 投影
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            ...
        )
        
        # 3. Rotary Embedding
        self.rotary_emb = get_rope(self.head_dim, ...)
        
        # 4. Attention 核心 (使用优化的 kernel)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            ...
        )
    
    def forward(self, positions, hidden_states):
        # 1. QKV 投影
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 2. Rotary Position Embedding
        q, k = self.rotary_emb(positions, q, k)
        
        # 3. Attention 计算 (PagedAttention kernel)
        attn_output = self.attn(q, k, v)
        
        # 4. Output 投影
        output, _ = self.o_proj(attn_output)
        return output
```

---

### 3.7 VllmConfig Pattern (设计模式)

**核心设计**: 所有类接受 `VllmConfig` 配置对象

```python
# 统一的构造器签名
class Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 从 vllm_config 提取所需配置
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.cache_config = vllm_config.cache_config
        ...
```

**设计优势**:
1. **扩展性**: 新增配置只需修改 VllmConfig，无需改每个类构造器
2. **统一性**: ModelRunner 可以统一初始化任何模型
3. **初始化时分片**: 支持在模型初始化时直接分片权重，避免加载全量权重到每个 GPU

---

## 四、并行策略

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          并行策略示意                                             │
└─────────────────────────────────────────────────────────────────────────────────┘

Tensor Parallelism (TP):
┌──────────────┐ ┌──────────────┐
│   GPU 0      │ │   GPU 1      │
│  (TP rank 0) │ │  (TP rank 1) │
│              │ │              │
│  QKV_proj    │ │  QKV_proj    │  ← 每个 GPU 持有部分 heads
│  (heads/2)   │ │  (heads/2)   │
│              │ │              │
│  o_proj      │ │  o_proj      │  ← RowParallelLinear: 结果 all-reduce
│  (部分权重)  │ │  (部分权重)   │
└──────────────┘ └──────────────┘

Pipeline Parallelism (PP):
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   GPU 0      │ │   GPU 1      │ │   GPU 2      │
│  (PP rank 0) │ │  (PP rank 1) │ │  (PP rank 2) │
│              │ │              │ │              │
│  Layers 0-8  │ │ Layers 9-17 │ │ Layers 18-26│  ← 每个阶段持有部分层
│              │ │              │ │              │
│  embed_tokens│ │              │ │    norm      │
│              │ │              │ │    lm_head   │
└──────────────┘ └──────────────┘ └──────────────┘
     │                │                │
     │  send tensor   │  send tensor   │
     │───────────────▶│───────────────▶│
     │  (intermediate)│  (intermediate)│

Data Parallelism (DP):
┌──────────────────┐  ┌──────────────────┐
│  Engine Core 0   │  │  Engine Core 1   │
│  (DP rank 0)     │  │  (DP rank 1)     │
│                  │  │                  │
│  完整模型副本     │  │  完整模型副本     │
│                  │  │                  │
│  处理请求组 A    │  │  处理请求组 B     │  ← 不同请求分配到不同 DP rank
└──────────────────┘  └──────────────────┘
         │                    │
         │     DP Coordinator │
         │    (负载均衡协调)   │
         └────────────────────┘

总 Worker 数量 = TP × PP × DP
例如: TP=2, PP=3, DP=4 → 24 个 Worker 进程
```

---

## 五、KV Cache 管理

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      KV Cache 块分配示意                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

请求 Request:
┌─────────────────────────────────────────────────────────────┐
│ prompt: "The capital of France is" + output: " Paris"       │
│ token_ids: [1, 2, 3, 4, 5, 6, 7, 8]                         │
└─────────────────────────────────────────────────────────────┘

KV Cache Blocks (block_size = 4 tokens):
┌──────────────┐ ┌──────────────┐
│  Block 0     │ │  Block 1     │
│              │ │              │
│ tokens 0-3   │ │ tokens 4-7   │
│ K0, V0      │ │ K4, V4      │
│ K1, V1      │ │ K5, V5      │
│ K2, V2      │ │ K6, V6      │
│ K3, V3      │ │ K7, V7      │
└──────────────┘ └──────────────┘

Prefix Caching:
┌─────────────────────────────────────────────────────────────┐
│ 如果多个请求有相同前缀 "The capital of France is"           │
│                                                             │
│ Request A: "The capital of France is Paris"                │
│ Request B: "The capital of France is Berlin"               │
│                                                             │
│ 共享 Block 0 (hash 计算)                                    │
│ Request A: Block 0 (共享) + Block 1_A                      │
│ Request B: Block 0 (共享) + Block 1_B                      │
└─────────────────────────────────────────────────────────────┘

KVCacheManager 核心方法:
- allocate_slots(): 为请求分配新块
- free(): 释放请求的块
- get_cached_blocks(): 查找前缀缓存
```

---

## 六、学习路径建议

作为初学者，建议按以下顺序深入理解:

1. **AsyncLLM** (`vllm/v1/engine/async_llm.py`): 理解请求如何进入系统
2. **EngineCore** (`vllm/v1/engine/core.py`): 理解调度循环如何工作
3. **Scheduler** (`vllm/v1/core/sched/scheduler.py`): 理解 token budget 分配
4. **MultiprocExecutor** (`vllm/v1/executor/multiproc_executor.py`): 理解进程间通信
5. **Worker** (`vllm/v1/worker/gpu_worker.py`): 理解 GPU 执行流程
6. **ModelRunner** (`vllm/v1/worker/gpu_model_runner.py`): 理解模型执行细节
7. **Model** (`vllm/model_executor/models/`): 理解模型结构与 Attention 实现

---

## 七、参考资料

- **架构文档**: `docs/design/arch_overview.md`
- **HuggingFace 集成**: `docs/design/huggingface_integration.md`
- **CUDA Graphs**: `docs/design/cuda_graphs.md`
- **PagedAttention**: `docs/design/paged_attention.md`
- **模型贡献指南**: `docs/contributing/model/`