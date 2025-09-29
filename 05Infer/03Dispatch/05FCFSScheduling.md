<!--Copyright © ZOMI 适用于 [License](https://github.com/Infrasys-AI/AIInfra) 版权许可 -->

# 07. LLM 推理请求调度：FCFS 调度

> 作者：Yuhao Chen

大模型推理的请求和传统的 Web/APP 请求或数据库查询完全不同：在传统的应用中，请求一般短、可预测、一次性完成。但是在大语言模型（LLM）的推理过程中，一个请求最终会生成多少个 token 通常是事先无法预知的。比如说，用户的提示词可能是“告诉我今天的天气”，也可能是“写一篇 200000 字的文章”。但推理系统在真正运行之前，并不知道解码会持续多少步。并且LLM 推理分为两个不同的阶段：Prefill（预填充）是把输入 prompt 全部送进模型，计算量大但只需要执行一次； 而Decode（解码）阶段会逐 token 生成输出，每次循环都要经过完整的前向传播，属于高频、低延迟的阶段。不同请求的 prefill / decode 长度差异很大，如果简单的混在一起处理，就会让 GPU 资源浪费或延迟飙升。因为这种输出的不确定性，许多 LLM 服务系统在调度请求时，都会选择一种最简单又直观的方式：先来先服务（FCFS, First-Come-First-Serve）。换句话说，调度器不会去预测哪个请求更短、更快完成，而是直接按照到达顺序排队。

## 深入理解 FCFS 调度

### FCFS的定义与直观理解

- **定义**：按请求的**到达时间**排序，先到的请求优先被调度。顾名思义，先来后到，每次从就绪队列选择最先进入队列的进程，然后一直运行，直到进程退出或被阻塞，才会继续从队列中选择第一个进程接着运行。这似乎很公平，但是当一个长作业先运行了，那么后面的短作业等待的时间就会很长，不利于短作业。
- **直观理解**：将 API 服务视为超市收银台——谁先排队，谁先结账。其特点是**公平、可解释、易于观测**。

![FCFS直觉图](./images/05FCFSScheduling01.png)
[http://interview.wzcu.com/System/%E8%B0%83%E5%BA%A6%E7%AE%97%E6%B3%95.html]

### FCFS 调度在大模型推理中的位置

1. **请求到达**

   * 用户通过 API/SDK 发起请求（包含输入文本、参数等）。
   * 系统将请求放入 **等待队列**。

2. **调度（Scheduling）**

   * 调度器决定哪些请求进入当前批次（batch）。
   * 这里涉及 **排队策略**（FCFS 就在这一环节）。

3. **Prefill 阶段**

   * 一次性处理整个输入上下文，算 Attention、KV cache 初始化。
   * 计算量大，算力密集。

4. **Decode 阶段**

   * 逐步生成 token，每次只前向推理一步。
   * 访存密集，依赖频繁的 KV cache 读写。

5. **结果返回用户**

   * token 流式输出或一次性返回。

---

**FCFS 位于第 2 步「调度」阶段**，作用是：

* 当多个请求堆在队列里时，**谁先到达，就先被调度进批次**。
* 在 **连续批处理（Continuous Batching）** 的迭代循环中，调度器会优先续跑已有的解码任务，然后再按照 **FCFS 顺序**补入新的预填充任务。

### 与连续批处理(continuous batching)的契合

现代推理服务采用 **迭代级调度（iteration-level scheduling）**：每个调度周期，调度器会组装一个**混合批次**（包含正在解码的序列 + 新到达或分块的预填充请求），并提交执行。在此机制下，FCFS 扮演的是**补位规则**：

1. **优先续跑解码任务**（保障稳定的产出速率 / TPOT）；
2. **用剩余资源按队列顺序注入预填充任务**（降低 TTFT）；
3. **下一轮重新组合批次**——这正是 Continuous Batching 的核心思想。

### FCFS的优缺点

#### 优点

1. **简单易实现**：算法复杂度极低，工程上几乎“零成本”落地，也不需要额外的请求特征估计（如上下文长度、剩余解码步数）。

2. **公平性好**：严格按照到达时间排队，保证了用户层面直观的“谁先来谁先算”，不会出现长请求starvation的情况。

3. **可解释性强**：排队结果对用户、监控系统都直观透明，方便做 SLA 指标（P95、P99 延迟）的监控与解释。

---

#### 缺点
1. **无法感知任务长度**

* FCFS 完全不考虑请求的 **上下文长度** 或 **目标输出长度**。如果队列前面有一个超长请求，它的 Prefill 阶段可能要消耗大量时间和显存。后面的一批短请求只能干等着，即使它们几乎可以“秒出结果”。

2. **吞吐与延迟的平衡受限**

* FCFS 追求的是“公平”，但并不保证 **端到端延迟的最优**。在负载不均衡的情况下，整体吞吐率可能没问题，但尾部延迟（P95/P99）会飙升。

3. **资源利用率不稳定**

* Prefill 和 Decode 的负载模式差异很大（算力 vs 内存）。如果 FCFS 把几个“大 Prefill”任务排在一起，就可能导致 GPU 在 Decode 端产生资源浪费。

4. **对服务策略不敏感**

* 在一些场景，用户请求的重要性并不一样：比如「付费用户」 vs 「免费用户」，「实时对话」 vs 「批量处理」。但 FCFS 不区分优先级，所有请求一视同仁。这会让系统缺乏灵活性，无法满足不同业务线的服务等级需求。

5. **极端情况下延迟爆炸**

* 当负载中存在极长的上下文输入时（例如几万 token），队列后面的短请求都会被“绑架”导致无法执行。这种情况被称为 **Head-of-line blocking（队首阻塞）**，在 LLM 推理里尤其常见，因为输入长度分布差异极大。

![Head-of-line blocking](./images/05FCFSScheduling02.png)

###
---
!!!!!!
FCFS有局限性，所以可以引出 HoL blocking调度。
调度策略的整体分类和演进，简单介绍其他调度策略。

## vLLM 中的 FCFS 调度策略

### 默认调度策略

vLLM对请求的调度处理流程：

    当一堆请求来到vLLM服务器上时，按照First-Come-First-Serve（FCFS）原则，优先处理那些最早到来的请求。
    当gpu资源不足时，为了让先来的请求能尽快做完推理，vLLM会对那些后到来的请求执行“抢占”，即暂时终止它们的执行。
    一旦vLLM决定执行抢占操作，它会暂停处理新到来的请求。在此期间，它会将被抢占的请求相关的KV block全部交换（swap）至cpu上。等交换完成后，vLLM才会继续处理新到来的请求。
    当vLLM认为gpu有足够资源时，它会将cpu上的KV block重新加载回gpu，恢复被抢占请求的执行（recomputation）


### vLLM 默认 FCFS 调度的具体实现

#### 基于 Token 预算的分片调度引擎

* 系统在每个调度周期（Step）里，会设置一个 **最大 Token 预算（token_budget）**，类似“这一轮 GPU 能算多少 token”。
* 调度器遍历请求队列（等待队列 + 运行队列），按照 **FCFS 顺序**（先来先服务），为请求分配 token。
* 如果请求太大、超出预算，就会分片执行，把一部分内容推到下一个 Step 再继续。

#### Step 初始化

* 计算出本轮 token_budget。
* 初始化要调度的请求集合（running/new/resumed）。

#### 遍历运行队列

1. 计算该请求需要多少新 token（num_new_tokens），受 **预算限制**、**长 Prefill 限制**、**模型最大长度限制** 等条件约束。
2. 如果超出预算 → 截断分片，只调度部分请求。
3. 如果 cache 空间不足，会触发 **FCFS抢占（preemption）**：从 **运行队列**的队尾抢占掉一个请求。被抢占请求会被清掉 KV cache，回到等待队列，状态改成 PREEMPTED。

#### 遍历等待队列

* 按照fcfs调度策略依次从等待队列的队头取出请求：
  * 检查状态是否满足执行条件（等待 KV、结构化输出 FSM、LoRA 等都可能跳过）。
  * 分配 KV cache 块，如果没空间则停止调度。
* 调度成功的请求转为RUNNING状态，并记录元数据（req_id ↔ kv_cache_block，scheduled_tokens 等）。

#### 更新状态

* 所有成功调度的请求会进入本轮的 **SchedulerOutput**，交给 Worker 执行。
* Step 结束后，每个请求会输出一个新 token（除非遇到停止条件），等待下轮继续。

#### 新请求的接入

* 外部调用 `scheduler.add_request`，新的请求进入等待队列。
* 上层引擎（EngineCoreProc）会调用 `schedule()`从 WAITING+RUNNING 中选出一批请求。
* 执行后通过 `update_from_output` 更新状态。

这样，新请求和旧请求都能被动态插入下一轮调度。

默认的 **FCFS 调度**在 vLLM 里并不是“纯排队”，而是：

* **基于 token 预算的动态分片**，每一轮 Step 都重新分配预算。
* **优先续跑 Decode（RUNNING 队列）**，保证输出平滑。
* **再按 FCFS 顺序补 Prefill（WAITING 队列）**，短请求和长请求交错执行。
* 如果资源不足，会触发 **抢占**，把低优先级或队尾请求放回等待队列。

这样，它既保持了 FCFS 的公平性，又通过分片和优先 Decode 避免了 Prefill“卡死”整个系统。


### Vllm 中 FCFS 策略的工程化增强

1. **分块预填充（Chunked Prefill）**

   - 将长 prompt 切分为 256/512 token 的小块，分多轮注入，**显著缓解队头阻塞**。
   - 通过 `max_num_partial_prefills` 控制并发分块数，避免过度打断解码流程。

2. **长短请求分级 + “短块优先”窗口**

   - 使用 `long_prefill_token_threshold` 划分长短请求；
   - 设置 `max_long_partial_prefills < max_num_partial_prefills`，使短块在分块窗口内**优先调度**，有效降低 P95/P99 延迟。

3. **预算与座位的“硬性护栏”**

   - 合理设置 `max_num_batched_tokens` 和 `max_num_seqs`，防止单次预填充耗尽全部资源；
   - 通过 KV 缓存高/低水位控制，**优先保障解码任务的稳定执行**。

4. **轻量级优先级（温和改造）**

   - 在 FCFS 基础上引入少量 `priority` 字段（如“交互流量 > 批量任务”），但**限制抢占频率**，避免产出抖动。

5. **KV 缓存管理与容量优化**

   - 采用混合/分页/压缩 KV 缓存技术，扩大并发容量，减少因缓存不足导致的请求排队膨胀。

## 6. 何时切换策略：从 FCFS 到工作负载感知调度

若满足以下任一条件，可有考虑升级调度策略：

- **混合负载显著**：交互式请求与长文本生成/离线批处理共存，**尾部延迟上升、用户抱怨增多**。
- **多租户需分级 SLO**：存在明确的金银铜用户或业务优先级，需要**差异化服务质量保障**。
- **追求极致成本/吞吐**：需在高并发下进一步压缩资源空泡，提升 TPS。

**可选演进方向（循序渐进）**：

- **带权 FCFS / 优先级调度**：在保持到达公平的基础上引入**有限分层**。
- **近似最短作业优先（SJF）或学习排序（LTR）**：基于长度或剩余步数预测，实现“短任务优先”。
- **基于预估执行时间的可抢占调度**：在安全同步点进行**低成本抢占**，避免频繁上下文切换。

### 可复用的 vLLM 配置模板

#### 低 TTFT（交互优先）

```yaml
policy: fcfs
enable_chunked_prefill: true
max_num_partial_prefills: 2
max_long_partial_prefills: 1
long_prefill_token_threshold: 0.04 * max_model_len
max_num_batched_tokens: max(DEFAULT_MAX_NUM_BATCHED_TOKENS, 2 * max_model_len)
max_num_seqs: 128
```

#### 高吞吐（离线/批量任务）

```yaml
policy: fcfs
enable_chunked_prefill: true
max_num_partial_prefills: 1
max_long_partial_prefills: 1
max_num_batched_tokens: 4 * max_model_len
max_num_seqs: 128
cuda_graph_sizes: [1,2,4,8,16,32,64,128]   # 可选
```

#### 通用抗阻塞（长上下文混合场景）

```yaml
policy: fcfs
enable_chunked_prefill: true
max_num_partial_prefills: 2
max_long_partial_prefills: 1
long_prefill_token_threshold: 0.06 * max_model_len
max_num_batched_tokens: max(2 * max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
max_num_seqs: 128
disable_hybrid_kv_cache_manager: false
```



## 8. 与其他调度策略的对比

| 维度             | FCFS                     | 带权/优先级         | 近似 SJF / LTR       | 可抢占（估时/细粒度）     |
|------------------|--------------------------|--------------------|----------------------|--------------------------|
| 到达公平性       | ★★★★★                    | ★★★★☆              | ★★☆☆☆                | 取决于具体实现           |
| TTFT（交互体验） | ★★★★☆（配合分块）        | ★★★★☆              | ★★★★★                | ★★★★☆                    |
| 稳定产出（TPOT） | ★★★★☆（解码优先）        | ★★★☆☆              | ★★★☆☆                | ★★★☆☆                    |
| 实现复杂度       | ★★☆☆☆                    | ★★★☆☆              | ★★★★☆                | ★★★★★                    |
| 适用场景         | 单业务/中等并发           | 多租户分层          | 混合负载显著          | 极致 SLO / 成本优化      |

## 9. 总结与思考

> **FCFS 是默认策略，不代表着保守和低性能**。在推理框架运用 **连续批处理** 时代，通过 **解码优先 + 适度分块预填充 + 预算与缓存水位控制** 的工程基线，FCFS 能同时兼顾**公平性、稳定性与效率**。但是当当混合负载和分级 SLO 的需求日益突出时，也可以考虑将FCFS**渐进式升级**到工作负载感知的调度策略以达到性能的最大优化。

## 参考与引用


!!!!!!!哪些地方引用了这些文章了？

- [1] vLLM Community, “Is FCFS scheduling holding back vLLM’s performance in production?”, vLLM Forum, 2024. [Online]. Available: [https://discuss.vllm.ai/t/is-fcfs-scheduling-holding-back-vllms-performance-in-production/1584](https://discuss.vllm.ai/t/is-fcfs-scheduling-holding-back-vllms-performance-in-production/1584)
- [2] Y. Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models,” OSDI, 2022. [Online]. Available: [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- [3] W. Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” 2024. [Online]. Available: [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)
- [4] Hugging Face, “LLM performance: prefill vs decode under concurrent requests,” Blog, 2024. [Online]. Available: [https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)
- [5] Hugging Face, “LLM performance: request queueing strategies,” Blog, 2024. [Online]. Available: [https://huggingface.co/blog/tngtech/llm-performance-request-queueing](https://huggingface.co/blog/tngtech/llm-performance-request-queueing)
- [6] Databricks, “LLM Inference Performance Engineering Best Practices,” Blog, 2024. [Online]. Available: [https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [7] vLLM, “SchedulerConfig & Continuous Batching (source code, version-dependent),” 2024. [Online]. Available: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (see `vllm/core/scheduler`)
```