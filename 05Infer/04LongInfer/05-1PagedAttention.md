# Ref
1. [图解大模型计算加速系列之：vLLM核心技术PagedAttention原理](https://zhuanlan.zhihu.com/p/691038809)
2. [PagedAttention 论文](https://arxiv.org/pdf/2309.06180)

# 概述

本文主要用于科普和简述PagedAttention的原理，主要用于支持`./05FastGen`的背景技术中的PagedAttention部分

## 前言 - 为什么有PagedAttention

首先，LLM Infer主要有`prefill`和`decode`两个阶段：

![](./images/05-1LLM-Infer.png)


如图，不难观察到：

- Decode阶段的是逐一生成token的，因此它不能像prefill阶段那样能做大段prompt的并行计算
- 随着prompt数量变多和序列变长，KV cache也变大，对gpu显存造成压力
- 由于输出的序列长度无法预先知道，所以我们很难提前为KV cache量身定制存储空间

因此，如何优化KV cache，节省显存，提高推理吞吐量，就成了LLM推理框架需要解决的重点问题。

## 背景 - 之前如何做的

针对这个KVCache分配，之前的做法是很简单粗暴的直接按照(batch_size, max_seq_len)这样的固定尺寸。显然，会造成大量的内存浪费。



## PagedAttention

类比传统的分页操作系统，PagedAttention对于不同的Request正如对于不同进程一样分别分配进程块并且通过虚实转换机制映射到物理内存中实现分页管理。

![](./images/05-2PagedAtten_2reqs.png)

