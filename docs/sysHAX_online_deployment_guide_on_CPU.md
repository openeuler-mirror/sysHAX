### 版本信息  
更新日期：2025年11月11日

***

# 1 产品概述

## 1.1 sysHAX整体介绍

**sysHAX** 是一款面向CPU + xPU（GPU/NPU/...）异构计算架构的推理加速系统，旨在通过智能任务调度与资源优化，充分发挥不同硬件平台（CPU与xPU）的计算优势，实现大语言模型（LLM）推理性能的最大化。其核心功能定位为 **“异构融合推理加速”**，主要包含以下两大能力：

* **推理动态调度**

* **CPU 推理性能加速**

在纯CPU场景下，不涉及推理动态调度，只用到CPU推理性能加速。

该系统特别针对 LLM 推理过程中不同阶段的计算特性进行优化，提升整体吞吐率与资源利用率。sysHAX目前的应用场景是单机多卡（CPU+xPU），未来计划支持多机多卡场景。

***

# 2 组件介绍

|**组件**|**运行的设备**|介绍|
|---|---|---|
|vllm-cpu|CPU|处理推理请求|

***

# 3 软件环境

|**类型**|**版本要求**|**说明**|
|---|---|---|
|操作系统|openEuler 22.03 LTS 、openEuler 24.03 LTS|-|
|python|3.11及以上|部署vllm服务需要python|
|docker|25及以上|vllm部署，实现开箱即用|
|模型|qwen2、qwen3系列|sysHAX当前支持qwen2、qwen3系列dense模型|

***

# 4 硬件规格

|**类型**|**型号**|**说明**|
|---|---|---|
|服务器|鲲鹏920系列服务器，推荐920 7280Z以上系列|推理加速的功能是参考920服务器的特性实现。|

***

# 5 快速开始

先把模型存放在宿主机的`/home/models`路径下。然后创建容器时，再将`/home/models`挂载到容器内的相同路径，就可以在容器内访问模型了。推理前，需要先搭建vllm-cpu容器、sysHAX服务。

## 5.1 搭建vllm-cpu容器

### 5.1.1 创建vllm-cpu容器

```shell
# 从远端仓库拉取镜像，镜像中配置了vllm-cpu的相关环境
docker pull hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1
# 创建名为vllm_cpu的容器。创建完容器自动进入容器的工作目录
docker run --name vllm_cpu \
    --shm-size=64g \
    --privileged \
    -p 8001:8001 \
    -v /home/models:/home/models \
    -w /home/ \
    -it hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1 bash

```

docker run命令的参数解释：

|**参数**|**解释**|
|---|---|
| `name` |指定容器名。|
| `shm-size` |设置容器的共享内存大小，单位为GB。|
| `privileged` |授予容器特权模式，使其拥有对主机设备的广泛访问权限。|
| `p` |port的缩写。将容器内端口映射到宿主机端口。|
| `v` |将宿主机的目录挂载到容器的路径。|
| `w` |设置容器的工作目录。|
| `it` | -i：保持 STDIN 打开。-t： 分配伪终端。|
|`hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1`|指定使用的容器镜像名称及版本。|

### 5.1.2 部署

```shell
# 部署vllm(CPU)服务
INFERENCE_OP_MODE=fused \
OMP_NUM_THREADS=160 \
CUSTOM_CPU_AFFINITY=0-159 \
SYSHAX_QUANTIZE=q4_0 \
NRC=4 \
vllm serve /home/models/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype=half \
    --block_size=16 \
    --preemption_mode=swap \
    --max_model_len=8192
```

|**参数**|**解释**|
|---|---|
| `INFERENCE_OP_MODE`|sysHAX自定义环境变量，表示是否启动CPU推理加速。是可选值：fused、None。|
| `OMP_NUM_THREADS` |sysHAX自定义环境变量，表示CPU推理加速时开启的线程数量。其值≤可用的CPU核的数量。用lscpu命令可查看服务器中的CPU核数。|
| `CUSTOM_CPU_AFFINITY` |sysHAX自定义环境变量，表示将线程绑定到哪些CPU核。注意这里的线程数量要跟`OMP_NUM_THREADS`一致，并且每个numa的线程数量要相等。格式为“start-end:step”，step默认为1。例如，0-159:2中，0、159、2分别代表开始的线程序号、结束的线程序号（包含）、步长。|
| `SYSHAX_QUANTIZE` |sysHAX自定义环境变量，表示采用的量化方式。不设置时默认不量化，设置时可选值为`q8_0`、`q4_0`，表示进行`q8_0`量化、`q4_0`量化。|
| `NRC` |sysHAX自定义环境变量，表示i8mm指令处理的矩阵分块的大小。可选值为2、4。|
| `/home/models/DeepSeek-R1-Distill-Qwen-32B` |模型存储路径。|
| `host` |指定服务器监听的网络接口。|
| `port` |指定服务器监听的网络端口号。|
| `dtype` |指定模型权重加载的数据类型。可选值为`half`。|
| `block_size` |定义 PagedAttention 中KV Cache块的大小，表示一个KV Cache块中包含的token的个数。可选值为`8`、`16`。|
| `preemption_mode` | 当新请求到达而资源不足时，vLLM 支持通过“抢占”旧请求的方式释放资源。该参数控制如何处理被抢占的请求。可选值为 `recompute`、`swap`、`none`。|
| `max_model_len` |设置模型支持的最大上下文长度（提问+回答的总token长度）。|

### 5.1.3 部署样例

下面使用一个920 7280Z的服务器为例，展示如何设置vllm serve命令的参数。

![lscp输出示例](pictures/lscpu.png "")

解释lscpu输出中的重点数值：

|**参数**|**解释**|
|---|---|
| `On-line CPU(s) list` | CPU 列表。CPU个数可通过`Socket(s) × Core(s) per socket × Thread(s) per core`计算|
| `Thread(s) per core` | 每个核心1个线程 |
| `Core(s) per socket` | 每个CPU插槽的物理核心个数 |
| `Socket(s)` | CPU插槽个数 |
| `Flags` | `Flags` 列出了CPU架构支持的特性。在本版本中`asimd`、`sve`、`svei8mm`、`i8mm`等特性对推理性能有比较大的影响。 |
| `NUMA` | NUMA节点相关信息。 |

该服务器有4个numa，每个numa拥有40个计算核心。根据不同的使用场景，可以如下设置：

|**场景**|**参考部署服务命令**|
|---|---|
|只运行sysHAX跟vllm服务。并且需要CPU高速推理。| `INFERENCE_OP_MODE=fused`<br> `OMP_NUM_THREADS=160`<br> `CUSTOM_CPU_AFFINITY=0-159`<br> `SYSHAX_QUANTIZE=q4_0`<br> `NRC=4` |
|除了运行sysHAX跟vllm服务外，还需留部分算力给其他进程。|`INFERENCE_OP_MODE=fused`<br> `OMP_NUM_THREADS=120`<br> `CUSTOM_CPU_AFFINITY=0-29,40-69,80-109,120-149`<br> `SYSHAX_QUANTIZE=q4_0`<br> `NRC=4` |

## 5.2 发起推理请求

至此，已经部署完成vllm-cpu，现在来发起推理请求。

### 5.2.1 curl请求

```shell
curl http://0.0.0.0:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
    "messages": [
        {
            "role": "user",
            "content": "介绍一下openEuler。"
        }
    ],
    "stream": true,
    "max_tokens": 1024
}'
```

|**参数**|**解释**|
|---|---|
| `stream` | 启用流式传输，结果将以数据流形式逐个token返回。stream设置为`false`可一次性返回所有生成的内容。 |
| `max_tokens` | 限制响应生成的最大token数 |

***

# 6. 附录
