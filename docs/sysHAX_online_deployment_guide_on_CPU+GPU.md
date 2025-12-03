### 版本信息

当前版本：v0.2.1  
更新日期：2025年11月11日

***

# 1. 产品概述

## 1.1. sysHAX整体介绍

![GPU执行Prefill+CPU执行Decode示意图](pictures/GPU_CPU_arch.png "")

**sysHAX** 是一款面向CPU + xPU（GPU/NPU/...）异构计算架构的推理加速系统，旨在通过智能任务调度与资源优化，充分发挥不同硬件平台（CPU与xPU）的计算优势，实现大语言模型（LLM）推理性能的最大化。其核心功能定位为 **“异构融合推理加速”**，主要包含以下两大能力：

* **推理动态调度**

* **CPU 推理性能加速**

该系统特别针对 LLM 推理过程中不同阶段的计算特性进行优化，提升整体吞吐率与资源利用率。sysHAX目前的应用场景是单机多卡（CPU+xPU），未来计划支持多机多卡场景。

## 1.2. 推理动态调度

在典型的自回归文本生成过程中，推理可分为两个关键阶段：

* **Prefill 阶段**：对输入提示（prompt）进行上下文编码，属于计算密集型任务，适合在高算力设备（GPU/NPU/...）上执行。

* **Decode 阶段**：逐 token 生成输出内容，具有更高的内存访问频率，属于访存密集型任务，更适合在具备高内存带宽和灵活调度能力的 CPU 上运行。

为了充分利用以上特性，sysHAX基于vllm推理框架 实现了 **动态任务拆分（PD分离）与调度机制（Decode任务的动态调度）**。

* **PD分离**：将全部的 Prefill 请求路由至 xPU（GPU/NPU/...），而将 Decode 请求则交由xPU和优化后的 CPU 后端处理，从而实现计算资源的最优匹配与利用。

* **Decode任务的动态调度**: 根据xPU和CPU上的实时负载，动态将Decode请求分配到负载更低的设备处理，以实现吞吐量最大化。

## 1.3. CPU 推理加速

为了提升 Decode 阶段在 CPU 端的执行效率，sysHAX 在底层集成了多项性能优化技术，包括：

* **NUMA 亲和性调度**：确保线程与本地内存节点绑定，减少跨节点访问延迟；

* **多级并行优化**：充分利用多核并发能力，提升指令级与任务级并行度；

* **算子级优化**：应用SIMD指令集，大幅提升推理过程中的矩阵乘积等计算速度，显著降低计算开销。

这些优化显著提升了 CPU 在 LLM 推理中的响应速度与吞吐能力。

***

# 2. 组件介绍

|**组件**|**运行的设备**|介绍|
|---|---|---|
|vllm-cpu|CPU|主要处理一部分Decode请求|
|vllm-gpu|GPU|主要处理全部的Prefill请求和一部分Decode请求|
|sysHAX|CPU|代理用户的请求，将所有的Prefill请求发往vllm-gpu。根据负载均衡的情况，将Decode请求发往vllm-cpu或vllm-gpu|

***

# 3. 软件环境

部署时要求宿主机已安装NVIDIA Container Toolkit，可参考：

[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

|**类型**|**版本要求**|**说明**|
|---|---|---|
|操作系统|openEuler 22.03 LTS 、openEuler 24.03 LTS |-|
|python|3.11及以上|部署vllm服务需要python|
|docker|25及以上|vllm部署，实现开箱即用|
|模型|大部分主流模型| vllm支持大部分主流模型的部署 |

***

# 4. 硬件规格

|**类型**|**型号**|**说明**|
|---|---|---|
|GPU|Nvidia系列GPU| 当前版本支持在Nvidia系列GPU上进行推理 |
|服务器|鲲鹏920系列服务器，推荐920 7280Z以上系列|推理加速的功能是参考920服务器的特性实现。|

***

# 5. 快速开始

先把模型存放在宿主机的`/home/models`路径下，然后创建容器时，再将`/home/models`挂载到容器内的相同路径，就可以在容器内访问模型了。推理前，需要先搭建vllm-gpu容器、vllm-cpu容器、sysHAX服务。

## 5.1. 搭建vllm-gpu容器

### 5.1.1. 创建vllm-gpu容器

```shell
# 从远端拉取vllm(GPU)的docker镜像
docker pull hub.oepkgs.net/neocopilot/syshax/syshax-vllm-gpu:0.2.1

# 创建名为vllm_gpu的docker容器，设置ipc为可共享，方便跟vllm-cpu共享ipc资源。创建结束后自动进入容器的工作目录
docker run --name vllm_gpu \
    --ipc="shareable" \
    --shm-size=64g \
    --gpus=all \
    -p 8001:8001 \
    -v /home/models:/home/models \
    -w /home/ \
    -it hub.oepkgs.net/neocopilot/syshax/syshax-vllm-gpu:0.2.1 bash
```

docker run命令的参数解释：

|**参数**|**解释**|
|---|---|
|`name` |指定容器名。|
|`ipc` |`ipc`命名空间，用来使能vllm-cpu和vllm-gpu之间的通信。|
|`shm-size`|设置容器的共享内存大小，单位为GB。|
|`gpus`|允许容器使用宿主机所有GPU设备|
|`p`|port的缩写。将容器内端口映射到宿主机端口。|
|`v`|将宿主机的目录挂载到容器的路径。|
|`w`|设置容器的工作目录。|
|`it`| -i：保持 STDIN 打开。-t： 分配伪终端。使用-it会在创建完容器后进入容器|
|`hub.oepkgs.net/neocopilot/syshax/syshax-vllm-gpu:0.2.1`|指定使用的容器镜像名称及版本。|

### 5.1.2. 部署

现在可以部署vllm-gpu来提供推理服务了。注意：如果需要支持sysHAX的PD分离，必须在vllm serve命令中加上`--enable_auto_pd_offload`来启动PD分离功能。

```shell
# 部署vllm(GPU)服务。注意默认是禁止pd分离的，要开启pd分离，要在vllm serve命令中添加--enable_auto_pd_offload启动参数
vllm serve /home/models/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype=half \
    --block_size=16 \
    --preemption_mode=swap \
    --max_model_len=8192 \
    --tensor-parallel-size 2 \
    --gpu_memory_utilization=0.8 \
    --enable_auto_pd_offload
```

vllm serve的参数解释如下：

|**参数**|**解释**|
|---|---|
|`/home/models/DeepSeek-R1-Distill-Qwen-32B`|模型存储路径。|
|`host`|指定服务器监听的网络接口。|
|`port`|指定服务器监听的网络端口号。|
|`dtype`|指定模型权重加载的数据类型。可选值为`auto`、`half`、`bfloat16`、`float32`。|
|`block_size`|定义 PagedAttention 中KV Cache块的大小，表示一个KV Cache块中包含的token的个数。通常可选值为`8`、`16`。|
|`preemption_mode`|当新请求到达而资源不足时，vLLM 支持通过“抢占”旧请求的方式释放资源。该参数控制如何处理被抢占的请求。可选值为`recompute`、`swap`、`none`。|
|`max_model_len`|设置模型支持的最大上下文长度（提问+回答的总token长度）。|
|`tensor-parallel-size`|张量并行度，表示将模型切分到多少个 GPU 上。|
|`gpu_memory_utilization`|GPU 现存占有率。|
|`enable_auto_pd_offload`|如果需要支持sysHAX的PD分离功能，需要添加该参数。|

**Tips**

* **如何设置`tensor-parallel-size`?**

使用`nvidia-smi`命令可以查看GPU的卡数和每张卡的显存量。保证`tensor-parallel-size`不超过服务器的卡数。

* **如何设置`gpu_memory_utilization`?**

保证：模型运行时的显存开销 ≤ 服务器的可用显存。

模型运行时的显存开销包括模型权重占用的显存开销以及额外显存开销（KV 缓存、激活值、临时缓冲区等），为了方便计算，模型运行时的显存开销可以简单算作模型权重的`1.2-1.5`倍。模型权重=权重数量 x 权重数值类型。例如，对于32B模型，部署服务时设置`dtype`为`half`，那么每个参数占`2`字节，一共`32B`参数，因此模型权重占`64B`，模型运行时的显存开销大概为`77B-96B`。

服务器的可用显存=服务器中GPU的卡数 x 每张卡的显存量 x `gpu_memory_utilization`。服务器中GPU的卡数和每张卡的显存量可以通过`nvidia-smi`命令查看。

## 5.2. 搭建vllm-cpu容器

### 5.2.1. 创建vllm-cpu容器

```shell
# 从远端仓库拉取镜像，镜像中配置了vllm-cpu的相关环境
docker pull hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1
# 创建名为vllm_cpu的容器，跟名为vllm_gpu的容器共享IPC资源。创建完容器自动进入容器的工作目录
docker run --name vllm_cpu \
    --ipc container:vllm_gpu \
    --shm-size=64g \
    --privileged \
    -p 8002:8002 \
    -v /home/models:/home/models \
    -w /home/ \
    -it hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1 bash
```

docker run命令的参数解释：

|**参数**|**解释**|
|---|---|
|`name` |指定容器名。|
|`ipc` |ipc命名空间，用来使能vllm-cpu和vllm-gpu之间的通信。|
|`shm-size`|设置容器的共享内存大小，单位为GB。|
|`privileged`|授予容器特权模式，使其拥有对主机设备的广泛访问权限。|
|`p`|port的缩写。将容器内端口映射到宿主机端口。|
|`v`|将宿主机的目录挂载到容器的路径。|
|`w`|设置容器的工作目录。|
|`it`| \-i：保持 STDIN 打开。\-t： 分配伪终端。|
|`hub.oepkgs.net/neocopilot/syshax/syshax-vllm-cpu:0.2.1`|指定使用的容器镜像名称及版本。|

### 5.2.2. 部署

```shell

# 部署vllm(CPU)服务。要开启pd分离，要添加--enable_auto_pd_offload
INFERENCE_OP_MODE=fused \
OMP_NUM_THREADS=160 \
CUSTOM_CPU_AFFINITY=0-159 \
SYSHAX_QUANTIZE=q4_0 \
NRC=4 \
vllm serve /home/models/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 \
    --port 8002 \
    --dtype=half \
    --block_size=16 \
    --preemption_mode=swap \
    --max_model_len=8192 \
    --enable_auto_pd_offload
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
| `enable_auto_pd_offload` |如果需要支持sysHAX的PD分离功能，需要添加。|

### 5.2.3. 部署样例

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

## 5.3. 搭建sysHAX服务

在sysHAX场景中，用户需要将请求发往sysHAX服务。sysHAX根据CPU和GPU的特性分配Prefill和Decode请求。启动vllm-cpu和vllm-gpu的服务时加上启动参数`enable_auto_pd_offload`，就可以让它们支持PD分离特性。

sysHAX服务可以在宿主机或者容器内搭建。

### 5.3.1. 配置sysHAX

下载sysHAX源代码，并对其配置。

```shell
# 下载sysHAX源代码。目前sysHAX更新到0.2.1版本，下载最新的版本的代码
git clone -b v0.2.1 https://gitee.com/openeuler/sysHAX.git

# 现在开始配置sysHAX

# 初始化配置文件，该命令会在根目录的config文件夹下拷贝yaml文件
python3 cli.py init

# 设置GPU端口，sysHAX将使用此端口访问vllm-gpu
python3 cli.py config gpu.port 8001

# 设置CPU端口，sysHAX将使用此端口访问vllm-cpu
python3 cli.py config cpu.port 8002

# 设置sysHAX的服务端口
python3 cli.py config conductor.port 8010

# 配置sysHAX是否开启自动 PD offload（true/false）
python3 cli.py config auto_pd_offload true

# 配置sysHAX在CPU 侧最大并发量。这里示例设置为5，请根据实际情况设置
python3 cli.py config cpu_max_batch_size 5

# 配置sysHAX请求超时时间（秒）。这里设置为3600s，请根据实际情况设置
python3 cli.py config request_timeout 3600
```

### 5.3.2. 启动sysHAX

```shell
# 启动sysHAX
python3 cli.py run
```

如果希望查看详细的DEBUG信息，可以通过如下命令：

```shell
DEBUG=1 python3 cli.py run
```

如果希望查看sysHAX的所有配置命令，可以执行`python3 cli.py config --help`。

## 5.4. 发起推理请求

至此，已经部署完成vllm-cpu、vllm-gpu以及sysHAX，现在来发起推理请求。

### 5.4.1. curl请求

```shell
curl http://0.0.0.0:8010/v1/chat/completions -H "Content-Type: application/json" -d '{
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

# 附录

## 常见报错与解决办法

1. 显存溢出，即`Torch.OutOfMemoryError: Cuda Out of Memory`

参考GPU部署部分的Tips，验证是否满足：模型运行时的显存开销小于等于服务器的可用显存。如果满足，执行`nvidia-smi`命令查看是否其他进程占用显存。

2. ipc连接失败，即`docker: Error response from daemon: fail to join ipc namespace`

当vllm-gpu容器创建时ipc设置为`shareable`，vllm-cpu容器创建时设置ipc为`container:vllm_gpu`，vllm-gpu容器和vllm-cpu容器需要共享ipc资源时。注意：虽然创建容器时出现这种报错，但是容器确实已经创建成功，需要使用`docker rm`命令将出现报错的容器删除。

3. 绑定端口失败，即`port is already allocated`

端口已经被占用，请换一个端口来创建容器或者先杀掉占用此端口的进程。
