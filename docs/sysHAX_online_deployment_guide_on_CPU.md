### 版本信息  
更新日期：2025年11月11日

***

# 1. 产品概述

## 1.1. sysHAX整体介绍

![sysHAX调度CPU推理示意图](pictures/CPU_arch.png "")

**sysHAX** 是一款面向 K+X（Kunpeng CPU + XPU\(GPU/NPU\)） 异构计算架构的推理加速系统，旨在通过智能任务调度与资源优化，充分发挥不同硬件平台（XPU与 CPU）的计算优势，实现大语言模型（LLM）推理性能的最大化。其核心功能定位为 **“异构融合推理加速”**，主要包含以下两大能力：

* **推理动态调度**

* **CPU 推理性能加速**

在纯CPU场景下，不涉及推理动态调度，只用到CPU推理性能加速。

该系统特别针对 LLM 推理过程中不同阶段的计算特性进行优化，提升整体吞吐率与资源利用率。sysHAX目前的应用场景是单机多卡（CPU+XPU），未来计划支持多机多卡场景。

***

# 2. 组件介绍

|**组件**|**运行的设备**|介绍|
|---|---|---|
|vllm\(CPU\)|CPU|处理推理请求|
|sysHAX|CPU|代理用户的请求，sysHAX再将请求发往vllm\(CPU\)|

***

# 3. 软件环境

|**类型**|**版本要求**|**说明**|
|---|---|---|
|操作系统|openEuler 22.03 LTS 、openEuler 24.03 LTS|\-|
|python|3.11及以上|部署vllm服务需要python|
|docker|25.0.3及以上|用docker将vllm\(CPU\)和vllm\(XPU\)的配置环境隔离，避免互相干扰。|
|模型|DeepSeek\-R1\-Distill\-Qwen\-32B|文档中以DeepSeek\-R1\-Distill\-Qwen\-32B为例介绍，可将DeepSeek\-R1\-Distill\-Qwen\-32B改为需要部署的模型。将需要部署的模型放在/home/models路径下|

***

# 4. 硬件规格

|**类型**|**型号**|**说明**|
|---|---|---|
|服务器|920系列arm架构服务器，推荐920 7280Z以上系列|推理加速的功能是参考920服务器（特别是920 7280Z服务器）的特性实现。|

***

# 5. 快速开始

先把模型存放在宿主机的/home/models路径下，然后创建容器时，再将/home/models挂载到容器内的相同路径，就可以在容器内访问模型了。推理前，需要先搭建vllm\(CPU\)容器、syshax服务。

## 5.1. 搭建vllm\(CPU\)容器

### 5.1.1. 创建vllm\(CPU\)容器

```shell
# 从远端仓库拉取镜像，镜像中配置了vllm(CPU)的相关环境
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
|name |指定容器名。|
|shm\-size|设置容器的共享内存大小，单位为GB。|
|privileged|授予容器特权模式，使其拥有对主机设备的广泛访问权限。|
|p|port的缩写。将容器内端口映射到宿主机端口。|
|v|将宿主机的目录挂载到容器的路径。|
|w|设置容器的工作目录。|
|it| \-i：保持 STDIN 打开。\-t： 分配伪终端。|
|hub.oepkgs.net/neocopilot/syshax/syshax\-vllm\-cpu|指定使用的容器镜像名称及版本。hub.oepkgs.net/neocopilot/syshax/syshax\-vllm\-cpu镜像已经配置好了vllm\(CPU\)，使用此镜像无需再进行配置|

### 5.1.2. 部署

```shell
#部署vllm(CPU)服务
INFERENCE_OP_MODE=fused OMP_NUM_THREADS=160 CUSTOM_CPU_AFFINITY=0-159 SYSHAX_QUANTIZE=q4_0 NRC=4 \
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
|INFERENCE\_OP\_MODE|vllm\_syshax自定义环境变量，表示是否启动CPU推理加速。是可选值：fused、None。|
|OMP\_NUM\_THREADS|vllm\_syshax自定义环境变量，表示CPU推理加速时开启的线程数量。其值&lt;=可用的CPU核的数量。用lscpu命令可查看服务器中的CPU核数。|
|CUSTOM\_CPU\_AFFINITY|vllm\_syshax自定义环境变量，表示将线程绑定到哪些CPU核。注意这里的线程数量要跟OMP\_NUM\_THREADS一致，并且每个numa的线程数量要相等。格式为“start\-end:step”，step默认为1。例如，0\-159:2中，0、159、2分别代表开始的线程序号、结束的线程序号（包含）、步长。|
|SYSHAX\_QUANTIZE|vllm\_syshax自定义环境变量，表示采用的量化方式。不设置时默认不量化，设置时可选值为q8\_0、q4\_0，表示进行q8\_0量化、q4\_0量化。|
|NRC|vllm\_syshax自定义环境变量，表示i8mm指令处理的矩阵分块的大小。可选值为2、4。|
|/home/models/DeepSeek\-R1\-Distill\-Qwen\-32B|模型存储路径。|
|host |指定服务器监听的网络接口。|
|port |指定服务器监听的网络端口号。|
|dtype|指定模型权重加载的数据类型。可选值为auto、half、bfloat16、float32。|
|block\_size|定义 PagedAttention 中KV Cache块的大小，表示一个KV Cache块中包含的token的个数。可选值为8，16。|
|preemption\_mode|当新请求到达而资源不足时，vLLM 支持通过“抢占”旧请求的方式释放资源。该参数控制如何处理被抢占的请求。可选值为recompute、swap、none。|
|max\_model\_len|设置模型支持的最大上下文长度（提问+回答的总token长度）。|

### 5.1.3. 部署样例

下面使用一个920 7280Z的服务器为例，展示如何设置vllm serve命令的参数。

![lscp输出示例](pictures/lscpu.png "")

解释lscpu输出中的重点数值：

|**参数**|**解释**|
|---|---|
|On\-line CPU\(s\) list:  0\-159|在线 CPU 列表为0\-159。说明服务器可以开启160个虚线程（通常等于Socket\(s\) \* Core\(s\) per socket）|
|BIOS Model name:      Kunpeng 920 7280Z|CPU型号为Kunpeng 920 7280Z。目前CPU推理加速在Kunpeng 920 7270Z和Kunpeng 920 7280Z上支持效果最好。|
|Thread\(s\) per core:   1|每个核心1个线程（无超线程）|
|Core\(s\) per socket:   80|每个CPU插槽80个物理核心|
|Socket\(s\):            2|2个CPU插槽（2路服务器）|
|Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint svei8mm svef32mm svef64mm                         svebf16 i8mm bf16 dgh rng ecv|Flags 列出了CPU支持的特性。其中比较重要的是asimd 、sve 、svei8mm 、i8mm。如果缺少i8mm、asimd ，那CPU的推理加速的速度会受到较大影响。|
|NUMA:<br>NUMA node\(s\):4<br>  NUMA node0 CPU\(s\):    0\-39<br>  NUMA node1 CPU\(s\):    40\-79<br>  NUMA node2 CPU\(s\):    80\-119<br>  NUMA node3 CPU\(s\):    120\-159<br>|**NUMA node\(s\): 4** \- 4个NUMA节点。**NUMA node0 CPU\(s\): 0\-39** \- 节点0包含CPU 0\-39。**NUMA node1 CPU\(s\): 40\-79** \- 节点1包含CPU 40\-79。**NUMA node2 CPU\(s\): 80\-119** \- 节点2包含CPU 80\-119。**NUMA node3 CPU\(s\): 120\-159** \- 节点3包含CPU 120\-159。|

这个服务器有4个numa，每个numa拥有40个物理核。根据不同的使用场景，可以如下设置：

|**场景**|**参考部署服务命令**|
|---|---|
|只运行sysHAX跟vllm服务。并且需要CPU高速推理。|INFERENCE\_OP\_MODE=fused OMP\_NUM\_THREADS=160 CUSTOM\_CPU\_AFFINITY=0\-159 SYSHAX\_QUANTIZE=q4\_0 NRC=4 vllm serve /home/models/DeepSeek\-R1\-Distill\-Qwen\-32B \-\-host 0.0.0.0 \-\-port 8001 \-\-dtype=half  \-\-block\_size=16 \-\-preemption\_mode=swap \-\-max\_model\_len=8192|
|除了运行sysHAX跟vllm服务外，还需留部分算力给其他进程。需要CPU中等速度推理。|INFERENCE\_OP\_MODE=fused OMP\_NUM\_THREADS=120 CUSTOM\_CPU\_AFFINITY=0\-29,40\-69,80\-109,120\-149 SYSHAX\_QUANTIZE=q4\_0 NRC=4 vllm serve /home/models/DeepSeek\-R1\-Distill\-Qwen\-32B \-\-host 0.0.0.0 \-\-port 8001 \-\-dtype=half  \-\-block\_size=16 \-\-preemption\_mode=swap \-\-max\_model\_len=8192|

## 5.2. 搭建syshax服务

在sysHAX场景中，用户需要将请求发往sysHAX服务。

目前已经搭建好vllm\(CPU\)容器了，现在搭建sysHAX服务对vllm\(CPU\)实现调度。

syshax服务在宿主机搭建，一共包含两个步骤：配置sysHAX、启动sysHAX。

### 5.2.1. 配置sysHAX

下载sysHAX源代码，并配置sysHAX。

```shell
# 下载sysHAX源代码。目前sysHAX更新到0.2.1版本，下载最新的版本的代码
git clone -b v0.2.1 https://gitee.com/openeuler/sysHAX.git

# 现在开始配置sysHAX

# 初始化一个新的配置文件，这一命令会在根目录的config文件夹下生成一个yaml文件，主要用以配置vllm(CPU)、sysHAX的对应ip地址以及端口，和模型名model_name
python3 cli.py init

# python3 cli.py init生成的配置文件中，CPU、NPU、sysHAX的对应ip地址初始为0.0.0.0，即本机ip。由于目前sysHAX在本机下运行vllm(CPU)、vllm(NPU)、sysHAX，因此这里不设置他们的ip

#  对python3 cli.py init生成的yaml配置文件设置端口，sysHAX将使用此端口访问vllm(CPU)
python3 cli.py config gpu.port 8001
python3 cli.py config cpu.port 8001

# 配置sysHAX的端口。sysHAX代理用户请求，即用户将请求发给sysHAX，sysHAX再将请求发给vllm(CPU)
python3 cli.py config conductor.port 8010

# 配置sysHAX是否开启自动 PD offload（true/false），默认不开启PD offload。这里是纯CPU环境，因此关闭
python3 cli.py config auto_pd_offload false

# 配置sysHAX在CPU 侧最大并发量。这里示例设置为50，请根据实际情况设置
python3 cli.py config cpu_max_batch_size 50

# 配置sysHAX请求超时时间（秒）。这里设置为3600s，请根据实际情况设置
python3 cli.py config request_timeout 3600
```

### 5.2.2. 启动sysHAX

```shell
#启动sysHAX有两种方式
#方式一
python3 main.py
#方式二
python3 cli.py run
```

如果希望查看详细的DEBUG信息，可以设置环境变量DEBUG=1再启动sysHAX。如果希望查看sysHAX的所有配置命令，可以执行python3 cli.py config \-\-help。

## 5.3. 发起推理请求

至此，已经部署完成vllm\(CPU\)以及sysHAX，现在来发起推理请求。

### 5.3.1. curl请求

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
|http://0.0.0.0:8010/v1/chat/completions|格式：协议://ip地址:端口/路径。含义：向运行在本地（0.0.0.0）或远程主机（x.x.x.x）上、监听 8010 端口的一个 LLM 推理服务，发送一个符合 OpenAI 格式的聊天补全请求。|
|stream|启用流式传输，结果将以数据流形式逐步返回\(一个token一个token地返回\)。stream设置为false可一次性返回所有生成的内容。|
|max\_tokens|限制响应生成的最大token数|

***

# 6. 附录
