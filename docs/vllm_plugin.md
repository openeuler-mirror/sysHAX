# sysHAX支持vllm_plugin方案设计文档

## 1. 需求分析

### 1.1 背景
当前的 sysHAX-adapter 通过直接导入 vLLM 0.9.1 的相关代码实现了 AF 分离（注意力提取分离）和 PD 分离（参数数据分离）功能。这种实现方式存在以下问题：
- 代码耦合度高，难以维护和升级
- 与 vLLM 版本绑定紧密，升级 vLLM 版本需要大量修改
- 部署复杂，需要修改 vLLM 源代码
- 不支持动态加载和卸载

### 1.2 必要性
将 sysHAX-adapter 转化为 vLLM 0.9.1 插件的必要性体现在：
- **解耦**：减少与 vLLM 源代码的直接耦合，便于独立维护和升级
- **兼容性**：通过插件机制，更好地适配 vLLM 的版本升级
- **易用性**：提供简单的安装和使用方式，无需修改 vLLM 源代码
- **灵活性**：支持动态加载和卸载，便于在不同场景下使用
- **可扩展性**：为未来功能扩展提供更好的架构基础

### 1.3 功能需求
转化后的 vllm_plugin 需要实现以下核心功能：

#### 1.3.1 AF 分离功能
- MoE 模型加载器：支持 MoE 模型的激活函数（FFN）分离
- NUMA-aware 专家加载：将 MoE 专家分离到 CPU，其他权重保留在 GPU
- 专家权重管理：合并和管理专家权重，优化访问效率
- 支持通过配置选择加载方案

#### 1.3.2 PD 分离功能
- 自动 PD 卸载：通过环境变量控制自动 PD 卸载机制
- 调度器优化：优先处理 PD 预填充任务
- 共享内存管理：实现跨进程 KV 缓存存储和传输
- 动态任务接力：支持 decode 阶段的任务接力

#### 1.3.3 配置与管理
- 环境变量配置：支持 ENABLE_AUTO_PD_OFFLOAD、MODEL_LOADING_SCHEME、USE_GREDDY 等配置
- 插件生命周期管理：提供 setup/teardown 接口
- 与 vLLM 0.9.1 兼容的动态补丁机制

### 1.4 影响分析

#### 1.4.1 对 vLLM 的影响
- 无需修改 vLLM 源代码
- 通过动态补丁机制添加功能
- 保持与 vLLM 0.9.1 的兼容性

#### 1.4.2 对原有功能的影响
- 完全保留原有 AF 分离和 PD 分离功能
- 保持 API 兼容性，减少用户代码修改
- 优化性能和稳定性

#### 1.4.3 对部署的影响
- 简化部署流程，只需安装插件
- 支持灵活的配置和功能开关

## 2. 方案设计

### 2.1 整体方案分析

#### 2.1.1 架构设计
采用分层架构设计，将插件功能划分为以下层次：
- **接口层**：提供插件的安装和使用接口
- **核心功能层**：实现 AF 分离和 PD 分离的核心逻辑
- **适配层**：通过动态补丁机制与 vLLM 集成
- **配置层**：管理插件的配置参数

#### 2.1.2 技术选型
- **插件机制**：基于 vLLM 的插件系统，通过 entry points 注册插件
- **动态补丁**：使用 monkey patching 技术修改 vLLM 内部类和方法
- **共享内存**：使用 Python multiprocessing.shared_memory 实现跨进程数据共享
- **配置管理**：基于环境变量的配置系统

### 2.2 详细设计

#### 2.2.1 目录结构
```
vllm_plugin/
├── __init__.py              # 插件入口点
├── setup.py                 # 插件安装配置
├── syshax_config.py         # 配置管理
├── shared_memory_manager.py # 共享内存管理
├── scheduler_patch.py       # 调度器补丁
├── syshax_engine.py         # 引擎补丁
├── syshax_scheduler.py      # 调度器工具函数
├── cpu_loader_tools.py      # CPU 权重加载工具
└── model_loader/
    └── moe_af_separated_loader.py # MoE AF 分离加载器
```

#### 2.2.2 核心组件设计

##### 2.2.2.1 AF 分离组件
- **SyshaxMoEAFSeparatedModelLoader**：继承自 BaseModelLoader，实现 MoE 模型的 AF 分离加载
- **cpu_loader_tools**：提供 CPU 权重存储和管理功能

##### 2.2.2.2 PD 分离组件
- **SharedMemoryManager**：管理跨进程共享内存，实现 KV 缓存存储和传输
- **scheduler_patch**：动态修改 vLLM Scheduler 类，添加 PD 分离逻辑

##### 2.2.2.3 配置组件
- **SyshaxConfig**：基于环境变量的配置管理，支持 ENABLE_AUTO_PD_OFFLOAD、MODEL_LOADING_SCHEME、USE_GREDDY 等配置

#### 2.2.3 关键流程设计

##### 2.2.3.1 AF 分离加载流程
1. 用户指定加载格式为 "syshax_moe_af_separated"
2. vLLM 调用注册的 SyshaxMoEAFSeparatedModelLoader
3. 加载器将 MoE 专家分离到 CPU，其他权重保留在 GPU
4. 合并专家权重并优化访问效率

##### 2.2.3.2 PD 分离流程
1. 用户设置环境变量 ENABLE_AUTO_PD_OFFLOAD=true
2. 插件启动时初始化共享内存管理器
3. 调度器优先处理 PD 预填充任务
4. 从共享内存加载 KV 缓存
5. 支持动态任务接力

### 2.3 API 设计

#### 2.3.1 插件接口
```python
# 插件安装
pip install -e /path/to/vllm_plugin

# 插件导入和使用
from vllm_plugin import setup, teardown

# 初始化插件
setup()

# 使用完毕后清理资源
teardown()
```

#### 2.3.2 配置接口
```python
# 通过环境变量配置
export ENABLE_AUTO_PD_OFFLOAD=true
# 0: 默认加载方案，1: AF 分离加载方案
export MODEL_LOADING_SCHEME=1
export USE_GREDDY=true
```

#### 2.3.3 AF 分离使用接口
```python
# 在 vLLM 配置中指定加载格式
from vllm import LLM, SamplingParams

llm = LLM(
    model="model_name_or_path",
    load_format="syshax_moe_af_separated"
)
```

#### 2.3.4 PD 分离使用接口
```python
# 在 SamplingParams 中指定 PD 相关参数
from vllm import LLM, SamplingParams

# 创建采样参数，指定请求 ID 和解码 token 数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    request_id_inference="parent_request_id",  # PD 接力请求
    num_decode_tokens=5  # PD 预解码 token 数
)

# 生成文本
outputs = llm.generate("Hello, my name is", sampling_params)
```

## 3. 测试设计

### 3.1 测试目标
验证 vllm_plugin 能够完全替代 sysHAX-adapter-dev，实现所有原有功能，并确保与 vLLM 0.9.1 的兼容性。

### 3.2 测试策略
采用分层测试策略：
- **单元测试**：测试各个组件的功能
- **集成测试**：测试插件与 vLLM 的集成
- **功能测试**：测试 AF 分离和 PD 分离功能
- **性能测试**：测试插件的性能影响

### 3.3 测试用例设计

#### 3.3.1 插件基本功能测试
| 测试用例 | 测试目的 | 测试步骤 | 预期结果 |
|---------|---------|---------|---------|
| 插件导入 | 验证插件可以正常导入 | 1. 导入 vllm_plugin<br>2. 检查插件版本 | 成功导入，显示正确版本 |
| 插件初始化 | 验证 setup/teardown 功能 | 1. 调用 setup()<br>2. 调用 teardown() | 无错误发生 |

#### 3.3.2 AF 分离功能测试
| 测试用例 | 测试目的 | 测试步骤 | 预期结果 |
|---------|---------|---------|---------|
| MoE 模型加载 | 验证 MoE 模型可以通过插件加载 | 1. 使用 syshax_moe_af_separated 格式加载 MoE 模型 | 模型加载成功 |
| 专家分离验证 | 验证专家权重被分离到 CPU | 1. 加载 MoE 模型<br>2. 检查专家权重位置 | 专家权重在 CPU，其他权重在 GPU |

#### 3.3.3 PD 分离功能测试
| 测试用例 | 测试目的 | 测试步骤 | 预期结果 |
|---------|---------|---------|---------|
| PD 预填充任务 | 验证 PD 预填充任务被优先处理 | 1. 提交包含 num_decode_tokens 的请求<br>2. 检查任务执行顺序 | PD 预填充任务优先执行 |
| KV 缓存加载 | 验证可以从共享内存加载 KV 缓存 | 1. 提交带有 request_id_inference 的请求<br>2. 检查 KV 缓存加载 | 成功从共享内存加载 KV 缓存 |

### 3.4 测试环境
- 操作系统：Linux
- Python 版本：3.8+
- vLLM 版本：0.9.1
- 硬件：支持 CUDA 的 GPU

### 3.5 测试工具
- pytest：单元测试和集成测试
- cProfile：性能测试
- NVIDIA Nsight Systems：性能分析

## 4. 部署方案

### 4.1 安装方式
```bash
# 开发模式安装
pip install -e /path/to/vllm_plugin

# 生产模式安装
pip install /path/to/vllm_plugin
```

### 4.2 配置方式
通过环境变量配置插件功能：
```bash
export ENABLE_AUTO_PD_OFFLOAD=true
export MODEL_LOADING_SCHEME=1
export USE_GREDDY=true
```

### 4.3 使用方式
```python
# 导入并初始化插件
from vllm_plugin import setup
setup()

# 使用 vLLM
from vllm import LLM, SamplingParams

# AF 分离示例
llm_af = LLM(model="moe_model", load_format="syshax_moe_af_separated")

# PD 分离示例
sampling_params = SamplingParams(
    request_id_inference="parent_id",
    num_decode_tokens=5
)
llm_pd = LLM(model="model_name")
outputs = llm_pd.generate("Hello", sampling_params)

# 清理资源
from vllm_plugin import teardown
teardown()
```

## 5. 风险评估与应对措施

### 5.1 风险评估
| 风险 | 影响 | 可能性 | 应对措施 |
|-----|-----|-------|--------|
| vLLM API 变更 | 插件无法正常工作 | 中 | 实现版本兼容层，适配不同 vLLM 版本 |
| 性能下降 | 影响模型推理速度 | 低 | 优化代码实现，减少额外开销 |
| 内存泄漏 | 系统资源耗尽 | 低 | 实现完善的资源清理机制 |
| 兼容性问题 | 与其他插件冲突 | 低 | 避免全局命名空间污染，使用模块化设计 |

### 5.2 降级方案
如果插件出现问题，可以通过以下方式降级：
1. 卸载 vllm_plugin：`pip uninstall syshax-vllm-plugin`
2. 恢复使用原始的 sysHAX-adapter-dev

## 6. 总结

通过将 sysHAX-adapter-dev 转化为 vLLM 0.9.1 插件，可以实现代码解耦、提高兼容性和易用性。转化后的插件保持了原有 AF 分离和 PD 分离功能，同时提供了更好的维护性和扩展性。

转化方案采用分层架构设计，核心组件包括 AF 分离加载器、PD 分离调度器、共享内存管理器和配置管理系统。通过动态补丁机制与 vLLM 集成，无需修改 vLLM 源代码。

测试设计覆盖了插件的基本功能、AF 分离和 PD 分离功能，确保插件的正确性和性能。部署方案简单易用，支持通过环境变量配置插件功能。

该方案为 sysHAX-adapter-dev 提供了一个现代化、可维护的架构基础，便于未来功能扩展和版本升级。