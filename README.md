# 现在华为官方主推的 MindIE (MindSpore Inference Engine) 结合 ATB 算子库，已经原生支持直接加载 HuggingFace 格式的权重，这极大简化了部署流程。
# bg：MindIE（MindSpore Inference Engine）是华为专门为大模型部署打造的高性能引擎，可以把它理解为昇腾版的 vLLM 或 TensorRT-LLM。它底层集成了 ATB（Ascend Tensor Boost）算子加速库，现在的版本已经可以做到“免转换”，直接读取 HuggingFace 上的 .safetensors 权重文件。自带顶级加速度：这是追求优化的关键。MindIE 内部原生实现了PagedAttention： 极大缓解显存碎片化，提升并发量。Continuous Batching： 提升高并发场景下的吞吐量和FlashAttention： 针对 910A 底层架构优化过的高效注意力算子，加速生成速度。
# 服务器基础检查与 NPU 验证
## 登录远程服务器
    环境：
    119.6.186.139
    User root
    Port 10263
登录命令   ssh root@119.6.186.139 -p 10263
## 找回历史命令
    history | grep -E "docker|npu-smi|huggingface|wget"
## 发现
## 登录远程服务器后，第一步是确认昇腾 910A 芯片的驱动和固件状态。
1. 这相当于 NVIDIA 的 nvidia-smi，是昇腾生态中最基础的命令。
命令行：npu-smi info
### 关键看点：
    Card / Device: 确认有几张卡（通常标号为 0, 1, 2...）。
    Health: 必须是 OK。
    Memory-Usage: 查看显存占用，确保有空闲显存（910A 通常为 32G 显存版本）。
2. 检查系统架构与磁盘
确认磁盘空间足够，大模型及其容器镜像通常需要数百 GB 空间。
    arch       # 确认架构，通常是 aarch64 或 x86_64
    df -h      # 检查磁盘空间，找一个空间最大的目录用来存模型
# 启动昇腾专属 Docker 容器
昇腾环境的依赖非常复杂（包含 CANNToolkit、驱动库等），建议使用官方镜像，不要在宿主机上强行编译。我们需要挂载昇腾的 NPU 设备节点（/dev/davinci*）和宿主机驱动。
1. 拉取 MindIE 镜像
前往华为云 SWR 或昇腾社区获取最新的 MindIE 镜像（这里以通用标签为例）：
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:latest
2. 启动容器（保姆级挂载命令）
假设你的模型文件将存放在宿主机的 /data/models 目录下，我们需要启动一个拥有硬件调用权限的容器。
docker run -it -d \
    --name ascend_llm \
    --net=host \
    --shm-size=500g \
    --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /data/models:/workspace/models \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:latest \
    /bin/bash
注：进入容器后，可以再次输入 npu-smi info 测试，如果能看到信息，说明容器与 NPU 握手成功。
解释：
    --device=/dev/davinci0: 挂载 0 号 NPU。如果需要多卡，依次添加 --device=/dev/davinci1 等。
    /usr/local/Ascend/driver: 灵魂挂载，容器必须依赖宿主机的底层驱动。
    --shm-size=500g: 扩大共享内存，大模型推理时卡间通信（如 NCCL/HCCL）极度依赖内存。
进入容器：
    docker exec -it ascend_llm bash
# 模型下载与格式准备（HuggingFace -> 昇腾）
因为是用来推理部署并优化速度，可以直接选用MindIE直接加载。MindIE 可以直接读取标准 HuggingFace 的模型文件夹。
1. 下载 HuggingFace 模型
如果是国内服务器，建议使用模型库镜像下载（以Qwen为例）：
pip install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen3.5-9B-Instruct --local-dir /workspace/models/Qwen2.5-7B-Instruct
确保下载下来的文件夹中包含 .safetensors 和 config.json。这就是昇腾需要的最终格式了。
# 配置并启动 MindIE 推理服务
现在我们使用昇腾的高性能推理引擎 MindIE 来启动服务。
1. 修改配置文件
找到 MindIE 的服务配置文件（通常位于容器内的 /usr/local/Ascend/mindie/latest/mindieservice_daemon/conf/config.json）。
使用 vi 打开并修改以下几个核心参数：
    {
        "ServerConfig": {
            "ipAddress": "0.0.0.0",
            "port": 1040,
            "httpsEnabled": false
        },
        "BackendConfig": {
            "npuDeviceIds": [[0]], 
            "ModelDeployConfig": {
                "ModelConfig": [
                    {
                        "modelInstanceType": "Standard",
                        "modelName": "qwen2.5-7b",
                        "modelWeightPath": "/workspace/models/Qwen2.5-7B-Instruct", 
                        "npuMemSize": 16,
                        "cpuMemSize": 16,
                        "worldSize": 1
                    }
                ]
            }
        }
    }
npuDeviceIds: [[0]] 表示单卡部署在卡 0 上。如果是双卡部署 72B 这种大模型，写 [[0, 1]]。
modelWeightPath: 指向你刚才下载的 HuggingFace 文件夹绝对路径。
worldSize: 必须与你分配的卡数保持一致。
2. 拉起服务
在容器内执行服务启动命令（推荐放在后台运行）：
    cd /usr/local/Ascend/mindie/latest/mindie-service/bin
    nohup ./mindieservice_daemon > output.log 2>&1 &
查看日志确认模型是否成功加载到 NPU：
    tail -f output.log
当看到类似 Daemon start success 或 Server is listening on 0.0.0.0:1040 时，说明昇腾 NPU 已经成功吃下了你的大模型。
# 测试调用
MindIE-Service 提供与 OpenAI 兼容的 API 接口。我们可以直接用 Linux 基础命令 curl 来测试 910A 的推理效果。
    curl -X POST http://127.0.0.1:1040/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "qwen2.5-7b",
    "messages": [
        {
        "role": "user",
        "content": "请写一段Python代码来实现快速排序。"
        }
    ],
    "max_tokens": 512,
    "stream": true
    }'
如果终端开始源源不断地吐出中文字符和代码，恭喜你，你已经成功在昇腾 910A 上跑通了大模型的推理全流程。
# 
