# CRKIT SDK Docker 部署指南

## 📋 概述

CRKIT SDK提供多阶段Docker部署方案，支持生产运行、开发调试和自动化测试等多种场景。

### 🎯 Docker镜像架构

```
ubuntu:22.04 (base)
    ↓
[builder] 编译阶段 (ONNX Runtime + OpenCV + 编译工具)
    ├─→ [runtime] 生产运行 (最小化，仅运行时依赖)
    ├─→ [development] 开发调试 (包含gdb/valgrind等开发工具)
    └─→ [test] 自动化测试 (运行ctest)
```

### 📦 镜像说明

| 镜像 | 大小 | 用途 | 包含内容 |
|------|------|------|---------|
| **crkit-sdk:runtime** | ~800MB | 生产环境 | 仅运行时库+可执行文件 |
| **crkit-sdk:development** | ~2GB | 开发调试 | builder + 开发工具 |
| **crkit-sdk:test** | ~2GB | CI/CD测试 | builder + 测试套件 |

## 🚀 快速开始

### 前置要求

```bash
# 检查Docker版本
docker --version  # 需要 Docker 20.10+
docker-compose --version  # 需要 Docker Compose 1.29+
```

### 方式1: 使用docker-compose（推荐）

#### 1. 构建所有镜像

```bash
# 构建所有阶段的镜像
docker-compose build

# 或者只构建特定镜像
docker-compose build crkit-runtime
docker-compose build crkit-dev
docker-compose build crkit-test
```

#### 2. 运行测试验证

```bash
# 运行测试套件（30个测试用例）
docker-compose run --rm crkit-test

# 预期输出:
# ✅ 所有测试通过!
# 总测试数: 30
# 通过: 30 ✓
# 失败: 0 ✗
# 通过率: 100.0%
```

#### 3. 启动开发环境

```bash
# 启动交互式开发容器
docker-compose run --rm crkit-dev

# 在容器内编译和开发
root@container:/workspace# mkdir -p build && cd build
root@container:/workspace/build# cmake .. -DCMAKE_BUILD_TYPE=Release
root@container:/workspace/build# make -j$(nproc)
root@container:/workspace/build# ctest
```

#### 4. 运行生产环境

```bash
# 准备模型和数据（在宿主机上）
mkdir -p models data output logs

# 放置模型文件
cp your_model.onnx models/

# 放置测试图像
cp test_image.jpg data/

# 启动生产容器
docker-compose up -d crkit-runtime

# 进入容器执行推理
docker-compose exec crkit-runtime bash

# 在容器内运行推理
crkit@container:/app$ test_comprehensive  # 验证功能
crkit@container:/app$ object_detection_example /app/models/your_model.onnx /app/data/test_image.jpg 0.5 0.45
```

### 方式2: 使用原生Docker命令

#### 1. 构建镜像

```bash
# 构建生产运行镜像
docker build --target runtime -t crkit-sdk:runtime .

# 构建开发镜像
docker build --target development -t crkit-sdk:dev .

# 构建测试镜像
docker build --target test -t crkit-sdk:test .
```

#### 2. 运行测试

```bash
docker run --rm crkit-sdk:test
```

#### 3. 运行生产容器

```bash
docker run -it --rm \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/logs:/var/log/crkit:rw \
  -v $(pwd)/output:/app/output:rw \
  --name crkit-runtime \
  crkit-sdk:runtime bash
```

#### 4. 运行开发容器

```bash
docker run -it --rm \
  -v $(pwd):/workspace:rw \
  --name crkit-dev \
  crkit-sdk:dev bash
```

## 📂 目录挂载说明

### 生产环境挂载

```yaml
volumes:
  - ./models:/app/models:ro          # 模型目录（只读）
  - ./data:/app/data:ro              # 输入数据目录（只读）
  - ./logs:/var/log/crkit:rw         # 日志目录（读写）
  - ./output:/app/output:rw          # 输出结果目录（读写）
```

**目录结构示例**:
```
crkit_sdk/
├── models/
│   ├── detection_model.onnx       # 检测模型
│   └── classification_model.onnx  # 分类模型
├── data/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── batch/
│       ├── img001.jpg
│       └── img002.jpg
├── output/
│   ├── result1.jpg                # 可视化结果
│   └── detections.json            # 检测结果JSON
└── logs/
    ├── crkit.log.1                # 日志轮转文件
    └── crkit.log.2
```

### 开发环境挂载

```yaml
volumes:
  - .:/workspace:rw                  # 挂载整个项目（实时开发）
```

## 🔧 环境变量配置

### 运行时环境变量

```bash
# docker-compose.yml 中配置
environment:
  - LD_LIBRARY_PATH=/usr/local/lib        # 动态库路径
  - OMP_NUM_THREADS=4                     # OpenMP线程数
  - TZ=Asia/Shanghai                      # 时区
  - CRKIT_LOG_LEVEL=INFO                  # 日志级别（可选）
  - CRKIT_LOG_FILE=/var/log/crkit/app.log # 日志文件（可选）
```

### 自定义环境变量

```bash
# 方式1: 在docker-compose.yml中添加
environment:
  - CRKIT_LOG_LEVEL=DEBUG

# 方式2: 使用.env文件
echo "CRKIT_LOG_LEVEL=DEBUG" > .env
docker-compose --env-file .env up
```

## 🎯 使用场景

### 场景1: CI/CD自动化测试

```bash
# .gitlab-ci.yml 或 .github/workflows/test.yml
test:
  script:
    - docker-compose build crkit-test
    - docker-compose run --rm crkit-test
```

### 场景2: 批量图像推理

```bash
#!/bin/bash
# batch_inference.sh

# 启动容器
docker-compose up -d crkit-runtime

# 批量处理图像
for img in data/*.jpg; do
    docker-compose exec crkit-runtime \
        object_detection_example \
        /app/models/model.onnx \
        /app/data/$(basename $img) \
        0.5 0.45
done

# 停止容器
docker-compose down
```

### 场景3: 多模型服务

```yaml
# docker-compose.yml 扩展
services:
  crkit-detection:
    image: crkit-sdk:runtime
    volumes:
      - ./models/detection.onnx:/app/model.onnx:ro
    environment:
      - MODEL_TYPE=detection

  crkit-classification:
    image: crkit-sdk:runtime
    volumes:
      - ./models/classification.onnx:/app/model.onnx:ro
    environment:
      - MODEL_TYPE=classification
```

### 场景4: GPU加速部署（NVIDIA Docker）

```yaml
# docker-compose.gpu.yml
services:
  crkit-runtime-gpu:
    image: crkit-sdk:runtime
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0        # 使用GPU 0
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**运行**:
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

## 📊 资源限制

### CPU和内存限制

```yaml
# docker-compose.yml
services:
  crkit-runtime:
    deploy:
      resources:
        limits:
          cpus: '4'           # 最多使用4个CPU核心
          memory: 8G          # 最多使用8GB内存
        reservations:
          cpus: '2'           # 至少保证2个CPU核心
          memory: 4G          # 至少保证4GB内存
```

### 查看资源使用

```bash
# 实时监控容器资源
docker stats crkit-runtime

# 查看容器详情
docker inspect crkit-runtime
```

## 🐛 故障排查

### 问题1: 构建失败 - ONNX Runtime下载超时

**现象**:
```
wget: unable to resolve host address 'github.com'
```

**解决方案**:
```bash
# 方式1: 手动下载ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
# 放在项目根目录，修改Dockerfile使用本地文件

# 方式2: 使用代理
docker build --build-arg HTTP_PROXY=http://proxy:port --target runtime .
```

### 问题2: 运行时找不到共享库

**现象**:
```
error while loading shared libraries: libonnxruntime.so.1.16.3: cannot open shared object file
```

**解决方案**:
```bash
# 检查LD_LIBRARY_PATH
docker-compose exec crkit-runtime bash -c "echo $LD_LIBRARY_PATH"

# 手动更新动态链接库缓存
docker-compose exec crkit-runtime ldconfig

# 检查库文件
docker-compose exec crkit-runtime ls -la /usr/local/lib/libonnxruntime*
```

### 问题3: 权限错误

**现象**:
```
Permission denied: '/var/log/crkit/crkit.log'
```

**解决方案**:
```bash
# 在宿主机上修改日志目录权限
chmod -R 777 logs/

# 或者在docker-compose.yml中以root运行（不推荐）
user: root
```

### 问题4: 测试失败

**现象**:
```
Test #1: test_comprehensive ................***Failed
```

**解决方案**:
```bash
# 查看详细测试输出
docker-compose run --rm crkit-test ctest --verbose --output-on-failure

# 进入容器手动运行测试
docker-compose run --rm crkit-test bash
root@container:/build/build# ./test_comprehensive
root@container:/build/build# ./test_logger
```

## 🔒 安全最佳实践

### 1. 使用非root用户运行

Dockerfile已配置非root用户 `crkit`:
```dockerfile
RUN useradd -m -u 1000 -s /bin/bash crkit
USER crkit
```

### 2. 只读挂载模型和数据

```yaml
volumes:
  - ./models:/app/models:ro   # 只读，防止误修改
  - ./data:/app/data:ro       # 只读
```

### 3. 限制网络访问

```yaml
networks:
  crkit-network:
    driver: bridge
    internal: true  # 禁止外部网络访问
```

### 4. 扫描镜像漏洞

```bash
# 使用Trivy扫描
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image crkit-sdk:runtime
```

## 📈 性能优化

### 1. 多阶段构建缓存

```bash
# 使用BuildKit加速构建
DOCKER_BUILDKIT=1 docker build --target runtime .

# 缓存中间层
docker build --cache-from crkit-sdk:builder --target runtime .
```

### 2. 减小镜像大小

```dockerfile
# 已实现的优化:
# - 多阶段构建（runtime镜像不包含编译工具）
# - 清理apt缓存（rm -rf /var/lib/apt/lists/*）
# - 只复制必要文件
```

**镜像大小对比**:
- builder: ~2.5GB（包含编译工具）
- runtime: ~800MB（仅运行时依赖）
- 压缩率: 68%

### 3. 并行编译

```dockerfile
# Dockerfile中已配置
RUN make -j$(nproc)  # 使用所有可用CPU核心
```

## 📚 参考命令速查

### Docker命令

```bash
# 构建
docker build --target runtime -t crkit-sdk:runtime .
docker-compose build

# 运行
docker run -it --rm crkit-sdk:runtime bash
docker-compose up -d crkit-runtime

# 测试
docker run --rm crkit-sdk:test
docker-compose run --rm crkit-test

# 查看日志
docker logs crkit-runtime
docker-compose logs -f crkit-runtime

# 进入容器
docker exec -it crkit-runtime bash
docker-compose exec crkit-runtime bash

# 停止和清理
docker stop crkit-runtime
docker-compose down
docker-compose down -v --rmi all  # 清理所有资源

# 镜像管理
docker images | grep crkit
docker rmi crkit-sdk:runtime
docker system prune -a  # 清理未使用的资源
```

### 健康检查

```bash
# 检查容器健康状态
docker inspect --format='{{.State.Health.Status}}' crkit-runtime

# 手动执行健康检查
docker-compose exec crkit-runtime test_comprehensive
```

## 🎓 进阶话题

### 1. Docker Swarm集群部署

```bash
# 初始化Swarm
docker swarm init

# 部署Stack
docker stack deploy -c docker-compose.yml crkit

# 扩展服务
docker service scale crkit_crkit-runtime=5
```

### 2. Kubernetes部署

```yaml
# crkit-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crkit-sdk
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crkit
  template:
    metadata:
      labels:
        app: crkit
    spec:
      containers:
      - name: crkit
        image: crkit-sdk:runtime
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: crkit-models
```

### 3. 镜像推送到Registry

```bash
# 标记镜像
docker tag crkit-sdk:runtime your-registry.com/crkit-sdk:runtime

# 推送
docker push your-registry.com/crkit-sdk:runtime

# 在生产环境拉取
docker pull your-registry.com/crkit-sdk:runtime
```

## 📞 获取帮助

- 📧 Email: wlxinchat@gmail.com
- 📚 文档: [BUILD_GUIDE.md](BUILD_GUIDE.md)
- 🐛 问题: 检查 `docker-compose logs`

## 总结

CRKIT SDK的Docker部署提供了：
- ✅ **多阶段构建**: 优化镜像大小
- ✅ **多场景支持**: 生产/开发/测试
- ✅ **安全隔离**: 非root用户运行
- ✅ **易于扩展**: 支持GPU、集群部署
- ✅ **完整文档**: 详细的使用说明

**快速验证**:
```bash
docker-compose build crkit-test && docker-compose run --rm crkit-test
```

**预期结果**: `✅ 所有测试通过!`
