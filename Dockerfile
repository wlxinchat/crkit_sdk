# 多阶段构建 - 基础镜像
FROM ubuntu:22.04 AS base

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# 构建阶段 - 编译SDK（优化层数）
# ==========================================
FROM base AS builder

WORKDIR /tmp

# 合并安装OpenCV和ONNX Runtime，减少层数
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopencv-dev \
    && wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz && \
    cp onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ && \
    cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-1.16.3* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制源码并编译（合并步骤）
WORKDIR /build
COPY . .

RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_EXAMPLES=ON \
        -DENABLE_ONNXRUNTIME=ON && \
    make -j$(nproc) && \
    ctest --output-on-failure

# ==========================================
# 运行阶段 - 最小化镜像
# ==========================================
FROM ubuntu:22.04 AS runtime

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 只安装运行时依赖
RUN apt-get update && apt-get install -y \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制必要文件
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
COPY --from=builder /build/build/libcrkit_core.so /usr/local/lib/
COPY --from=builder /build/build/libcrkit_onnxruntime.so /usr/local/lib/
COPY --from=builder /build/include /usr/local/include/crkit
COPY --from=builder /build/build/examples/* /usr/local/bin/
COPY --from=builder /build/build/test_comprehensive /usr/local/bin/
COPY --from=builder /build/build/test_logger /usr/local/bin/

# 更新动态链接库缓存
RUN ldconfig

# 创建工作目录
WORKDIR /app

# 创建日志目录
RUN mkdir -p /var/log/crkit

# 非root用户运行
RUN useradd -m -u 1000 -s /bin/bash crkit && \
    chown -R crkit:crkit /app /var/log/crkit
USER crkit

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD test_comprehensive || exit 1

# 默认命令
CMD ["bash"]

# ==========================================
# 开发阶段 - 包含开发工具
# ==========================================
FROM builder AS development

# 安装开发工具
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 复制源码
COPY . .

# 开发模式：挂载卷进行实时开发
VOLUME ["/workspace"]

CMD ["bash"]

# ==========================================
# 测试阶段 - 只用于测试
# ==========================================
FROM builder AS test

WORKDIR /build/build

# 运行所有测试
CMD ["ctest", "--output-on-failure", "--verbose"]
