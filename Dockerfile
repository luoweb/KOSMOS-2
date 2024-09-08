# FROM python:3.10 as builder
# # FROM python:3.11 as builder
# # FROM python:3.11-slim as builder
# # FROM registry.cn-hangzhou.aliyuncs.com/roweb/chatapp:base-20231020 as builder

# # 指定构建过程中的工作目录
# WORKDIR /app

# # 将当前目录（dockerfile所在目录）下所有文件都拷贝到工作目录下（.dockerignore中文件除外）
# COPY . /app/

FROM python:3.10
# FROM python:3.11
# FROM python:3.11-slim
# 引入存了镜像加速构建
# FROM ccr.ccs.tencentyun.com/tcb-100032372308-hrru/ca-lhapuwki_chatapp:chatapp-012-20230802205629
# FROM registry.cn-hangzhou.aliyuncs.com/roweb/chatapp:base-20231020

# 指定运行时的工作目录
WORKDIR /app

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y python3-pip && rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir pip --upgrade && \
    pip3 install -q git+https://github.com/huggingface/transformers.git accelerate bitsandbytes && \
    # pip3 install modelscope[multi-modal] && \
    pip3 install -r requirements.txt


# 将构建产物/app/main拷贝到运行时的工作目录中
# COPY --from=builder /app/main /app/index.html /app/
# COPY --from=builder /app/requirements.txt /app/requirements.txt

# RUN  pip3 install --no-cache-dir -r requirements.txt
# ENV OPENSSL_CONF /app/libs/openssl/openssl.cnf

# COPY --from=builder /app/ /app/

EXPOSE 5000
EXPOSE 8080
EXPOSE 9090
 
# CMD ["python","main.py"]
# CMD ["uvicorn","main:app","--host", "0.0.0.0", "--port", "8000"]