#!/bin/bash
proj_name=DSRL_pi0_Infer_Libero
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=~/.cache/openpi
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 设置推理服务器参数
HOST="127.0.0.1"
PORT=8000
ENV="libero"
TASK_DESCRIPTION="libero90-task57"
QUERY_FREQ=20

# SAC agent 检查点路径（可选，如果不需要 SAC noise 可以设为空）
AGENT_CHECKPOINT="/home/mxy/robot/rl/dsrl_pi0/logs/DSRL_pi0_Libero/dsrl_pi0_libero_2025_12_05_15_43_59_0000--s-0/checkpoint500000"

pip install mujoco==3.3.1

echo "=================================================="
echo "启动 LIBERO 推理服务器"
echo "环境: ${ENV}"
echo "任务: ${TASK_DESCRIPTION}"
echo "服务器地址: ws://${HOST}:${PORT}"
echo "=================================================="

python3 examples/inference_server.py \
--host ${HOST} \
--port ${PORT} \
--env ${ENV} \
--task_description ${TASK_DESCRIPTION} \
--add_states \
--query_freq ${QUERY_FREQ} \
--agent_checkpoint ${AGENT_CHECKPOINT} \
--resize_image 64 \
--num_cameras 1 \
--seed 42
