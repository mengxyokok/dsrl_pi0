#!/bin/bash
proj_name=DSRL_pi0_Libero_Infer_Local
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=~/.cache/openpi
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 推理参数
ENV="libero"
TASK_SUITE="libero_10"  # task suite: libero_spatial, libero_object, libero_goal, libero_10, libero_90
TASK_ID=all             # task ID: 0-89 for libero90, 0-9 for libero10, etc. 或者使用 "all" 对所有任务进行推理
QUERY_FREQ=20
ACTION_MAGNITUDE=1.0
NUM_EPISODES=2
MAX_STEPS_PER_EPISODE=900
VIDEO_DIR=""           # 视频保存目录（如果为空，将自动生成路径）
VIDEO_FPS=30           # 视频帧率

# SAC agent 检查点路径（可选，如果不需要 SAC noise 可以留空或注释掉）
# AGENT_CHECKPOINT="/home/mxy/robot/rl/dsrl_pi0/logs/DSRL_pi0_Libero/dsrl_pi0_libero_2025_12_05_15_43_59_0000--s-0/checkpoint50000"

pip install mujoco==3.3.1


echo "=================================================="
echo "运行 LIBERO 本地推理测试"
echo "环境: ${ENV}"
echo "Task Suite: ${TASK_SUITE}"
echo "Task ID: ${TASK_ID}"
echo "Episodes: ${NUM_EPISODES}"
if [ -n "$VIDEO_DIR" ]; then
    echo "视频保存目录: ${VIDEO_DIR}"
else
    echo "视频保存目录: 自动生成"
fi
echo "视频帧率: ${VIDEO_FPS}"
echo "=================================================="

# 如果有 SAC checkpoint 且想使用 SAC noise，添加 --use_sac_noise 标志
if [ -n "$AGENT_CHECKPOINT" ]; then
    python3 examples/infer_local.py \
    --env ${ENV} \
    --task_suite ${TASK_SUITE} \
    --task_id ${TASK_ID} \
    --add_states \
    --agent_checkpoint ${AGENT_CHECKPOINT} \
    --resize_image 64 \
    --num_cameras 1 \
    --seed 42 \
    --num_episodes ${NUM_EPISODES} \
    --max_steps_per_episode ${MAX_STEPS_PER_EPISODE} \
    --query_freq ${QUERY_FREQ} \
    --action_magnitude ${ACTION_MAGNITUDE} \
    --video_fps ${VIDEO_FPS} \
    ${VIDEO_DIR:+--video_dir ${VIDEO_DIR}} \
    --use_sac_noise
else
    python3 examples/infer_local.py \
    --env ${ENV} \
    --task_suite ${TASK_SUITE} \
    --task_id ${TASK_ID} \
    --add_states \
    --resize_image 64 \
    --num_cameras 1 \
    --seed 42 \
    --num_episodes ${NUM_EPISODES} \
    --max_steps_per_episode ${MAX_STEPS_PER_EPISODE} \
    --query_freq ${QUERY_FREQ} \
    --action_magnitude ${ACTION_MAGNITUDE} \
    --video_fps ${VIDEO_FPS} \
    ${VIDEO_DIR:+--video_dir ${VIDEO_DIR}}
fi
