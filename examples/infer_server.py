#!/usr/bin/env python
"""
基于 WebSocket 的推理服务器
参考 examples/train_utils_sim.py 中的 perform_control_eval 函数实现
"""
import asyncio
import argparse
import logging
import traceback
import pathlib
import os

import jax
import numpy as np
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download
from examples.train_utils_sim import obs_to_pi_zero_input, obs_to_img, obs_to_qpos

# 设置 JAX 配置
jax.config.update("jax_default_prng_impl", "rbg")
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

# 防止 tensorflow 使用 GPU
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceServer:
    """基于 WebSocket 的推理服务器，参考 perform_control_eval 的实现"""
    
    def __init__(
        self,
        agent_dp,
        agent=None,
        env_type='libero',
        task_description=None,
        add_states=False,
        query_freq=1,
        host="127.0.0.1",
        port=8000,
    ):
        """
        Args:
            agent_dp: pi0 策略模型
            agent: 可选的 SAC agent，用于生成噪声
            env_type: 环境类型 ('libero' 或 'aloha_cube')
            task_description: 任务描述（用于 libero）
            add_states: 是否添加状态信息
            query_freq: 查询频率
            host: 服务器 IP 地址
            port: 服务器端口
        """
        self._agent_dp = agent_dp
        self._agent = agent
        self._env_type = env_type
        self._task_description = task_description
        self._add_states = add_states
        self._query_freq = query_freq
        self._host = host
        self._port = port
        
        # 创建虚拟 variant 对象用于辅助函数
        self._variant = type('Variant', (), {
            'env': env_type,
            'task_description': task_description,
            'add_states': add_states,
            'resize_image': 0,  # 默认不调整大小，假设客户端已处理
        })()
        
        # RNG 用于生成噪声
        self._rng = jax.random.PRNGKey(42)
        
        logger.info(f"初始化推理服务器: env_type={env_type}, host={host}, port={port}")
        if agent is not None:
            logger.info(f"SAC agent 已加载，将用于生成噪声")
        else:
            logger.info(f"未加载 SAC agent，将使用标准高斯噪声")
    
    def serve_forever(self):
        """启动服务器并永久运行"""
        asyncio.run(self.run())
    
    async def run(self):
        """运行 WebSocket 服务器"""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            # 输出服务器地址信息
            print("=" * 60)
            print(f"推理服务器已启动")
            print(f"WebSocket 地址: ws://{self._host}:{self._port}")
            print(f"IP 地址: {self._host}")
            print(f"端口: {self._port}")
            print("=" * 60)
            logger.info(f"推理服务器启动在 ws://{self._host}:{self._port}")
            await server.serve_forever()
    
    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        """处理 WebSocket 连接"""
        logger.info(f"客户端连接: {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        
        try:
            # 发送元数据
            metadata = {
                "env_type": self._env_type,
                "add_states": self._add_states,
                "query_freq": self._query_freq,
            }
            await websocket.send(packer.pack(metadata))
            
            while True:
                try:
                    # 接收消息
                    message = msgpack_numpy.unpackb(await websocket.recv())
                    
                    # 解析消息
                    method = message.get("method", "infer")
                    
                    if method == "infer":
                        # 执行推理
                        result = await self._infer(message)
                    elif method == "get_info":
                        # 获取服务器信息
                        result = {
                            "env_type": self._env_type,
                            "add_states": self._add_states,
                            "query_freq": self._query_freq,
                            "has_agent": self._agent is not None,
                        }
                    else:
                        raise ValueError(f"未知方法: {method}")
                    
                    # 转换结果为 numpy 数组并发送
                    result = jax.tree.map(lambda x: np.asarray(x).astype(np.float32), result)
                    await websocket.send(packer.pack(result))
                    
                except websockets.ConnectionClosed:
                    logger.info(f"客户端断开连接: {websocket.remote_address}")
                    break
                except Exception as e:
                    error_msg = traceback.format_exc()
                    logger.error(f"处理消息时出错: {error_msg}")
                    try:
                        await websocket.send(packer.pack({"error": error_msg}))
                    except:
                        pass
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="内部服务器错误",
                    )
                    break
                    
        except Exception as e:
            logger.error(f"连接处理错误: {traceback.format_exc()}")
    
    async def _infer(self, message):
        """
        执行推理
        
        消息格式:
        {
            "obs": {...},  # 原始观察数据（需要转换）或已转换的 pi0 输入
            "noise": np.ndarray,  # 可选的噪声，如果不提供则自动生成
            "use_raw_obs": bool,  # 如果为 True，则从原始观察转换，否则直接使用 obs 作为 pi0 输入
        }
        """
        obs = message.get("obs")
        if obs is None:
            raise ValueError("观察数据 'obs' 不能为空")
        
        noise = message.get("noise")
        use_raw_obs = message.get("use_raw_obs", True)
        
        if use_raw_obs:
            # 验证观察数据格式
            if self._env_type == 'libero':
                required_keys = ["agentview_image", "robot0_eye_in_hand_image", 
                                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
                missing_keys = [key for key in required_keys if key not in obs or obs[key] is None]
                if missing_keys:
                    raise ValueError(f"观察数据缺少必需的字段或字段为 None: {missing_keys}")
            elif self._env_type == 'aloha_cube':
                if "pixels" not in obs or obs["pixels"] is None:
                    raise ValueError("观察数据缺少 'pixels' 字段或字段为 None")
                if "top" not in obs["pixels"] or obs["pixels"]["top"] is None:
                    raise ValueError("观察数据缺少 'pixels.top' 字段或字段为 None")
                if "agent_pos" not in obs or obs["agent_pos"] is None:
                    raise ValueError("观察数据缺少 'agent_pos' 字段或字段为 None")
            
            # 从原始观察转换为 pi0 输入格式
            obs_pi_zero = obs_to_pi_zero_input(obs, self._variant)
            # 同时获取用于 agent 的观察
            curr_image = obs_to_img(obs, self._variant)
            qpos = obs_to_qpos(obs, self._variant)
        else:
            # 直接使用提供的观察（假设已经是 pi0 输入格式）
            obs_pi_zero = obs
            curr_image = None
            qpos = None
        
        # 生成噪声（如果需要）
        if noise is None:
            if self._agent is not None and use_raw_obs:
                # 使用 SAC agent 生成噪声
                if self._add_states and qpos is not None:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }
                
                actions_noise = self._agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, self._agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
            else:
                # 使用标准高斯噪声
                noise = self._generate_default_noise()
        
        # 执行推理
        actions = self._agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
        
        return {"actions": actions}
    
    def _generate_default_noise(self):
        """生成默认的高斯噪声"""
        # 默认噪声形状: (1, 50, 32)
        # 这应该根据实际的 action_chunk_shape 调整
        if self._agent is not None:
            chunk_shape = self._agent.action_chunk_shape
            action_dim = chunk_shape[1] if len(chunk_shape) > 1 else 32
        else:
            # 如果没有 agent，使用默认值
            action_dim = 32
        noise = jax.random.normal(self._rng, (1, 50, action_dim))
        self._rng, _ = jax.random.split(self._rng)
        return noise


def load_agent_dp(env_type):
    """加载 pi0 策略"""
    if env_type == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif env_type == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise ValueError(f"不支持的环境类型: {env_type}")
    
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    logger.info(f"已加载 pi0 策略从: {checkpoint_dir}")
    return agent_dp


def load_agent(checkpoint_path, variant):
    """加载 SAC agent（可选）"""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None
    
    try:
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        from jaxrl2.utils.general_utils import add_batch_dim
        from gym.spaces import Dict, Box
        
        # 创建虚拟环境用于初始化 agent
        class DummyEnv:
            def __init__(self, variant):
                self.variant = variant
                self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
                obs_dict = {}
                obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
                if variant.add_states:
                    if variant.env == 'libero':
                        state_dim = 8
                    elif variant.env == 'aloha_cube':
                        state_dim = 14
                    obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
                self.observation_space = Dict(obs_dict)
                self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)
        
        dummy_env = DummyEnv(variant)
        sample_obs = add_batch_dim(dummy_env.observation_space.sample())
        sample_action = add_batch_dim(dummy_env.action_space.sample())
        
        # 创建 agent
        # 从 variant 获取训练参数，如果不存在则使用默认值
        train_kwargs = getattr(variant, 'train_kwargs', {})
        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **train_kwargs)
        
        # 加载检查点
        agent.restore_checkpoint(checkpoint_path)
        logger.info(f"已加载 SAC agent 从: {checkpoint_path}")
        return agent
    except Exception as e:
        logger.warning(f"加载 SAC agent 失败: {e}，将使用标准高斯噪声")
        return None


def main():
    parser = argparse.ArgumentParser(description="启动推理服务器")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器 IP 地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--env", type=str, default="libero", choices=["libero", "aloha_cube"], help="环境类型")
    parser.add_argument("--task_description", type=str, default="libero90-task57", help="任务描述（用于 libero）")
    parser.add_argument("--add_states", action="store_true", help="是否添加状态信息")
    parser.add_argument("--query_freq", type=int, default=1, help="查询频率")
    parser.add_argument("--agent_checkpoint", type=str, default="/home/mxy/robot/rl/dsrl_pi0/logs/DSRL_pi0_Libero/dsrl_pi0_libero_2025_12_05_15_43_59_0000--s-0/checkpoint500000", help="SAC agent 检查点路径（可选）")
    parser.add_argument("--resize_image", type=int, default=64, help="图像调整大小（0 表示不调整）")
    parser.add_argument("--num_cameras", type=int, default=1, help="相机数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 创建 variant 对象
    variant = type('Variant', (), {
        'env': args.env,
        'task_description': args.task_description,
        'add_states': args.add_states,
        'resize_image': args.resize_image,
        'num_cameras': args.num_cameras,
        'seed': args.seed,
        'train_kwargs': {},  # 可以根据需要添加
    })()
    
    # 加载模型
    logger.info("正在加载 pi0 策略...")
    agent_dp = load_agent_dp(args.env)
    
    # 可选：加载 SAC agent
    agent = None
    if args.agent_checkpoint:
        logger.info("正在加载 SAC agent...")
        agent = load_agent(args.agent_checkpoint, variant)
    
    # 创建并启动服务器
    server = InferenceServer(
        agent_dp=agent_dp,
        agent=agent,
        env_type=args.env,
        task_description=args.task_description,
        add_states=args.add_states,
        query_freq=args.query_freq,
        host=args.host,
        port=args.port,
    )
    
    server.serve_forever()


if __name__ == "__main__":
    main()

