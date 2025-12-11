#!/usr/bin/env python
"""
本地推理脚本 - 直接加载模型并在环境中进行推理测试
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

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
    print(f"已加载 pi0 策略从: {checkpoint_dir}")
    return agent_dp


def load_sac_agent(checkpoint_path, variant):
    """加载 SAC agent（可选）"""
    import os
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
        train_kwargs = getattr(variant, 'train_kwargs', {})
        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **train_kwargs)

        # 加载检查点
        agent.restore_checkpoint(checkpoint_path)
        print(f"已加载 SAC agent 从: {checkpoint_path}")
        return agent
    except Exception as e:
        print(f"加载 SAC agent 失败: {e}，将使用标准高斯噪声")
        return None


def create_env(env_type, task_description=None):
    """创建环境"""
    if env_type == 'libero':
        import libero.libero.envs.env_wrapper as env_wrapper
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark

        benchmark_name = task_description.split('-')[0]  # e.g., "libero90"
        task_id = int(task_description.split('task')[-1])  # e.g., 57

        benchmark = get_benchmark(benchmark_name)(task_id)
        task_name = benchmark.get_task_names()[0]
        task = benchmark.get_task(task_id)
        task_description_str = task.language
        task_bddl_file = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

        print(f"任务: {task_name}")
        print(f"描述: {task_description_str}")

        env = env_wrapper.ControlEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=128,
            camera_widths=128,
            render_gpu_device_id=0,
        )
        env.reset()
        return env, task_description_str

    elif env_type == 'aloha_cube':
        from envs.aloha_cube import AlohaCubeEnv
        env = AlohaCubeEnv()
        env.reset()
        return env, None

    else:
        raise ValueError(f"不支持的环境类型: {env_type}")


def run_inference(args):
    """运行推理"""
    # 创建 variant 对象
    variant = type('Variant', (), {
        'env': args.env,
        'task_description': args.task_description,
        'add_states': args.add_states,
        'resize_image': args.resize_image,
        'num_cameras': args.num_cameras,
        'seed': args.seed,
        'train_kwargs': {},
    })()

    # 加载模型
    print("正在加载 pi0 策略...")
    agent_dp = load_agent_dp(args.env)

    # 可选：加载 SAC agent
    agent = None
    if args.agent_checkpoint:
        print("正在加载 SAC agent...")
        agent = load_sac_agent(args.agent_checkpoint, variant)

    # 创建环境
    print(f"正在创建环境: {args.env}")
    env, task_description_str = create_env(args.env, args.task_description)
    if task_description_str:
        variant.task_description = task_description_str

    # RNG 用于生成噪声
    rng = jax.random.PRNGKey(args.seed)

    # 运行多个 episode
    success_count = 0
    for episode in range(args.num_episodes):
        print(f"\n========== Episode {episode + 1}/{args.num_episodes} ==========")
        obs = env.reset()
        done = False
        step_count = 0

        while not done and step_count < args.max_steps_per_episode:
            # 将观察转换为 pi0 输入格式
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)

            # 生成噪声
            if agent is not None and args.use_sac_noise:
                # 使用 SAC agent 生成噪声
                curr_image = obs_to_img(obs, variant)
                qpos = obs_to_qpos(obs, variant)

                if args.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jnp.concatenate([actions_noise, noise], axis=0)[None]
            else:
                # 使用标准高斯噪声
                noise = jax.random.normal(rng, (1, 50, 32))
                rng, _ = jax.random.split(rng)

            # 执行推理
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action = np.array(actions[0, args.query_freq - 1])

            # 应用 action magnitude
            action = action * args.action_magnitude

            # 执行动作
            obs, reward, done, info = env.step(action)
            step_count += 1

            if args.render:
                env.render()

            # 打印进度
            if step_count % 10 == 0:
                print(f"Step {step_count}, Reward: {reward:.3f}")

        # 检查是否成功
        success = info.get('success', False) if isinstance(info, dict) else False
        if success:
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS in {step_count} steps")
        else:
            print(f"Episode {episode + 1}: FAILED after {step_count} steps")

    # 打印汇总
    print(f"\n========== 推理汇总 ==========")
    print(f"总 Episodes: {args.num_episodes}")
    print(f"成功: {success_count}")
    print(f"成功率: {success_count / args.num_episodes * 100:.1f}%")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="本地推理测试")
    parser.add_argument("--env", type=str, default="libero", choices=["libero", "aloha_cube"], help="环境类型")
    parser.add_argument("--task_description", type=str, default="libero90-task57", help="任务描述（用于 libero）")
    parser.add_argument("--add_states", action="store_true", help="是否添加状态信息")
    parser.add_argument("--agent_checkpoint", type=str, default=None, help="SAC agent 检查点路径（可选）")
    parser.add_argument("--resize_image", type=int, default=244, help="图像调整大小")
    parser.add_argument("--num_cameras", type=int, default=1, help="相机数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_episodes", type=int, default=20, help="运行的 episode 数量")
    parser.add_argument("--max_steps_per_episode", type=int, default=900, help="每个 episode 的最大步数")
    parser.add_argument("--query_freq", type=int, default=20, help="查询频率")
    parser.add_argument("--action_magnitude", type=float, default=1.0, help="动作幅度")
    parser.add_argument("--use_sac_noise", action="store_true", help="使用 SAC agent 生成噪声（需要提供 agent_checkpoint）")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
