#!/usr/bin/env python
"""
本地推理脚本 - 直接加载模型并在环境中进行推理测试
"""
import argparse
import os
import csv
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import imageio
import PIL.Image

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


def create_env(env_type, task_suite=None, task_id=None):
    """创建环境"""
    if env_type == 'libero':
        import libero.libero.envs.env_wrapper as env_wrapper
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark

        # 创建 benchmark 实例（task_order_index 默认为 0）
        benchmark = get_benchmark(task_suite)()
        # 从 benchmark 中获取指定的 task
        task = benchmark.get_task(task_id)
        task_name = task.name
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


def run_single_task_inference(args, task_id, variant, agent_dp, agent, video_base_dir):
    """对单个任务运行推理"""
    # 创建环境
    env, task_description_str = create_env(args.env, args.task_suite, task_id)
    if task_description_str:
        variant.task_description = task_description_str
    
    # 使用基础视频目录，不创建任务子文件夹
    task_video_dir = video_base_dir

    # RNG 用于生成噪声
    rng = jax.random.PRNGKey(args.seed)

    # 运行多个 episode
    success_count = 0
    total_steps = 0
    env_max_reward = variant.env_max_reward
    
    for episode in range(args.num_episodes):
        print(f"\n========== Task {task_id}, Episode {episode + 1}/{args.num_episodes} ==========")
        obs = env.reset()
        done = False
        step_count = 0
        last_reward = 0  # 记录最后一个 reward

        actions = None  # 存储当前的 action chunk
        image_list = []  # 存储图像用于视频
        
        while not done and step_count < args.max_steps_per_episode:
            # 每 query_freq 步查询一次新的 action chunk
            if step_count % args.query_freq == 0:
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

                # 执行推理，获取 action chunk
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
                # 如果 actions 有 batch 维度，去掉它
                if actions.ndim == 3:
                    actions = actions[0]  # 去掉 batch 维度，形状变为 (action_horizon, action_dim)
            
            # 从 action chunk 中选择当前步的动作
            action_idx = step_count % args.query_freq
            # 确保不超出 action chunk 的长度
            if action_idx >= len(actions):
                action_idx = len(actions) - 1
            action = np.array(actions[action_idx])

            # 应用 action magnitude
            action = action * args.action_magnitude

            # 执行动作
            obs, reward, done, info = env.step(action)
            step_count += 1
            last_reward = reward  # 更新最后一个 reward

            # 收集图像用于视频
            if task_video_dir:
                if args.env == 'libero' and 'agentview_image' in obs:
                    # libero 环境：翻转图像（因为观察是上下颠倒的）
                    img = obs['agentview_image'][::-1, ::-1]
                    image_list.append(img)
                elif args.env == 'aloha_cube' and 'pixels' in obs:
                    # aloha 环境
                    img = obs['pixels']['top']
                    image_list.append(img)

            # 打印进度
            if step_count % 10 == 0:
                print(f"Step {step_count}, Reward: {reward:.3f}")

        # 检查是否成功：使用环境的 check_success() 方法（最准确）
        # 对于 libero，使用 env.check_success()；对于 aloha_cube，使用 reward 判断
        if args.env == 'libero':
            is_success = env.check_success()
        else:
            # aloha_cube 使用 reward 判断
            is_success = (last_reward == env_max_reward)
        
        if is_success:
            success_count += 1
            print(f"Task {task_id}, Episode {episode + 1}: SUCCESS in {step_count} steps (reward={last_reward})")
        else:
            print(f"Task {task_id}, Episode {episode + 1}: FAILED after {step_count} steps (reward={last_reward})")
        
        total_steps += step_count
        
        # 保存视频
        if task_video_dir and len(image_list) > 0:
            # 生成视频文件名：rollout_task00_ep000_failure_910steps_30fps.mp4
            task_id_str = f"task{task_id:02d}" if args.env == 'libero' else "task00"
            ep_str = f"ep{episode:03d}"
            success_str = "success" if is_success else "failure"
            steps_str = f"{step_count}steps"
            fps_str = f"{args.video_fps}fps"
            video_filename = f"rollout_{task_id_str}_{ep_str}_{success_str}_{steps_str}_{fps_str}.mp4"
            video_path = os.path.join(task_video_dir, video_filename)
            
            # 调整图像大小为 224x224 并确保是 uint8 格式
            images = []
            for img in image_list:
                img_array = np.asarray(img).astype(np.uint8)
                # 如果图像尺寸不是 224x224，则调整大小
                if img_array.shape[:2] != (224, 224):
                    img_pil = PIL.Image.fromarray(img_array)
                    img_pil = img_pil.resize((224, 224), PIL.Image.Resampling.LANCZOS)
                    img_array = np.array(img_pil)
                images.append(img_array)
            imageio.mimwrite(video_path, images, fps=args.video_fps)
            print(f"已保存视频: {video_path} (分辨率: 224x224)")
    
    env.close()
    
    # 返回统计信息
    return {
        'task_id': task_id,
        'num_episodes': args.num_episodes,
        'num_successes': success_count,
        'success_rate': success_count / args.num_episodes if args.num_episodes > 0 else 0.0,
        'task_description': task_description_str,
        'average_steps': total_steps / args.num_episodes if args.num_episodes > 0 else 0.0,
    }


def run_inference(args):
    """运行推理"""
    # 创建 variant 对象
    # 注意：train_kwargs 必须与训练时使用的参数完全一致
    train_kwargs = {
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'temp_lr': 3e-4,
        'hidden_dims': (128, 128, 128),
        'cnn_features': (32, 32, 32, 32),
        'cnn_strides': (2, 1, 1, 1),
        'cnn_padding': 'VALID',
        'latent_dim': 50,
        'discount': 0.999,
        'tau': 0.005,
        'critic_reduction': 'mean',
        'dropout_rate': 0.0,
        'aug_next': 1,
        'use_bottleneck': True,
        'encoder_type': 'small',  # 重要：必须与训练时一致
        'encoder_norm': 'group',
        'use_spatial_softmax': True,
        'softmax_temperature': -1,
        'target_entropy': 'auto',
        'num_qs': 10,
        'action_magnitude': 1.0,
        'num_cameras': args.num_cameras,
    }
    
    # 设置环境的最大奖励值（用于判断成功）
    if args.env == 'libero':
        env_max_reward = 1
    elif args.env == 'aloha_cube':
        env_max_reward = 4
    else:
        env_max_reward = 1
    
    variant = type('Variant', (), {
        'env': args.env,
        'add_states': args.add_states,
        'resize_image': args.resize_image,
        'num_cameras': args.num_cameras,
        'seed': args.seed,
        'train_kwargs': train_kwargs,
        'env_max_reward': env_max_reward,
    })()

    # 加载模型
    print("正在加载 pi0 策略...")
    agent_dp = load_agent_dp(args.env)

    # 可选：加载 SAC agent
    agent = None
    if args.agent_checkpoint:
        print("正在加载 SAC agent...")
        agent = load_sac_agent(args.agent_checkpoint, variant)

    # 创建视频保存目录
    if args.video_dir:
        video_base_dir = args.video_dir
    else:
        # 自动生成路径：logs/libero/时间戳_suitename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_name = args.task_suite if args.env == 'libero' else args.env
        video_base_dir = os.path.join("logs", args.env, f"{timestamp}_{suite_name}")
    
    os.makedirs(video_base_dir, exist_ok=True)
    # 视频实际文件放在 video_base_dir/videos 下，统计信息放在 video_base_dir
    video_dir = os.path.join(video_base_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    print(f"视频将保存到: {video_dir}")

    # 确定要运行的任务列表
    if args.task_id == "all" and args.env == 'libero':
        # 获取该 suite 下的所有任务 ID
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(args.task_suite)()
        task_ids = list(range(benchmark.get_num_tasks()))
        print(f"\n将对 {args.task_suite} suite 下的 {len(task_ids)} 个任务进行推理")
    else:
        # 单个任务，将字符串转换为整数
        task_ids = [int(args.task_id)]
    
    # 存储所有任务的统计信息
    all_task_stats = []
    
    # 对每个任务运行推理
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"开始推理任务 {task_id}/{task_ids[-1]}")
        print(f"{'='*60}")
        
        stats = run_single_task_inference(args, task_id, variant, agent_dp, agent, video_dir)
        all_task_stats.append(stats)
        
        print(f"\n任务 {task_id} 完成:")
        print(f"  成功率: {stats['success_rate']*100:.1f}% ({stats['num_successes']}/{stats['num_episodes']})")
        print(f"  平均步数: {stats['average_steps']:.1f}")
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print("任务推理汇总")
    print(f"{'='*80}")
    print(f"{'Task ID':<10} {'Episodes':<10} {'Successes':<12} {'Success Rate':<15} {'Avg Steps':<12} {'Task Description':<50}")
    print(f"{'-'*80}")
    
    total_episodes = 0
    total_successes = 0
    total_steps = 0
    
    for stats in all_task_stats:
        task_desc = stats['task_description'] or "N/A"
        if len(task_desc) > 50:
            task_desc = task_desc[:47] + "..."
        print(f"{stats['task_id']:<10} {stats['num_episodes']:<10} {stats['num_successes']:<12} "
              f"{stats['success_rate']*100:>6.1f}%{'':<8} {stats['average_steps']:>10.1f}  {task_desc:<50}")
        total_episodes += stats['num_episodes']
        total_successes += stats['num_successes']
        total_steps += stats['average_steps'] * stats['num_episodes']
    
    print(f"{'-'*80}")
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    overall_avg_steps = total_steps / total_episodes if total_episodes > 0 else 0.0
    print(f"{'总计':<10} {total_episodes:<10} {total_successes:<12} "
          f"{overall_success_rate*100:>6.1f}%{'':<8} {overall_avg_steps:>10.1f}")
    print(f"{'='*80}")
    
    # 保存统计信息到 CSV 文件
    # 固定文件名：tasks_summary.csv
    csv_filename = "tasks_summary.csv"
    csv_path = os.path.join(video_base_dir, csv_filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['task_id', 'num_episodes', 'num_successes', 'success_rate', 'task_description', 'average_steps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入每个任务的统计信息
        for stats in all_task_stats:
            writer.writerow({
                'task_id': stats['task_id'],
                'num_episodes': stats['num_episodes'],
                'num_successes': stats['num_successes'],
                'success_rate': f"{stats['success_rate']:.4f}",
                'task_description': stats['task_description'] or "N/A",
                'average_steps': f"{stats['average_steps']:.2f}"
            })
        
        # 写入总计行
        writer.writerow({
            'task_id': '总计',
            'num_episodes': total_episodes,
            'num_successes': total_successes,
            'success_rate': f"{overall_success_rate:.4f}",
            'task_description': '',
            'average_steps': f"{overall_avg_steps:.2f}"
        })
    
    print(f"\n统计信息已保存到: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="本地推理测试")
    parser.add_argument("--env", type=str, default="libero", choices=["libero", "aloha_cube"], help="环境类型")
    parser.add_argument("--task_suite", type=str, default="libero_90", help="任务套件（用于 libero）:  libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    parser.add_argument("--task_id", type=str, default="57", help="任务 ID（用于 libero），可以是数字或 'all'（对所有任务进行推理）")
    parser.add_argument("--add_states", action="store_true", help="是否添加状态信息")
    parser.add_argument("--agent_checkpoint", type=str, default=None, help="SAC agent 检查点路径（可选）")
    parser.add_argument("--resize_image", type=int, default=64, help="图像调整大小（必须与训练时一致，libero 默认 64）")
    parser.add_argument("--num_cameras", type=int, default=1, help="相机数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_episodes", type=int, default=20, help="运行的 episode 数量")
    parser.add_argument("--max_steps_per_episode", type=int, default=900, help="每个 episode 的最大步数")
    parser.add_argument("--query_freq", type=int, default=20, help="查询频率")
    parser.add_argument("--action_magnitude", type=float, default=1.0, help="动作幅度")
    parser.add_argument("--use_sac_noise", action="store_true", help="使用 SAC agent 生成噪声（需要提供 agent_checkpoint）")
    parser.add_argument("--video_dir", type=str, default=None, help="视频保存目录（如果指定，将保存每个 episode 的视频）")
    parser.add_argument("--video_fps", type=int, default=30, help="视频帧率（fps）")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
