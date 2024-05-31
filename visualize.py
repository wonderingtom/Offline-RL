import random
from pathlib import Path
import hydra
import torch
import dmc
import utils
from video import VideoRecorder

def eval(global_step, agent, env, num_eval_episodes, video_recorder):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step=global_step, eval_mode=True)
            time_step = env.step(action)
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{global_step}.mp4')

    episode_reward = total_reward / episode

    return episode_reward

def set_seed(cfg):
    if cfg.seed is None:
        cfg.seed = random.randint(0, 100000)
    utils.set_seed_everywhere(cfg.seed)

@hydra.main(config_path='config', config_name='video')
def main(cfg):
    work_dir = Path.cwd()
    set_seed(cfg)
    
    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)
    
    # create agent
    agent = hydra.utils.instantiate(cfg.agent, obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape, num_expl_steps=0)
    agent.load(work_dir)
    global_step = 0
    video_recorder = VideoRecorder(work_dir)
    reward = eval(global_step, agent, env, cfg.num_eval_episodes, video_recorder)
    print(reward)

if __name__ == '__main__':
    main()