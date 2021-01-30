import argparse
import numpy as np
import torch
import time
import pickle
import os
import matplotlib.pyplot as plt

from attacks.action import softmax_attack, random_attack, guassian_attack
from trainer.maddpg.maddpg import MADDPGAgentTrainer
from trainer.maddpg.maddpg_infer import MADDPGInferAgentTrainer
from trainer.maddpg.maddpg_merge import MADDPGMergeAgentTrainer
from utils.common import save_state, restore_state, latest_checkpoint, generate_checkpoint_path, get_device

# torch.manual_seed(0)

def mkdir(dirs):
    for directory in dirs:
        dirs = directory.split('/')
        _dirs = '.'
        for _dir in dirs:
            _dirs = os.path.join(_dirs, _dir)
            if not os.path.exists(_dirs):
                os.mkdir(_dirs)

def parse_args():
    parser = argparse.ArgumentParser(
        "Reinforce Learning for Multi-Agent Environments")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--policy", type=str, default="maddpg", help="policy for agents")
    parser.add_argument("--num-agents", type=int, default=3, help="number of agents")
    
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--grad-norm-clip", type=float, default=0.5, help="max norm for clip gradients")
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu')
    parser.add_argument('--buffer-size', type=int, default=1e6, help='size of replay buffer')
    parser.add_argument('--depth', type=int, default=3, help='depth of MLP')
    parser.add_argument('--update-gap', type=int, default=100, help='amount of samples for online fashion')
    parser.add_argument('--beta', type=float, default=1e-2, help='factor for online fashion')
    
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./save_dir/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--summary-dir", type=str, default="", help="directory in which the summary of tranining process should be saved")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument('--run_index', type=int, default=0, help="index of which run")
    
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario()
    from scenarios.spread import Scenario
    # from scenarios.spread_policy_reg import Scenario
    scenario = Scenario()
    
    # create world
    world = scenario.make_world()

    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)

    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    if arglist.policy == 'maddpg':
        trainer = MADDPGAgentTrainer
    elif arglist.policy == 'maddpg_pi':
        trainer = MADDPGInferAgentTrainer
    elif arglist.policy == 'maddpg_si':
        trainer = MADDPGMergeAgentTrainer
    else:
        trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainer_instance = train()
        trainers.append(trainer_instance)
    for i in range(num_adversaries, env.n):
        trainer_instance = trainer("agent_%d" % i, obs_shape_n, env.action_space,
                                   i, arglist, local_q=(arglist.policy == 'ddpg'))
        trainers.append(trainer_instance)
    return trainers


def run(arglist):
    # prepare save dir
    # print(arglist.save_dir)
    
    arglist.device = get_device(arglist.use_gpu, arglist.gpu_id)

    # save params of arglist
    arglist_dict = vars(arglist)
    with open(os.path.join(arglist.save_dir, 'params.txt'), 'w', encoding='utf-8') as f:
        keys = list(arglist_dict.keys())
        keys.sort()
        for k in keys:
            f.write(k + ':\n\t' + str(arglist_dict[k]) + '\n')


    # Create Environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)

    # create agent trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    # num_adversaries = min(env.n, arglist.num_adversaries)
    num_adversaries = 0
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
    print('Policy:{}'.format(arglist.policy))

    # Prepare necessary data recorders
    mean_episode_rewards = []
    episode_rewards = [0.0]                         # sum of rewards for all agents
    final_ep_rewards = []                           # sum of rewards for training curve
    final_ep_ag_rewards = []                        # agent rewards for training curve
    agent_rewards = [[0.0] for _ in range(env.n)]   # individual agent reward
    agent_info = [[[]]]                             # placeholder for benchmarking info
    

    # Run experiments
    print('Start iterations...')
    obs_n = env.reset()
    episode_step = 0    # amount of steps inside one episode
    num_episodes = 0    # amount of episodes
    train_step = 0      # amount of training steps
    t_start = time.time()

    # Load previous results if necessary
    if arglist.load_dir == '':
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:
        print('Loading previous states...')
        path, latest_index = latest_checkpoint(arglist.load_dir)
        restore_state(trainers, path, map_location=arglist.device)
        num_episodes = latest_index
    if arglist.display or arglist.benchmark:
        for agent in trainers:
            agent.eval()
    else:
        for agent in trainers:
            agent.train()
            # agent.sync_target_nets()

    while True:
        # Get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

        # Evolution of the environment
        new_obs_n, reward_n, done_n, info_n = env.step(action_n)

        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # Collect experience for replay buffer
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], reward_n[i],
                             new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, reward in enumerate(reward_n):
            episode_rewards[-1] += reward
            agent_rewards[i][-1] += reward

        if done or terminal:
            num_episodes += 1
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0.0)
            for a in agent_rewards:
                a.append(0.0)
            agent_info.append([[]])

        # Increment global train step counter
        train_step += 1

        # # for benchmarking learned policies
        # if arglist.benchmark:
        #     for i, info in enumerate(info_n):
        #         agent_info[-1][i].append(info_n['n'])
        #     if train_step > arglist.benchmark_iters and (done or terminal):
        #         file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
        #         print('Finished benchmarking, now saving...')
        #         with open(file_name, 'wb') as f:
        #             pickle.dump(agent_info[:-1],f)
        #         break3, q-value: -115.43821716308594, p-loss:115.20866394042969, time: 134.907
        #     continue

        # For displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # Update parameters of all trainers if not in display or benchmark mode
        info = None
        for agent in trainers:
            agent.preupdate()   # reset sample index of replay buffer
        for agent in trainers:
            info = agent.update(trainers, train_step)

        # Save mode, display training output
        if terminal and ( num_episodes % arglist.save_rate == 0):
            path = generate_checkpoint_path(arglist.save_dir, num_episodes)
            save_state(trainers, path)

            _steps = train_step
            _episodes = num_episodes
            _mean_episode_reward = np.mean(episode_rewards[-arglist.save_rate:])
            mean_episode_rewards.append(_mean_episode_reward)
            q_value = info['q-value']
            p_loss = info['p_loss']
            _time = round(time.time() - t_start, 3)
            if num_adversaries == 0:
                print("steps: {}, epsiodes: {}, mean episode reward: {:.4f}, q-value: {:.4f}, p-loss:{}, time: {}".format(
                    _steps, _episodes, _mean_episode_reward, q_value, p_loss, _time))
            else:
                _agent_episode_reward = [np.mean(reward[-arglist.save_rate:]) for reward in agent_rewards]
                print("steps: {}, epsiodes: {}, mean episode reward: {}, agent episode reward: {}, q-value: {}, p-loss:{}, time: {}".format(
                    _steps, _episodes, _mean_episode_reward,_agent_episode_reward, q_value, p_loss, _time
                ))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(_mean_episode_reward)
            for reward in agent_rewards:
                final_ep_ag_rewards.append(np.mean(reward[-arglist.save_rate:]))

            # Save final episode reward for plotting training curve later
            if num_episodes >= arglist.num_episodes:
                np.savetxt(os.path.join(arglist.save_dir, 'episode_reward.txt'), episode_rewards[arglist.save_rate:-1], fmt='%.1f')
                plt.plot(list(range(1, len(episode_rewards) - arglist.save_rate)), episode_rewards[arglist.save_rate:-1])
                plt.savefig(os.path.join(arglist.save_dir, 'episode_reward.png'))
                np.savetxt(os.path.join(arglist.save_dir, 'mean_episode_rewards.txt'), mean_episode_rewards, fmt='%.1f')
                plt.clf()
                plt.plot(list(range(1, len(mean_episode_rewards) + 1)), mean_episode_rewards)
                plt.savefig(os.path.join(arglist.save_dir, 'mean_episode_rewards.png'))
                # reward_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                # with open(reward_file_name, 'wb') as f:
                #     pickle.dump(final_ep_rewards, f)
                
                # ag_rewards_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                # with open(ag_rewards_file_name, 'wb') as f:
                #     pickle.dump(final_ep_ag_rewards, f)
                print('...Finished tottal of {} episodes'.format(len(episode_rewards)))
                break


if __name__ == "__main__":
    for i in range(5):
        np.random.seed(seed=i)
        arglist = parse_args()
        arglist.run_index = i
        save_dir = './save_dir/{}/agent_{}/run_{}/'.format(arglist.policy, arglist.num_agents, arglist.run_index)
        mkdir([save_dir,])
        arglist.save_dir = save_dir
        run(arglist)
