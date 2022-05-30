import grpc
import TH_pb2
import TH_pb2_grpc
import time
import torch
import numpy as np
from model import ActorCritic
from ppo import PPO
from storage import RolloutStorage
import matplotlib.pyplot as plt


def run():
    torch.set_num_threads(2)

    lr = 0.0003
    clip_param = 0.1
    value_loss_coef = 0.5
    num_steps = 10000
    epoch = 8
    obs_size, hid_size, num_outputs = 91, 128, 4
    gamma = 0.99
    gae_lambda = 0.99
    model_name = 'avoid.pt'
    try:
        actor_critic = torch.load(model_name, map_location='cpu')
    except Exception as e:
        print(e)
        actor_critic = ActorCritic(obs_size, hid_size, num_outputs)
    agent = PPO(
        actor_critic,
        clip_param,
        epoch,
        value_loss_coef,
        lr=lr,
        eps=1e-5,
        max_grad_norm=0.5)
    rollouts = RolloutStorage(num_steps, obs_size, 3)
    env_count = 0
    reward_sum = 0.0
    epi_rewards = []
    epi_rewards_group = []
    epi_s = []

    with grpc.insecure_channel('localhost:30051') as channel:
        stub = TH_pb2_grpc.CommunicatorStub(channel)
        response = stub.GetInfo(TH_pb2.Empty())
        obs = torch.FloatTensor(response.obs_list).view([response.group_len, response.obs_len])
        rollouts.obs[0].copy_(obs)
        for j in range(1000):
            print('epoch : ', j)
            for step in range(num_steps):
                with torch.no_grad():
                    action, action_log_prob, value = actor_critic.act(rollouts.obs[step])
                stub.SendAct(TH_pb2.GroupAct(action_list=action))
                # 만약 동기화 문제(데이터 손실 등)가 있다면 살짝 지연을 시키자. while 문 안으로 덜 들어갈 것이다.
                # time.sleep(0.01)
                response = stub.GetInfo(TH_pb2.Empty())
                group_len = response.group_len
                while group_len == 0:
                    response = stub.GetInfo(TH_pb2.Empty())
                    group_len = response.group_len
                obs_len, obs_list, reward_list, done_list = response.obs_len, response.obs_list, response.reward_list, response.done_list
                reward_sum += np.mean(reward_list)
                env_count += 1
                if sum(done_list) == 0:
                    epi_rewards.append(reward_sum)
                    if len(epi_rewards) >= 100:
                        epi_rewards_group.append(np.mean(epi_rewards))
                        epi_s.append(len(epi_s) + 1)
                        epi_rewards = []
                    print(reward_sum)
                    env_count = 0
                    reward_sum = 0.0

                rollouts.insert(torch.FloatTensor(obs_list).view([group_len, obs_len]), action, action_log_prob, value,
                                torch.FloatTensor(reward_list).unsqueeze(1), torch.FloatTensor(done_list).unsqueeze(1))

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
            rollouts.compute_returns(next_value, gamma, gae_lambda)
            agent.update(rollouts)
            rollouts.after_update()
            torch.save(actor_critic, model_name)

        # plt.scatter(epi_s, epi_rewards_group, s=0.1)
        plt.plot(epi_s, epi_rewards_group, linewidth=0.5)
        # plt.show()
        plt.savefig('result.png')


if __name__ == '__main__':
    run()
