import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.simlator.Simulator import *
from learner.common.PPONet import *
import matplotlib.pyplot as plt

from src.common.Parameters import *

class PPO:
    data = []
    gamma = 0.99
    lmbda = 0.95
    eps_clip = 0.1
    k_epoch = 3
    T_horizon = 30

    @classmethod
    def put_data(cls, transition):
        cls.data.append(transition)

    @classmethod
    def make_batch(cls):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in cls.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        cls.data = []

        return s, a, r, s_prime, done_mask, prob_a
    @classmethod
    def train_net(cls):
        s, a, r, s_prime, done_mask, prob_a = cls.make_batch()

        td_target = r + cls.gamma * cls.model.v(s_prime) * done_mask
        delta = td_target - cls.model.v(s)
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = cls.gamma * cls.lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)

        for i in range(cls.k_epoch):

            pi, log_pi = cls.model.pi_log_prob(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - cls.eps_clip, 1 + cls.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(cls.model.v(s), td_target.detach())

            cls.optimizer.zero_grad()
            loss.mean().backward()
            cls.optimizer.step()

    @classmethod
    def main(cls):
        env = Simulator
        cls.model = PPONet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=Hyperparameters.learning_rate)
        score = 0.0
        print_interval = 1

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}"  # 디렉토리 경로를 지정합니다.

        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)  # 경로 없을 시 생성

        mean_util_list = []
        max_util_list = []
        min_util_list = []
        mean_list = []
        max_list = []
        min_list = []

        for n_epi in range(10000):
            reward_list = []
            util_list = []
            for dataid in Parameters.datasetId:
                score = 0
                s= env.reset(dataid)
                done = False
                while not done:
                    prob = cls.model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done = env.step(a)

                    cls.put_data((s, a, r , s_prime, prob[a].item(), done))
                    s = s_prime

                    score += r
                    if done:
                        break
                reward_list, util_list = cls.script_performance(env, n_epi, score,dataid, False, reward_list, util_list)
                # 네트워크 학습
            cls.train_net()

            cls.dataset_total_script(reward_list,util_list, n_epi)
            mean_util_list.append(sum(util_list)/len(util_list))
            max_util_list.append(max(util_list))
            min_util_list.append(min(util_list))
            mean_list.append(sum(reward_list)/len(reward_list))
            max_list.append(max(reward_list))
            min_list.append(min(reward_list))

        cls.draw_plot(mean_util_list,max_util_list,min_util_list,mean_list,max_list,min_list)

    @classmethod
    def script_performance(cls, env, n_epi, score, dataid,type, reward_list, util_list):
        Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_over_time, rtf = env.performance_measure()

        output_string = "--------------------------------------------------\n" + \
                        f"flow time: {Flow_time}, util : {util:.3f}, makespan : {makespan}, rtf: {rtf}\n" + \
                        f"Tardiness: {Tardiness_time}, Lateness : {Lateness_time}, T_max : {T_max}\n" + \
                        f"q_true_op: {q_time_true}, q_false_op : {q_time_false}, q_true_job : {q_job_t}, q_false_job : {q_job_f}, q_over_time : {q_over_time}\n" + \
                        f"n_episode: {n_epi}, score : {score:.1f}, dataset : {dataid}"
        #print(output_string)
        if Parameters.log_on:
            logging.info(f'performance :{output_string}')
        reward_list.append(score)
        util_list.append(util)

        return reward_list, util_list
    @classmethod
    def dataset_total_script(cls, reward_list,util_list, episode):
        output_string = "------------------------------------------\n" +\
            f"max_reward: {max(reward_list)}, min_reward :{min(reward_list)} mean : {sum(reward_list)/len(reward_list)}\n " +\
            f"max_util: {max(util_list)}, min_util :{min(util_list)} mean : {sum(util_list)/len(util_list)}\n " +\
            f"total_data_set : {len(util_list)}, episode : {episode}\n"+\
            "--------------------------------------------------------"
        print(output_string)
        if Parameters.log_on:
            logging.info(f'performance_TOTAL :{output_string}')


    @classmethod
    def draw_plot(cls, mean_util_list,max_util_list,min_util_list,mean_list,max_list,min_list):
        x = [i for i in range(len(mean_list))]
        plt.plot()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        axes[0,0].plot(x, max_list)
        axes[0,1].plot(x, min_list)
        axes[0,2].plot(x, mean_list)
        axes[1,0].plot(x, max_util_list)
        axes[1,1].plot(x, min_util_list)
        axes[1,2].plot(x, mean_util_list)

        plt.show()