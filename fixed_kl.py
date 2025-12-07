import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.distributions

# hyperparameters
HYPERPARAMS = {
    'env_id': 'HalfCheetah-v5',
    'horizon': 2048,
    'adam_stepsize': 3e-4,
    'n_epochs': 10,
    'minibatch_size': 64,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'total_timesteps': 1_000_000,

    'beta': 10.0,  # fixed KL Penalty 계수
}


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_ppo(seed):
    print(f"--- Seed {seed}로 Fixed KL Penalty 학습 시작 (Beta={HYPERPARAMS['beta']}) ---")
    set_seed(seed)
    env = gym.make(HYPERPARAMS['env_id'])
    obs, _ = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('device = ',device)

    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    optimizer_policy = optim.Adam(policy.parameters(), lr=HYPERPARAMS['adam_stepsize'])
    optimizer_value = optim.Adam(value_net.parameters(), lr=HYPERPARAMS['adam_stepsize'])

    global_step = 0
    episode_rewards = []
    current_episode_reward = 0.0
    num_updates = HYPERPARAMS['total_timesteps'] // HYPERPARAMS['horizon']
    beta = HYPERPARAMS['beta']
    log_rewards_per_update = []

    for update in range(1, num_updates + 1):
        buffer_obs, buffer_act, buffer_logprob, buffer_val, buffer_done, buffer_rew = [], [], [], [], [], []
        buffer_mu_old, buffer_std_old = [], []

        for _ in range(HYPERPARAMS['horizon']):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
                dist = policy(obs_tensor)

                mu_old = dist.mean.cpu().numpy().flatten()
                std_old = dist.stddev.cpu().numpy().flatten()

                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=1)
                val = value_net(obs_tensor)

            action_np = action.cpu().numpy().flatten()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer_obs.append(obs)
            buffer_act.append(action_np)
            buffer_logprob.append(log_prob.item())
            buffer_val.append(val.item())
            buffer_done.append(done)
            buffer_rew.append(reward)
            buffer_mu_old.append(mu_old)
            buffer_std_old.append(std_old)

            obs = next_obs
            global_step += 1
            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
            next_val = value_net(next_obs_tensor).item()

        advantages = np.zeros(HYPERPARAMS['horizon'])
        last_gae_lam = 0

        # GAE 계산
        for t in reversed(range(HYPERPARAMS['horizon'])):
            if t == HYPERPARAMS['horizon'] - 1:
                next_non_terminal = 1.0 - (terminated or truncated)
                next_value = next_val
            else:
                next_non_terminal = 1.0 - buffer_done[t]
                next_value = buffer_val[t + 1]

            delta = buffer_rew[t] + HYPERPARAMS['gamma'] * next_value * next_non_terminal - buffer_val[t]
            last_gae_lam = delta + HYPERPARAMS['gamma'] * HYPERPARAMS['gae_lambda'] * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + np.array(buffer_val)

        b_obs = torch.FloatTensor(np.array(buffer_obs)).to(device)
        b_act = torch.FloatTensor(np.array(buffer_act)).to(device)
        b_logprob = torch.FloatTensor(np.array(buffer_logprob)).to(device)
        b_adv = torch.FloatTensor(advantages).to(device)
        b_ret = torch.FloatTensor(returns).to(device)
        b_mu_old = torch.FloatTensor(np.array(buffer_mu_old)).to(device)
        b_std_old = torch.FloatTensor(np.array(buffer_std_old)).to(device)

        b_inds = np.arange(HYPERPARAMS['horizon'])
        old_dist_full = torch.distributions.Normal(b_mu_old, b_std_old)

        for epoch in range(HYPERPARAMS['n_epochs']):
            np.random.shuffle(b_inds)

            for start in range(0, HYPERPARAMS['horizon'], HYPERPARAMS['minibatch_size']):
                end = start + HYPERPARAMS['minibatch_size']
                mb_inds = b_inds[start:end]

                new_dist = policy(b_obs[mb_inds])
                new_logprob = new_dist.log_prob(b_act[mb_inds]).sum(dim=1)
                new_val = value_net(b_obs[mb_inds]).squeeze()

                old_dist_mb = torch.distributions.Normal(b_mu_old[mb_inds], b_std_old[mb_inds])

                logratio = new_logprob - b_logprob[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # KL Divergence Term: KL[old || new]
                kl_div = torch.distributions.kl_divergence(old_dist_mb, new_dist).sum(dim=1)

                # Policy Loss: L_KLPEN = r_t*A - beta * KL
                surrogate_objective = mb_adv * ratio
                pg_loss = -(surrogate_objective.mean() - beta * kl_div.mean())

                v_loss = 0.5 * ((new_val - b_ret[mb_inds]) ** 2).mean()

                # Policy Update
                optimizer_policy.zero_grad()
                pg_loss.backward()
                optimizer_policy.step()

                # Value Update
                optimizer_value.zero_grad()
                v_loss.backward()
                optimizer_value.step()

        # logging
        with torch.no_grad():
            new_dist_full = policy(b_obs)
            kl_div_full = torch.distributions.kl_divergence(old_dist_full, new_dist_full).sum(dim=1)
            mean_kl = kl_div_full.mean().item()

        if len(episode_rewards) > 0:
            # 마지막 10개 에피소드 평균 보상을 계산하여 기록
            avg_ep_reward = np.mean(episode_rewards[-10:])
            log_rewards_per_update.append(avg_ep_reward)

            if update % 10 == 0:
                print(
                    f"[{update}/{num_updates}] Steps: {global_step} | Avg. Episode Reward (Last 10): {avg_ep_reward:.2f} | beta: {beta:.4f} | Mean KL: {mean_kl:.4f}")


    env.close()

    # 최종적으로 논문과 같이 마지막 100개 episode 평균을 계산
    final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"\n--- Seed {seed} 학습 완료. Final Avg. Reward: {final_avg_reward:.2f} ---")

    return log_rewards_per_update


# plotting function
def plot_confidence_interval(all_results, window=10):

    results = np.array(all_results)
    num_updates = results.shape[1]

    timesteps_per_update = HYPERPARAMS['horizon']
    total_timesteps = num_updates * timesteps_per_update
    timesteps = np.linspace(0, total_timesteps, num_updates)

    mean_rewards = np.mean(results, axis=0)

    std_rewards = np.std(results, axis=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    color = '#ff7f0e'

    ax.plot(timesteps, mean_rewards, label=f'Avg. Reward (N={results.shape[0]} runs)', color=color)

    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        color=color,
        label='Confidence Interval ($\pm 1$ Std Dev)'
    )

    ax.set_title(f'{HYPERPARAMS["env_id"]} PPO Fixed KL Penalty ($\\beta={HYPERPARAMS["beta"]}$)', fontsize=16)
    ax.set_xlabel('Environment Timesteps', fontsize=12)
    ax.set_ylabel(f'Mean Episode Reward (Avg. over {window} episodes)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    save_path = f'ppo_kl_penalty_beta_{HYPERPARAMS["beta"]}.png'
    plt.savefig(save_path)
    print(f"\n[결과] 신뢰구간 그래프가 {save_path}로 저장되었습니다.")
    plt.show()


def main_experiment():
    #random seed 3개 선택
    random_seeds = random.sample(range(1, 10001), 3)
    HYPERPARAMS['seeds'] = random_seeds

    print(f"\n==========================================")
    print(f"** PPO (Fixed KL Penalty) 무작위 시드 실험 **")
    print(f"실험에 사용될 3개의 무작위 Seed: {random_seeds}")
    print(f"==========================================\n")

    all_rewards = []

    for seed in random_seeds:
        rewards = train_ppo(seed)
        all_rewards.append(rewards)

    if not all(len(r) == len(all_rewards[0]) for r in all_rewards):
        print("경고: 리워드 로그 길이가 일치하지 않습니다. 최소 길이에 맞춰 자릅니다.")
        min_len = min(len(r) for r in all_rewards)
        all_rewards = [r[:min_len] for r in all_rewards]

    plot_confidence_interval(all_rewards, window=10)


if __name__ == "__main__":
    main_experiment()