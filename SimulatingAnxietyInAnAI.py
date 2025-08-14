#TODOS
#1. make it so anyone can install the env to their gymnasium install :)
#2. open-source the project
#3. FUTURE: maybe ask a museum to make this an exhibit?
#4. soundgen???????????
#5. some sort of way to synthisize text for the agent's thoughts?

debuggpuaccelwarn = False
# SimulatingAnxietyInAnAI.py
import sys
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tkinter import messagebox
# ====== Custom Environment with Invigilator and Panic ======
class AnxiousTestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_steps = 10000
        
        self.agent_pos = np.random.uniform(0.0,1.0)
        self.observer_pos = np.random.uniform(0.0, 1.0)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 0.5
        self.observer_pos = np.random.uniform(0.0, 1.0)
        self.state = np.array([
            np.random.uniform(0.3, 0.9),
            1.0,
            np.random.uniform(0.5, 1.0),
            np.random.uniform(0.0, 0.3),
            abs(self.agent_pos - self.observer_pos)
        ], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        difficulty, time_left, confidence, stress, _ = self.state

        self.observer_pos += 0.01 * (self.agent_pos - self.observer_pos)
        self.observer_pos = np.clip(self.observer_pos, 0.0, 1.0)
        observer_distance = abs(self.agent_pos - self.observer_pos)

        observer_threat = 1.0 - observer_distance
        stress += observer_threat * 0.8

        correct = confidence > difficulty and action in [0, 1]
        reward = 1.5 if correct else -1.2
        if correct:
            stress = stress / 2.2
        else:
            stress = stress * 1.7
        stress *= 0.5  # calming but with a higher exponenent to the curve of the calming
        if action == 1: stress += 0.1 * difficulty
        elif action == 2: stress += 0.05 * difficulty; reward -= 0.5
        elif action == 3: stress += 0.3 * difficulty; reward -= 2.0
        safe_reward = 5.8 - (1.0 - observer_distance)
        maxstress = (1.2 * safe_reward) / (2 * stress * difficulty / random.uniform(1.6,5.0))
        if maxstress < 3.1:
            maxstress = 3.1
        elif maxstress > 7.0:
            maxstress = 7.0
        print(maxstress)
        if stress > maxstress:
            if np.random.uniform(0.0,1.0) > maxstress / 100:
                terminated = False # Nope! the agent coped!
                truncated = False # this wont do anything, but we HAVE to define it
                reward += 40 # reward for calming down!
                stress  = 0 
                
                print("soothed: panic adverted!")
            else:
                terminated = True # TERMINATE the current episode, this is a programming parallel to when schools figurativeley, SLAM THE DOOR on the learning of students with anxiety when they panic
                truncated = False # it WAS NOT ended via time running out.
                reward -= 20.0
                print("[TERMINATION_EVENT] anxiety_attack: panic attack triggered! ! (episode TERMINATED!)") # this simulates, sadly, the harsh reality for many people with anxiety when they panic
            
            
        else:
            terminated = stress >= 2.0 or time_left <= 0.0
            if (self.steps >= self.max_steps):
                print("TIME UP!")
            truncated = self.steps >= self.max_steps
            
            

        time_left -= 1.0 / self.max_steps
        self.steps += 1

        difficulty = np.random.uniform(0.5, 0.9)
        confidence = np.clip(confidence + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
        if correct:
            confidence += 0.08
        else:
            confidence -= 0.08
        confidence = np.clip(confidence, 0.0, 1.0)

        self.state = np.array([
            difficulty,
            time_left,
            confidence,
            stress,
            observer_distance
        ], dtype=np.float32)

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Step {self.steps} | Stress: {self.state[3]:.2f} | Observer Dist: {self.state[4]:.2f}")

    def close(self):
        pass

# ====== DQN Agent ======
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ====== GUI App ======
class DQNTrainerApp:
    def on_exit(self):
        self.running = False
        self.root.destroy()
        sys.exit(0)
    def __init__(self, root):
        self.root = root
        self.root.title("Anxiety: a Deep Q learning demo")
        self.running = False
        self.reset_interval = 100
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.start_button = ttk.Button(root, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(root, text="Stop Training", command=self.stop_training)
        self.stop_button.pack(pady=5)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 8))
        self.ax1.set_title("Episode Rewards")
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Reward")
        self.reward_line, = self.ax1.plot([], [], color="blue")

        self.ax2.set_title("Max Stress")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Stress")
        self.stress_line, = self.ax2.plot([], [], color="red")

        self.ax3.set_title("Observer Distance")
        self.ax3.set_xlabel("Episode")
        self.ax3.set_ylabel("Distance")
        self.observer_line, = self.ax3.plot([], [], color="green")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.episode_rewards = []
        self.stress_values = []
        self.observer_distances = []

    def start_training(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.train_dqn, daemon=True).start()

    def stop_training(self):
        self.running = False

    def train_dqn(self):
        env = AnxiousTestEnv()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and not debuggpuaccelwarn:
            messagebox.showinfo("CUDA ACCELERATION ON! ",
            "A CUDA compatible GPU was found, it will be used to run the simulation!")
        else:
            messagebox.showwarning(
        "PERFORMANCE WARNING! ! !",
        ""
        "No CUDA-compatible NVIDIA GPU detected. (do you have the CUDA build of pytorch?)\n\n"
        "⚠️ Non-CUDA GPUs (like Intel or AMD) are NOT supported for acceleration.\n"
        "Inference will proceed on CPU, but performance will be significantly slower.\n\n"
        "Consider switching to a CUDA-enabled system for optimal experience."
        f"{ ' note that this may not be true since you have debugging of this warning on!' if debuggpuaccelwarn else ''}"
        )


        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        policy_net = DQN(input_dim, output_dim).to(device)
        target_net = DQN(input_dim, output_dim).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
        replay_buffer = deque(maxlen=500)
        batch_size = 100
        gamma = 0.85
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.05
        update_target_every = 5
        episode = 0

        while self.running:
            if episode % self.reset_interval == 0 and episode != 0:
                self.episode_rewards.clear()
                self.stress_values.clear()
                self.observer_distances.clear()

            state, _ = env.reset()
            total_reward = 0
            max_stress = 0.0
            max_observer_dist = 0.0

            for t in range(env.max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                        action = torch.argmax(q_values).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                replay_buffer.append((state, action, reward, next_state, terminated))
                state = next_state
                total_reward += reward
                max_stress = max(max_stress, state[3])
                max_observer_dist = max(max_observer_dist, state[4])

                if terminated or truncated:
                    break

                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.FloatTensor(states).to(device)
                    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                    next_states = torch.FloatTensor(next_states).to(device)
                    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                    q_values = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q = rewards + gamma * max_next_q * (1 - dones)

                    loss = nn.functional.mse_loss(q_values, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            if episode % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            self.episode_rewards.append(total_reward)
            self.stress_values.append(max_stress)
            self.observer_distances.append(max_observer_dist)
            self.update_plot()
            episode += 1

        env.close()

    def update_plot(self):
        self.reward_line.set_data(range(len(self.episode_rewards)), self.episode_rewards)
        self.ax1.relim()
        self.ax1.autoscale_view()

        self.stress_line.set_data(range(len(self.stress_values)), self.stress_values)
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.observer_line.set_data(range(len(self.observer_distances)), self.observer_distances)
        self.ax3.relim()
        self.ax3.autoscale_view()

        self.canvas.draw()

# ====== Run App ======
if __name__ == "__main__":
    root = tk.Tk()
    app = DQNTrainerApp(root)
    root.mainloop()
    

