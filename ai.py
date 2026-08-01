import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from game import Direction, Snake


class DuelingDQN(nn.Module):
    """Dueling DQN: splits into value stream + advantage stream."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.feature = nn.Sequential(*layers)

        value_layers: list[nn.Module] = []
        for _ in range(num_layers - 1):
            value_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        value_layers.append(nn.Linear(hidden_size, 1))
        self.value_stream = nn.Sequential(*value_layers)

        advantage_layers: list[nn.Module] = []
        for _ in range(num_layers - 1):
            advantage_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        advantage_layers.append(nn.Linear(hidden_size, output_size))
        self.advantage_stream = nn.Sequential(*advantage_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayMemory:
    """Experience replay buffer storing transitions."""

    def __init__(self, capacity: int) -> None:
        self.memory: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


def get_state(game: Snake, max_snake_length: int) -> list[float]:
    """Convert game state to a feature vector for the neural network.

    Features (all normalized to [0, 1]):
        - Snake body positions as [x, y] pairs, padded to max_snake_length
        - Food position [x, y]
        - Game dimension [w, h]
    """
    w, h = game.dimension
    flat: list[float] = []
    for seg in game.snake:
        flat.extend([seg[0] / w, seg[1] / h])
    # Pad remaining snake slots with zeros
    flat.extend([0.0] * (max_snake_length * 2 - len(flat)))
    # Append normalized food and dimension
    flat.extend([game.food[0] / w, game.food[1] / h])
    flat.extend([1.0, 1.0])  # dimension normalized by itself
    # Append direction one-hot
    flat.extend(
        [
            float(game.direction == Direction.UP),
            float(game.direction == Direction.DOWN),
            float(game.direction == Direction.LEFT),
            float(game.direction == Direction.RIGHT),
        ]
    )
    return flat


ACTION_SIZE = 4  # up, down, left, right
ACTION_MAP = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class DQNAgent:
    """DQN Agent with experience replay, target network, and Double DQN."""

    def __init__(
        self,
        state_size: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 500000,
        batch_size: int = 128,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(state_size, hidden_size, ACTION_SIZE, num_layers).to(
            self.device
        )
        self.target_net = DuelingDQN(state_size, hidden_size, ACTION_SIZE, num_layers).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=50
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayMemory(memory_size)

    def select_action(self, state: list[float], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)

        with torch.no_grad():
            state_t = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self) -> float | None:
        """Perform one training step using Double DQN."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy net selects action, target net evaluates it
            best_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.scheduler.step(loss.item())
        return loss.item()

    def update_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str = "dqn_model.pth") -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str = "dqn_model.pth") -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def train(
    episodes: int = 5000,
    grid_size: int = 8,
    max_steps: int = 1000,
    target_update: int = 10,
    print_every: int = 50,
    display: bool = False,
    curriculum: bool = True,
) -> DQNAgent:
    """Train the DQN agent on the snake game.

    Args:
        display: Show game state live during training (slower but fun to watch).
        curriculum: Start with small grids and grow over time.
    """
    import os

    # Curriculum: start at 6x6, grow to target grid_size
    sizes = [6] if curriculum else [grid_size]
    if curriculum and grid_size > 6:
        sizes = list(range(6, grid_size + 1, 2))
        if sizes[-1] != grid_size:
            sizes.append(grid_size)

    agent: DQNAgent | None = None
    scores: list[int] = []

    for stage, current_size in enumerate(sizes):
        state_size = current_size * current_size * 2 + 8
        max_snake_length = current_size * current_size

        if agent is None:
            agent = DQNAgent(state_size=state_size)
        else:
            # Resize network for new grid: reinitialize with correct input size
            agent = DQNAgent(state_size=state_size)
            if scores:
                print(f"  Resized network for grid {current_size}x{current_size}")

        episodes_this_stage = episodes // len(sizes) if curriculum else episodes

        for episode in range(1, episodes_this_stage + 1):
            game = Snake(current_size, current_size)
            state = get_state(game, max_snake_length)
            total_reward = 0
            steps_since_food = 0
            stall_limit = current_size * current_size

            for _ in range(max_steps):
                action_idx = agent.select_action(state)
                action = ACTION_MAP[action_idx]

                reward, score, done, _, _, _ = game.tick(action)

                # Reward shaping
                if not done:
                    steps_since_food += 1

                    # Survival bonus
                    reward += 0.1

                    # Stall penalty
                    if steps_since_food >= stall_limit:
                        reward -= 5.0

                    # Reset stall counter when food is eaten
                    if game.score > (scores[-1] if scores else 0):
                        steps_since_food = 0

                next_state = get_state(game, max_snake_length) if not done else state

                agent.memory.push(state, action_idx, reward, next_state, float(done))
                agent.train_step()

                state = next_state
                total_reward += reward

                if display:
                    os.system("cls" if os.name == "nt" else "clear")
                    game.display()
                    print(
                        f"Stage {stage+1}/{len(sizes)} | "
                        f"Grid {current_size}x{current_size} | "
                        f"Episode {episode}/{episodes_this_stage} | "
                        f"Epsilon: {agent.epsilon:.3f}"
                    )

                if done:
                    break

            agent.update_epsilon()
            scores.append(game.score)

            if episode % target_update == 0:
                agent.update_target_network()

            if episode % print_every == 0:
                recent = scores[-print_every:]
                avg = sum(recent) / len(recent)
                best_grid = max(sizes[: stage + 1])
                print(
                    f"[{current_size}x{current_size}] "
                    f"Ep {episode:5d} | "
                    f"Avg: {avg:.1f} | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"Best: {max(scores)} | "
                    f"LR: {agent.optimizer.param_groups[0]['lr']:.6f}"
                )

    if agent is None:
        agent = DQNAgent(state_size=grid_size * grid_size * 2 + 8)
    agent.save()
    return agent


def play(agent: DQNAgent | None = None, grid_size: int = 16) -> None:
    """Play the snake game using a trained agent."""
    if agent is None:
        state_size = grid_size * grid_size * 2 + 8
        agent = DQNAgent(state_size=state_size)
        agent.load()
        agent.epsilon = 0.0

    max_snake_length = grid_size * grid_size
    game = Snake(grid_size, grid_size)
    state = get_state(game, max_snake_length)

    while game.alive:
        game.display()
        action_idx = agent.select_action(state, training=False)
        action = ACTION_MAP[action_idx]
        _, _, done, _, _, _ = game.tick(action)
        state = get_state(game, max_snake_length) if not done else state

    game.display()
    print(f"Final Score: {game.score}")


if __name__ == "__main__":
    import sys

    live = "--live" in sys.argv
    train(episodes=5000, grid_size=8, display=live)
