# Create a callback to track rewards
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.logger import (
    Figure,
)  # For potential future logging integration
from typing import Dict, List, Optional, Tuple, Union, Any

# Note: Callbacks currently assume interaction with a single vectorized environment (num_vec_envs=1)
# due to accessing self.locals elements with index [0]. Generalizing requires iterating
# or aggregating results across the VecEnv's environments.


class MARLRewardCallback(BaseCallback):
    """
    Callback for tracking and plotting MARL training metrics.

    Tracks rewards, episode lengths, and capture counts during training.
    Assumes a single vectorized environment (num_vec_envs=1).

    :param num_pursuers: Number of pursuer agents.
    :param num_evaders: Number of evader agents.
    :param verbose: Verbosity level.
    """

    def __init__(self, num_pursuers: int, num_evaders: int, verbose: int = 0):
        super().__init__(verbose)
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders

        # Overall tracking
        self.total_steps = 0
        self.total_capture_count = 0
        self.episodes = 0

        # Per-episode tracking during training
        self._current_episode_steps = 0
        self._episode_total_reward = 0.0
        self._episode_capture_count = 0

        # History storage
        self.episode_lengths: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_capture_counts: List[int] = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Check if the number of environments is greater than 1, warn if so.
        if self.training_env.num_envs > 1 and self.n_calls == 0:
            print(
                "Warning: MARLRewardCallback is designed for num_vec_envs=1. Metrics might only reflect the first environment."
            )

        # Assume num_vec_envs=1
        if len(self.locals["rewards"]) > 0:
            self.total_steps += self.training_env.num_envs  # Increment by num envs
            self._current_episode_steps += 1

            reward = self.locals["rewards"][0]
            self._episode_total_reward += reward

            info = self.locals["infos"][0]
            # Use .get() for safer access
            if info.get("capture", False):
                self._episode_capture_count += 1
                self.total_capture_count += 1  # Tracks across all episodes

            done = self.locals["dones"][0]
            if done:
                self.episodes += 1
                self.episode_lengths.append(self._current_episode_steps)
                self.episode_rewards.append(self._episode_total_reward)
                self.episode_capture_counts.append(self._episode_capture_count)

                # Log metrics using SB3 logger (integrates with TensorBoard etc.)
                if self.logger is not None:
                    # Log smoothed/averaged values for cleaner graphs
                    # Calculate mean over last 100 episodes, or fewer if less than 100 total
                    ep_len_mean = (
                        np.mean(self.episode_lengths[-100:])
                        if self.episode_lengths
                        else 0
                    )
                    ep_rew_mean = (
                        np.mean(self.episode_rewards[-100:])
                        if self.episode_rewards
                        else 0
                    )
                    ep_cap_mean = (
                        np.mean(self.episode_capture_counts[-100:])
                        if self.episode_capture_counts
                        else 0
                    )

                    self.logger.record("rollout/ep_len_mean", ep_len_mean)
                    self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
                    self.logger.record("rollout/ep_captures_mean", ep_cap_mean)
                    self.logger.record("rollout/episodes", self.episodes)
                    # Optionally log raw values too, perhaps less frequently
                    # self.logger.record("rollout/ep_reward_raw", self._episode_total_reward)
                    # self.logger.record("rollout/ep_length_raw", self._current_episode_steps)

                if self.verbose > 0:
                    print(
                        f"Episode {self.episodes}: Len={self._current_episode_steps}, Reward={self._episode_total_reward:.2f}, Captures={self._episode_capture_count}"
                    )

                # Reset episode trackers
                self._current_episode_steps = 0
                self._episode_total_reward = 0.0
                self._episode_capture_count = 0

        return True

    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        Generates and saves plots for key training metrics.

        :param save_dir: Directory to save the plot images. If None, saves to current directory.
        """
        if not self.episode_rewards:
            print("No episode data collected yet. Skipping metric plotting.")
            return

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
            base_path = save_dir
        else:
            base_path = "."  # Save to current directory if none provided

        summary_plot_path = os.path.join(base_path, "marl_metrics_summary.png")
        reward_curve_path = os.path.join(base_path, "reward_curve.png")

        # --- Plot 1: Summary Metrics ---
        fig_summary, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig_summary.suptitle("Training Metrics Summary")

        # 1a. Total reward per episode
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title("Total Reward per Episode")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # 1b. Captures per episode
        axes[0, 1].plot(self.episode_capture_counts)
        axes[0, 1].set_title("Captures per Episode")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Number of Captures")
        axes[0, 1].grid(True)

        # 1c. Capture rate (cumulative average)
        if self.episode_capture_counts:
            cumulative_capture_rate = np.cumsum(self.episode_capture_counts) / (
                np.arange(len(self.episode_capture_counts)) + 1
            )
            axes[1, 0].plot(cumulative_capture_rate)
            axes[1, 0].set_title("Average Captures per Episode (Cumulative)")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Avg. Captures")
            axes[1, 0].grid(True)
        else:
            axes[1, 0].set_title("Capture Rate (No Data)")

        # 1d. Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title("Episode Lengths")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Steps")
        axes[1, 1].grid(True)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to prevent title overlap
        try:
            plt.savefig(summary_plot_path, dpi=300)
            print(f"Summary metrics plot saved to {summary_plot_path}")
        except Exception as e:
            print(f"Error saving summary metrics plot: {e}")
        plt.close(fig_summary)

        # --- Plot 2: Reward Curve with Moving Average ---
        plt.figure(figsize=(10, 6))
        window_size = min(
            50, len(self.episode_rewards)
        )  # Ensure window is not larger than data
        if window_size > 0:
            # Calculate moving average only if window size is valid
            moving_avg = np.convolve(
                self.episode_rewards, np.ones(window_size) / window_size, mode="valid"
            )
            # Plot raw rewards
            plt.plot(
                self.episode_rewards,
                alpha=0.3,
                color="blue",
                label="Raw Episode Reward",
            )
            # Plot moving average - adjust x-axis to align correctly
            plt.plot(
                np.arange(window_size - 1, len(self.episode_rewards)),
                moving_avg,
                color="red",
                label=f"Moving Average (window={window_size})",
            )
            plt.title("Episode Rewards with Moving Average")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.legend()
            plt.grid(True)
            try:
                plt.savefig(reward_curve_path, dpi=300)
                print(f"Reward curve plot saved to {reward_curve_path}")
            except Exception as e:
                print(f"Error saving reward curve plot: {e}")
            plt.close()
        else:
            print("Not enough data for moving average plot.")
            plt.close()  # Close the figure if not used

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calculates and returns a dictionary of summary metrics."""
        avg_len = np.mean(self.episode_lengths) if self.episode_lengths else 0
        avg_rew = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_caps = (
            np.mean(self.episode_capture_counts) if self.episode_capture_counts else 0
        )
        last_10_rew_avg = (
            np.mean(self.episode_rewards[-10:])
            if len(self.episode_rewards) >= 10
            else avg_rew
        )  # Use overall avg if < 10 eps

        return {
            "total_episodes": self.episodes,
            "total_steps": self.total_steps,
            "average_episode_length": avg_len,
            "average_episode_reward": avg_rew,
            "average_captures_per_episode": avg_caps,
            "final_10_episodes_avg_reward": last_10_rew_avg,
        }


class CaptureDebugCallback(BaseCallback):
    """Simple callback to print a message when a capture occurs."""

    def __init__(self, verbose: int = 1):  # Default verbose to 1 for printing
        super().__init__(verbose)
        self.capture_count = 0

    def _on_step(self) -> bool:
        # Assume num_vec_envs=1
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if info.get("capture", False):  # Use .get()
                self.capture_count += 1
                if self.verbose > 0:
                    print(
                        f"Step {self.n_calls}: Capture detected! Total: {self.capture_count}"
                    )
        return True


class DetailedDebugCallback(BaseCallback):
    """Callback to print detailed step information periodically."""

    def __init__(
        self, print_freq: int = 1000, verbose: int = 1
    ):  # Default verbose to 1
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # Assume num_vec_envs=1
        if (
            self.verbose > 0
            and self.n_calls % self.print_freq == 0
            and self.n_calls > 0
        ):
            if len(self.locals["infos"]) > 0:
                info = self.locals["infos"][0]
                rewards = self.locals["rewards"][0]
                dones = self.locals["dones"][0]
                print(f"\n--- Detailed Debug Callback (Step {self.n_calls}) ---")
                print(f"  Reward: {rewards}")
                print(f"  Done: {dones}")
                print(f"  Info: {info}")
                print(f"--- End Debug ---")
        return True


class EscapeDebugCallback(BaseCallback):
    """
    Callback to detect and print when an escape event occurs.
    Checks for the 'escape_event' key in the VecEnv's info dictionary.
    Assumes num_vec_envs=1.
    """

    def __init__(self, verbose: int = 1):  # Default verbose to 1
        super().__init__(verbose)
        self.escape_count = 0
        self._last_escape_step = -1  # Track step to avoid double counting within a step

    def _on_step(self) -> bool:
        # Assume num_vec_envs=1
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            done = self.locals["dones"][0]

            # Use .get() for safety
            if info.get("escape_event", False):
                # Avoid counting the same event multiple times if logic causes issues
                if self.n_calls != self._last_escape_step:
                    self.escape_count += 1
                    self._last_escape_step = self.n_calls
                    if self.verbose > 0:
                        print(
                            f"Step {self.n_calls}: Escape detected! Total: {self.escape_count}"
                        )

            # Reset step tracker if the episode ended
            if done:
                self._last_escape_step = -1

        return True
