import rl_zoo3
import rl_zoo3.record_video
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ
import runpy
import sys

# Patch ALGOS
rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
# See note below to use DroQ configuration
# rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.record_video.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    # Forward command-line arguments to the script
    sys.argv = [
        sys.argv[0],  # script name
        "--algo", "ppo",
        "--env", "CartPole-v1",
        "-n", "1000",
        "--load-best",
        "--folder", "logs"
    ]
    runpy.run_path(rl_zoo3.record_video.__file__, run_name="__main__")