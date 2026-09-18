"""
Lunar Lander -- Discrete Variant
=============================================

Task
----
Fire directional thrusters to land a rigid-body lander safely on a landing
pad between two flags, minimizing fuel use and impact velocity.

Design decisions (see accompanying explanation for the general framework)
---------------------------------------------------------------------
1. Observation space  -> Box(8,): x, y, vx, vy, angle, angular velocity,
                          left-leg contact, right-leg contact.
2. Action space        -> Discrete(4): {no-op, fire left engine,
                          fire main engine, fire right engine}.
3. Transition dynamics -> Box2D physics simulation (deterministic given
                          seed; no additional noise injected here).
4. Reward function     -> built into the environment: shaping toward the
                          pad, angle/velocity penalties, leg-contact
                          bonuses, +100 for a safe landing, -100 for a
                          crash, small per-frame fuel penalty for firing
                          engines. Documented component-by-component
                          below since Gymnasium does not expose the
                          breakdown directly.
5. Termination         -> terminated: lander comes to rest (safe landing)
                          OR crashes OR leaves the viewport.
                          truncated: step-limit cutoff (rare in practice
                          for this task; episodes are usually short).
"""

from __future__ import annotations

import gymnasium as gym

# --------------------------------------------------------------------------- 
# Action enumeration -- Gymnasium's own mapping for LunarLander-v3.
# --------------------------------------------------------------------------- 
NOOP, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT = 0, 1, 2, 3
ACTION_NAMES = {
    NOOP: "noop",
    FIRE_LEFT: "fire_left_engine",
    FIRE_MAIN: "fire_main_engine",
    FIRE_RIGHT: "fire_right_engine",
}

# --------------------------------------------------------------------------- 
# Observation index reference (Box(8,), shared with the continuous variant)
# --------------------------------------------------------------------------- 
OBS_FIELDS = [
    "x_position", "y_position", "x_velocity", "y_velocity",
    "angle", "angular_velocity", "left_leg_contact", "right_leg_contact",
]

ENV_ID = "LunarLander-v3"


def make_env(render_mode: str | None = None, **kwargs) -> gym.Env:
    """
    Factory for the discrete Lunar Lander environment.

    Kept as a factory function (rather than a bare gym.make call at import
    time) so this file can be imported by a training script without
    immediately spinning up a rendering context.
    """
    return gym.make(ENV_ID, render_mode=render_mode, **kwargs)


def describe(env: gym.Env) -> None:
    """Print the observation/action space"""
    print(f"[{ENV_ID}] observation_space: {env.observation_space}")
    print(f"[{ENV_ID}] action_space:      {env.action_space}")
    print(f"[{ENV_ID}] action mapping:    {ACTION_NAMES}")
    print(f"[{ENV_ID}] obs field order:   {OBS_FIELDS}")


if __name__ == "__main__":
    # Smoke test: random policy for a few steps, printed to console.
    # Use render_mode="human" locally if you want a live window; headless
    env = make_env(render_mode="human")
    describe(env)

    obs, info = env.reset(seed=2)
    total_reward = 0.0
    for t in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if t < 5 or terminated or truncated:
            print(f"t={t:03d} action={ACTION_NAMES[action]:18s} "
                  f"reward={reward:+.3f} terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            break
    print(f"Episode return (random policy, {t + 1} steps): {total_reward:.2f}")
    env.close()
