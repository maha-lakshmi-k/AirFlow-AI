import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartAirEnv(gym.Env):
    def __init__(self):
        super(SmartAirEnv, self).__init__()
        
        # 4 Discrete Actions: 0: OFF, 1: ECO, 2: MED, 3: TURBO
        self.action_space = spaces.Discrete(4)
        
        # Observation Space bounds
        low = np.array([0.0, 15.0, 20.0, 0.0], dtype=np.float32)
        high = np.array([500.0, 40.0, 90.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.reset()

    def get_discrete_state(self):
        aqi_bin = int(np.clip(np.digitize(self.aqi, [50, 100, 150, 200, 250]), 0, 5))
        temp_bin = int(np.clip(np.digitize(self.temp, [22, 26, 30, 34]), 0, 4))
        tariff_bin = int(self.tariff)
        return (aqi_bin, temp_bin, tariff_bin)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.aqi = 200.0
        self.temp = 29.0
        self.humidity = 50.0
        self.tariff = 0.0
        self.time_step = 0

        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        """Return the continuous observation vector as required by the Box space."""
        return np.array(
            [self.aqi, self.temp, self.humidity, self.tariff],
            dtype=np.float32,
        )

    def step(self, action):
        self.time_step += 1

        power_map     = {0: 0.0, 1: 20.0, 2: 40.0, 3: 80.0}
        cleansing_map = {0: 0.0, 1: 15.0, 2: 30.0, 3: 50.0}

        power_used      = power_map[action]
        cleansing_power = cleansing_map[action]

        # Pollution influx dynamics  (use seeded np_random for determinism)
        natural_pollution_influx = float(self.np_random.uniform(4.0, 10.0))
        self.aqi  = float(max(10.0, self.aqi - cleansing_power + natural_pollution_influx))

        # Temperature cooling effect
        self.temp = float(max(18.0, self.temp - (0.15 * action) + self.np_random.uniform(-0.1, 0.2)))

        # ── Reward components ─────────────────────────────────────────────
        # 1. Quadratic AQI penalty (report formulation: max(0, AQI - 50)²)
        aqi_penalty = max(0.0, self.aqi - 50.0) ** 2

        # 2. Thermal comfort penalty — penalise deviation outside 22–26 °C
        if self.temp < 22.0:
            thermal_penalty = (22.0 - self.temp) ** 2
        elif self.temp > 26.0:
            thermal_penalty = (self.temp - 26.0) ** 2
        else:
            thermal_penalty = 0.0

        # 3. Energy cost (tariff-weighted)
        energy_cost = (power_used / 10.0) * (2.5 if self.tariff == 1.0 else 1.0)

        reward = float(-(aqi_penalty * 0.01 + thermal_penalty * 0.1 + energy_cost * 0.9))
        done   = self.time_step >= 40

        info = {
            "power_used_watts": power_used,
            "aqi":              self.aqi,
            "temp":             self.temp,
            "tariff":           self.tariff,
            "aqi_penalty":      aqi_penalty,
            "thermal_penalty":  thermal_penalty,
            "energy_cost":      energy_cost,
        }

        return self._obs(), reward, done, False, info