import os
import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 2000


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    daily_water_usage = rng.integers(80, 650, size=n_samples)
    rice_consumption_kg = rng.uniform(0.0, 1.8, size=n_samples)
    meat_consumption_kg = rng.uniform(0.0, 1.2, size=n_samples)
    electricity_usage_kwh = rng.integers(1, 24, size=n_samples)
    household_size = rng.integers(1, 7, size=n_samples)

    food_water = (rice_consumption_kg * 2500.0) + (meat_consumption_kg * 4300.0)
    electricity_water = electricity_usage_kwh * 50.0

    # Add mild noise to avoid a perfectly deterministic label.
    noise = rng.normal(0.0, 120.0, size=n_samples)

    total_water_footprint = (
        daily_water_usage
        + food_water
        + electricity_water
        + (household_size * 40.0)
        + noise
    )

    df = pd.DataFrame(
        {
            "daily_water_usage": daily_water_usage,
            "rice_consumption_kg": rice_consumption_kg,
            "meat_consumption_kg": meat_consumption_kg,
            "electricity_usage_kwh": electricity_usage_kwh,
            "household_size": household_size,
            "food_water": food_water,
            "electricity_water": electricity_water,
            "total_water_footprint": np.clip(total_water_footprint, a_min=0, a_max=None),
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_dataset()
    out_path = "data/water_data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Dataset created: {out_path} ({len(dataset)} rows)")
