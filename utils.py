from __future__ import annotations

from typing import Dict, List


def water_components(
    daily_water_usage: float,
    rice_consumption_kg: float,
    meat_consumption_kg: float,
    electricity_usage_kwh: float,
    household_size: int,
) -> Dict[str, float]:
    return {
        "Direct Water": daily_water_usage,
        "Rice Water": rice_consumption_kg * 2500.0,
        "Meat Water": meat_consumption_kg * 4300.0,
        "Electricity Water": electricity_usage_kwh * 50.0,
        "Household Impact": household_size * 40.0,
    }


def classify_risk(total_footprint: float) -> str:
    if total_footprint >= 6500:
        return "High"
    if total_footprint >= 3500:
        return "Moderate"
    return "Low"


def suggestions(risk: str) -> List[str]:
    if risk == "High":
        return [
            "Reduce meat-heavy meals 2-3 times per week.",
            "Fix leak points and monitor direct tap usage daily.",
            "Adopt efficient appliances and optimize electricity use.",
        ]
    if risk == "Moderate":
        return [
            "Track weekly consumption and set a reduction target.",
            "Shift one meal per day to lower-water foods.",
            "Use efficient laundry and dishwashing schedules.",
        ]
    return [
        "Maintain current habits and continue monthly tracking.",
        "Share conservation practices with your household.",
        "Set stretch goals for another 5-10% reduction.",
    ]