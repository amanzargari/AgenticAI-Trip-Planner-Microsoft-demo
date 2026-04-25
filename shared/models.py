from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, field_validator


def _coerce_float(v: Any) -> Optional[float]:
    """Convert strings like '€20', 'Free', 'Varies' to float or None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[^\d.]", "", v)
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class Location(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None


class BudgetConstraint(BaseModel):
    total_budget: float
    currency: str = "EUR"


class PlaceCandidate(BaseModel):
    id: str
    name: str
    location: Location
    estimated_visit_duration_minutes: int = 60
    estimated_cost: Optional[float] = None
    category: Optional[str] = None
    rating: Optional[float] = None
    summary: Optional[str] = None

    @field_validator("estimated_cost", "rating", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> Optional[float]:
        return _coerce_float(v)


class RestaurantCandidate(BaseModel):
    id: str
    name: str
    location: Location
    price_level: Optional[float] = None
    cuisines: Optional[list[str]] = None
    rating: Optional[float] = None
    summary: Optional[str] = None

    @field_validator("price_level", "rating", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> Optional[float]:
        return _coerce_float(v)


class VisitEvent(BaseModel):
    type: Literal["visit"] = "visit"
    start_time: datetime
    end_time: datetime
    place: PlaceCandidate


class MealEvent(BaseModel):
    type: Literal["meal"] = "meal"
    time: datetime
    restaurant: RestaurantCandidate
    meal_slot: Literal["breakfast", "lunch", "dinner"]


class DailySchedule(BaseModel):
    date: date
    events: list[Union[VisitEvent, MealEvent]]


class TripItinerary(BaseModel):
    city: str
    trip_start: datetime
    trip_end: datetime
    total_budget: Optional[float] = None
    schedules: list[DailySchedule]
