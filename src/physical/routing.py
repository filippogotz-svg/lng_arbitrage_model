"""
Phase 2 Module 2: Marine Routing, Voyage Duration & Schedule Calculator.
Computes laden steaming days, ballast return days, canal transit delays,
and exact loading/arrival calendar timestamps.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

from src.physical.cargo_specs import TerminalPhysicalSpec, VesselProfileSpec, load_physical_config


@dataclass(frozen=True)
class RouteSpec:
    """Geographical and operational parameters for a shipping route."""
    route_id: str
    destination_region: str
    route_name: str
    distance_laden_nm: float
    distance_ballast_nm: float
    canal_transit_days_laden: float
    canal_transit_days_ballast: float
    weather_port_buffer_days_laden: float
    weather_port_buffer_days_ballast: float
    requires_panama: bool
    requires_suez: bool


@dataclass(frozen=True)
class VoyageSchedule:
    """Complete duration and calendar breakdown for a single cargo voyage."""
    route_id: str
    destination_region: str
    speed_laden_knots: float
    speed_ballast_knots: float
    steaming_days_laden: float
    canal_days_laden: float
    buffer_days_laden: float
    total_laden_sea_days: float
    steaming_days_ballast: float
    canal_days_ballast: float
    buffer_days_ballast: float
    total_ballast_sea_days: float
    loading_port_days: float
    unloading_port_days: float
    total_round_trip_days: float
    loading_start_date: Optional[str] = None
    departure_date: Optional[str] = None
    arrival_date: Optional[str] = None
    discharge_complete_date: Optional[str] = None
    round_trip_complete_date: Optional[str] = None


def load_routes_catalog(
    config_path: Path = Path("config/physical_assumptions.toml"),
) -> Dict[str, RouteSpec]:
    """Loads all configured shipping routes from physical_assumptions.toml."""
    cfg = load_physical_config(config_path)
    routes: Dict[str, RouteSpec] = {}
    for r_id, r_data in cfg["routes"].items():
        routes[r_id] = RouteSpec(
            route_id=r_id,
            destination_region=str(r_data["destination_region"]),
            route_name=str(r_data["route_name"]),
            distance_laden_nm=float(r_data["distance_laden_nm"]),
            distance_ballast_nm=float(r_data["distance_ballast_nm"]),
            canal_transit_days_laden=float(r_data["canal_transit_days_laden"]),
            canal_transit_days_ballast=float(r_data["canal_transit_days_ballast"]),
            weather_port_buffer_days_laden=float(r_data["weather_port_buffer_days_laden"]),
            weather_port_buffer_days_ballast=float(r_data["weather_port_buffer_days_ballast"]),
            requires_panama=bool(r_data["requires_panama"]),
            requires_suez=bool(r_data["requires_suez"]),
        )
    return routes


def calculate_voyage_schedule(
    route: RouteSpec,
    vessel: VesselProfileSpec,
    terminals: TerminalPhysicalSpec,
    loading_start_date: Optional[date] = None,
    speed_laden_override_knots: Optional[float] = None,
    speed_ballast_override_knots: Optional[float] = None,
) -> VoyageSchedule:
    """
    Calculates exact steaming, canal, port, and round-trip durations.
    Equation:
        t_steaming = Distance_nm / (24 * Speed_knots)
        t_laden_sea = t_steaming_laden + t_canal_laden + t_buffer_laden
    """
    v_laden = speed_laden_override_knots or vessel.design_speed_laden_knots
    v_ballast = speed_ballast_override_knots or vessel.design_speed_ballast_knots

    if v_laden <= 0 or v_ballast <= 0:
        raise ValueError("Vessel speed in knots must be strictly positive.")

    steaming_laden = route.distance_laden_nm / (24.0 * v_laden)
    total_laden_sea = (
        steaming_laden + route.canal_transit_days_laden + route.weather_port_buffer_days_laden
    )

    steaming_ballast = route.distance_ballast_nm / (24.0 * v_ballast)
    total_ballast_sea = (
        steaming_ballast + route.canal_transit_days_ballast + route.weather_port_buffer_days_ballast
    )

    total_round_trip = (
        terminals.loading_duration_days
        + total_laden_sea
        + terminals.unloading_duration_days
        + total_ballast_sea
    )

    load_str = dep_str = arr_str = dis_str = rt_str = None
    if loading_start_date is not None:
        load_str = loading_start_date.isoformat()
        dep_dt = loading_start_date + timedelta(days=terminals.loading_duration_days)
        arr_dt = dep_dt + timedelta(days=total_laden_sea)
        dis_dt = arr_dt + timedelta(days=terminals.unloading_duration_days)
        rt_dt = dis_dt + timedelta(days=total_ballast_sea)

        dep_str = dep_dt.isoformat()
        arr_str = arr_dt.isoformat()
        dis_str = dis_dt.isoformat()
        rt_str = rt_dt.isoformat()

    return VoyageSchedule(
        route_id=route.route_id,
        destination_region=route.destination_region,
        speed_laden_knots=v_laden,
        speed_ballast_knots=v_ballast,
        steaming_days_laden=steaming_laden,
        canal_days_laden=route.canal_transit_days_laden,
        buffer_days_laden=route.weather_port_buffer_days_laden,
        total_laden_sea_days=total_laden_sea,
        steaming_days_ballast=steaming_ballast,
        canal_days_ballast=route.canal_transit_days_ballast,
        buffer_days_ballast=route.weather_port_buffer_days_ballast,
        total_ballast_sea_days=total_ballast_sea,
        loading_port_days=terminals.loading_duration_days,
        unloading_port_days=terminals.unloading_duration_days,
        total_round_trip_days=total_round_trip,
        loading_start_date=load_str,
        departure_date=dep_str,
        arrival_date=arr_str,
        discharge_complete_date=dis_str,
        round_trip_complete_date=rt_str,
    )