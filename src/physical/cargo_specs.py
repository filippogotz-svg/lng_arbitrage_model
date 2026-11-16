"""
Phase 2 Module 1: LNG Cargo Specifications, Thermodynamic Properties & Vessel Profiles.
Converts liquid LNG cubic metres (m3) into mass (kg/mt) and thermal energy (MMBtu HHV/LHV).
Principle 4.1 compliance: Never assumes static MMBtu/m3 across different LNG grades.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Dict, Tuple


@dataclass(frozen=True)
class LNGGradeSpec:
    """Thermodynamic properties of a specific LNG composition."""
    grade_id: str
    description: str
    density_kg_m3: float
    mass_hhv_mmbtu_kg: float
    mass_lhv_mmbtu_kg: float
    methane_mol_pct: float
    nitrogen_mol_pct: float

    @property
    def volumetric_hhv_mmbtu_m3(self) -> float:
        """Gross calorific value per cubic metre of liquid LNG (MMBtu / m3)."""
        return self.density_kg_m3 * self.mass_hhv_mmbtu_kg

    @property
    def volumetric_lhv_mmbtu_m3(self) -> float:
        """Net calorific value per cubic metre of liquid LNG (MMBtu / m3)."""
        return self.density_kg_m3 * self.mass_lhv_mmbtu_kg


@dataclass(frozen=True)
class VesselProfileSpec:
    """Technical and operational parameters of an LNG carrier class."""
    vessel_id: str
    name: str
    propulsion_type: str
    nominal_capacity_m3: float
    max_filling_limit_pct: float
    arrival_heel_pct: float
    natural_bor_laden_pct_per_day: float
    natural_bor_ballast_pct_per_day: float
    design_speed_laden_knots: float
    design_speed_ballast_knots: float
    propulsion_energy_laden_mmbtu_per_day: float
    propulsion_energy_ballast_mmbtu_per_day: float
    pilot_fuel_fraction: float
    reliquefaction_fraction: float
    reliq_power_mgo_mt_per_mmbtu: float

    @property
    def departure_volume_m3(self) -> float:
        """Total liquid LNG volume onboard at departure after loading to fill limit (m3)."""
        return self.nominal_capacity_m3 * self.max_filling_limit_pct

    @property
    def arrival_heel_volume_m3(self) -> float:
        """Cryogenic liquid heel onboard prior to loading (m3)."""
        return self.nominal_capacity_m3 * self.arrival_heel_pct

    @property
    def net_loaded_volume_m3(self) -> float:
        """Net liquid LNG transferred across the loading flange at origin terminal (m3)."""
        return self.departure_volume_m3 - self.arrival_heel_volume_m3


@dataclass(frozen=True)
class TerminalPhysicalSpec:
    """Physical conversion efficiencies and port turn-around times at terminals."""
    physical_liquefaction_loss_fraction: float
    contractual_feedgas_multiplier: float
    loading_duration_days: float
    unloading_duration_days: float
    nwe_regasification_loss_fraction: float
    nea_regasification_loss_fraction: float


def load_physical_config(config_path: Path = Path("config/physical_assumptions.toml")) -> dict:
    """Loads physical assumptions TOML configuration safely (strips UTF-8 BOM if present)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Physical configuration file not found at {config_path}")
    raw_text = config_path.read_text(encoding="utf-8-sig")
    return tomllib.loads(raw_text)


def load_physical_catalog(
    config_path: Path = Path("config/physical_assumptions.toml"),
) -> Tuple[Dict[str, LNGGradeSpec], Dict[str, VesselProfileSpec], TerminalPhysicalSpec, dict]:
    """
    Parses the physical configuration file into strongly-typed catalogs of
    LNG grades, vessel profiles, and terminal specifications.
    """
    cfg = load_physical_config(config_path)

    grades: Dict[str, LNGGradeSpec] = {}
    for g_id, g_data in cfg["lng_grades"].items():
        grades[g_id] = LNGGradeSpec(
            grade_id=g_id,
            description=str(g_data["description"]),
            density_kg_m3=float(g_data["density_kg_m3"]),
            mass_hhv_mmbtu_kg=float(g_data["mass_hhv_mmbtu_kg"]),
            mass_lhv_mmbtu_kg=float(g_data["mass_lhv_mmbtu_kg"]),
            methane_mol_pct=float(g_data["methane_mol_pct"]),
            nitrogen_mol_pct=float(g_data["nitrogen_mol_pct"]),
        )

    vessels: Dict[str, VesselProfileSpec] = {}
    for v_id, v_data in cfg["vessels"].items():
        vessels[v_id] = VesselProfileSpec(
            vessel_id=v_id,
            name=str(v_data["name"]),
            propulsion_type=str(v_data["propulsion_type"]),
            nominal_capacity_m3=float(v_data["nominal_capacity_m3"]),
            max_filling_limit_pct=float(v_data["max_filling_limit_pct"]),
            arrival_heel_pct=float(v_data["arrival_heel_pct"]),
            natural_bor_laden_pct_per_day=float(v_data["natural_bor_laden_pct_per_day"]),
            natural_bor_ballast_pct_per_day=float(v_data["natural_bor_ballast_pct_per_day"]),
            design_speed_laden_knots=float(v_data["design_speed_laden_knots"]),
            design_speed_ballast_knots=float(v_data["design_speed_ballast_knots"]),
            propulsion_energy_laden_mmbtu_per_day=float(v_data["propulsion_energy_laden_mmbtu_per_day"]),
            propulsion_energy_ballast_mmbtu_per_day=float(v_data["propulsion_energy_ballast_mmbtu_per_day"]),
            pilot_fuel_fraction=float(v_data["pilot_fuel_fraction"]),
            reliquefaction_fraction=float(v_data["reliquefaction_fraction"]),
            reliq_power_mgo_mt_per_mmbtu=float(v_data["reliq_power_mgo_mt_per_mmbtu"]),
        )

    t_cfg = cfg["terminals"]
    terminals = TerminalPhysicalSpec(
        physical_liquefaction_loss_fraction=float(t_cfg["physical_liquefaction_loss_fraction"]),
        contractual_feedgas_multiplier=float(t_cfg["contractual_feedgas_multiplier"]),
        loading_duration_days=float(t_cfg["loading_duration_days"]),
        unloading_duration_days=float(t_cfg["unloading_duration_days"]),
        nwe_regasification_loss_fraction=float(t_cfg["nwe_regasification_loss_fraction"]),
        nea_regasification_loss_fraction=float(t_cfg["nea_regasification_loss_fraction"]),
    )

    return grades, vessels, terminals, cfg