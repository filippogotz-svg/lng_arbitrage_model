# Physical LNG Cargo Economics & Cross-Basin Arbitrage

When a US Gulf Coast LNG cargo loads at Sabine Pass or Cameron, a trading desk faces three practical questions:
1. **Where should the vessel go?** Should it head across the Atlantic to Northwest Europe (TTF) or cross to Northeast Asia (JKM) via Panama, Suez, or the Cape of Good Hope?
2. **At what speed should it sail?** Does current freight and bunker pricing justify steaming at 19.5 knots, or does slow-steaming maximize daily return?
3. **Should we lift the cargo at all?** In an oversupplied market, does the destination netback at least cover variable upstream and voyage costs, or is it better to shut in the lift and swallow the sunk liquefaction fee?

Looking at a simple paper spread like `JKM - TTF` on a screen rarely answers these questions properly. Physical LNG arbitrage depends heavily on thermodynamics, ship propulsion, voyage duration, and contract structure. I built this repository to model those mechanics from first principles and backtest them across nearly nine years of market history (December 2017 – November 2026).

---

## Why Paper Spreads Are Misleading (and How This Model Fixes Them)

Most simplified netback calculators make three common mistakes that distort physical routing decisions. This model was designed specifically to avoid them:

* **Zero Energy Leaks & Closed-Cycle Heel Management:** An LNG carrier cannot discharge 100% of its tanks at the destination terminal. It has to retain enough liquid heel (`E_heel_out`) to keep the tanks cold at -161.5°C during the ballast voyage back and still arrive at the next loading port with its required minimum arrival heel (`1.5%` of capacity). On a 174,000 m³ modern two-stroke carrier, a direct voyage to Rotterdam delivers **98.38%** of loaded energy, whereas routing to Tokyo via the Cape of Good Hope delivers **94.95%**.
* **No Double-Counting Between Boil-Off Gas (BOG) and Bunker Fuel:** Natural boil-off gas is burned in the ship's engines first (after accounting for mandatory liquid pilot fuel). You only pay cash for supplementary MGO/VLSFO—or force additional boil-off from the cargo—when propulsion demand at a given speed exceeds natural evaporation. On slow-steaming legs where natural BOG exceeds engine demand, a two-stroke vessel with partial reliquefaction condenses 75% of the excess back into the tanks.
* **Sunk Capacity Fees vs. Marginal Routing Costs:** Once an FOB portfolio player has signed a long-term SPA, the ~$2.50/MMBtu liquefaction tolling fee is a sunk commitment. Including it when deciding whether to divert a cargo from Europe to Asia—or whether to cancel a cargo in a weak market—leads to wrong commercial decisions. The model tags every cost into a 5-tier taxonomy (*Sunk/Committed*, *Variable Upstream*, *Destination-Dependent*, *Avoidable*, and *Working Capital Financing*).

---

## What's Inside the Model

### 1. Multi-Basin Point-in-Time Data Pipeline (`src/data/`)
Combines public EIA/FRED daily Henry Hub spot prices (`DHHNGSP`) with Bloomberg historical settlements for Henry Hub front-month futures (`NG1`), Dutch TTF (`TZT1`), Platts JKM (`JKL1`), and spot `EUR/USD`.
* Converts TTF from `EUR/MWh` to `USD/MMBtu` using same-day FX rates (`(EUR/MWh * EURUSD) / 3.41214`).
* Applies a 60-day rolling MAD filter to flag extreme physical spikes (such as Winter Storm Uri in Feb 2021 or the Aug 2022 European gas squeeze) without deleting real market data.
* Leaves asynchronous exchange holidays (e.g., US Thanksgiving, Lunar New Year in Asia) as `NaN` rather than forward-filling stale prices.

### 2. 11-Stage Thermodynamic & Marine Engine (`src/physical/`)
Tracks volume (m³), mass (mt), and thermal energy (MMBtu HHV/LHV) from pipeline feedgas to downstream regasification grid injection.
* **Fleet Profiles:** `174,000 m³ Two-Stroke (ME-GI / X-DF)` with 0.085%/day laden BOR and partial reliquefaction (baseline), alongside `160,000 m³ TFDE` (0.105%/day BOR) and legacy `145,000 m³ Steam Turbine` (0.150%/day BOR) vessels.
* **LNG Grades:** Supports both standard USGC Lean LNG (`438 kg/m³`) and Rich LNG (`458 kg/m³`) to capture calorific density differences within the 98.5% IMO filling limit.
* **Routes:** USGC to NWE Direct (`4,850 nm`), NEA via Panama (`9,950 nm`), NEA via Suez (`14,450 nm`), and NEA via Cape of Good Hope (`15,600 nm`).

### 3. Cargo Pricer, Break-Evens & Stress Scenarios (`src/economics/`)
Computes full-cycle P&L, FOB flange netbacks, wellhead netbacks, analytical break-even prices, and the **implied breakeven daily charter rate** (`$/day`) that makes a trader indifferent between sending a cargo to Europe or Asia. Includes six deterministic stress scenarios (European winter cold snap, Asian demand surge, US freeze-off, global oversupply, and Panama/Red Sea canal disruptions).

### 4. Dynamic Slow-Steaming Optimizer (`src/optimization/`)
Uses the cubic propulsion power law ($E_{\text{prop}} \propto v^3$) to grid-search laden and ballast speeds between `13.0` and `21.0` knots, finding the speed profile that maximizes daily net profit after accounting for extra charter hire days and working capital lock-up.

### 5. Historical Backtest & Regime Risk Analytics (`src/backtesting/`, `src/risk/`)
Simulates daily cargo economics across **2,229 complete three-basin trading days** (2017–2026) and breaks down performance, 95% Value-at-Risk (VaR), Expected Shortfall (CVaR), and the embedded value of geographic diversion flexibility across four distinct market regimes:
1. **Pre-Crisis Baseline (2017–2019)**
2. **COVID Demand Shock & US Shut-Ins (2020)**
3. **European Energy Crisis (2021–2022)**
4. **Market Rebalancing & New Supply Wave (2023–2026)**

---

## Project Layout

```text
lng_arbitrage_model/
├── config/
│   ├── data_config.toml            # Data sources, FX conversion & QC parameters
│   ├── physical_assumptions.toml   # Vessel specs, LNG grades, terminals & routes
│   ├── commercial_terms.toml       # SPA multiplier, tolling, canal & regas tariffs
│   ├── scenarios.toml              # Deterministic market & routing shock scenarios
│   ├── optimization_params.toml    # Slow-steaming speed bounds
│   └── backtesting_params.toml     # Historical simulation settings
├── src/
│   ├── data/                       # FRED & Bloomberg loaders, schema & QC engine
│   ├── physical/                   # Cargo specs, routing & 11-stage energy balance
│   ├── economics/                  # 5-tier cost taxonomy, pricer, break-evens & scenarios
│   ├── optimization/               # Cubic-law speed optimizer
│   ├── backtesting/                # Daily historical arbitrage simulator (2017-2026)
│   └── risk/                       # Regime decomposition, VaR/CVaR & 2D heatmaps
├── tests/                          # 16 automated pytest unit tests
└── outputs/
    ├── tables/                     # Generated CSV matrices and backtest tables
    └── charts/                     # Generated diagnostic PNG charts


## How to Run It

### 1. Environment Setup
Clone the repository, create a virtual environment, and install the core dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib openpyxl pyarrow requests pytest
```

*(Make sure the raw Bloomberg Excel exports are placed inside `data/raw/commercial/` as mapped in `config/data_config.toml`.)*

### 2. Verify the Test Suite
Run all 16 unit tests to confirm thermodynamic energy conservation, cost taxonomy separation, speed optimization, and tail-risk calculations:

```powershell
$env:PYTHONPATH="."; .venv\Scripts\python.exe -m pytest -p no:cacheprovider -v
```

### 3. Run the Analytical Pipeline
Execute the modules in sequence from the project root to build the market panel, run the physical engine, price cargoes, and generate the 2017–2026 historical backtest:

```powershell
# Phase 1: Ingest FRED Henry Hub spot & Bloomberg multi-basin futures/FX
$env:PYTHONPATH="."; .venv\Scripts\python.exe src/data/henry_hub_pipeline.py
$env:PYTHONPATH="."; .venv\Scripts\python.exe src/data/market_data_loader.py

# Phase 2: Compute 11-stage thermodynamic balances across vessels & routes
$env:PYTHONPATH="."; .venv\Scripts\python.exe src/physical/energy_balance.py

# Phase 3: Evaluate NWE vs. NEA netbacks, break-evens & stress scenarios
$env:PYTHONPATH="."; .venv\Scripts\python.exe src/economics/scenario_engine.py

# Phases 5 & 6: Run the 2017–2026 daily backtest, regime VaR/CVaR & sensitivity heatmaps
$env:PYTHONPATH="."; .venv\Scripts\python.exe src/backtesting/historical_engine.py
$env:PYTHONPATH="."; .venv\Scripts\python.exe src/risk/regime_analytics.py
```

All generated CSV tables and diagnostic PNG charts are exported automatically to `outputs/tables/` and `outputs/charts/`.