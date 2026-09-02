# -*- coding: utf-8 -*-
"""Hourly greenhouse dynamics and economic objective.

Controls: temperature setpoint, supplemental light, CO2 injection, RH setpoint.
States: LAI, dry matter, indoor CO2, humidity and soil water content.

References: van Henten (1994); Jones et al. (1991); Boulard & Baille (1993);
Stanghellini (1987); ASHRAE Handbook (2019); de Zwart (1996)."""
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

# PHYSICAL CONSTANTS


# Psychrometric constants
RHO_AIR = 1.204        # Air density at 20C (kg/m3)
CP_AIR = 1005.0        # Specific heat of air (J/kg-K)
LATENT_HEAT = 2.45e6   # Latent heat of vaporization (J/kg)
STEFAN_BOLTZMANN = 5.67e-8  # (W/m2-K4)

# Crop constants (tomato default, from van Henten 1994)
P_MAX = 5.0            # Max leaf photosynthesis (g DM / m2 leaf / h)  — calibrated for Dutch winter Hemming (2019)
SLA = 0.030             # Specific leaf area (m2/g DM) — winter-adapted tomato
LAI_MAX = 5.0           # Maximum LAI
MAINT_RESP = 0.002      # Maintenance respiration (g/g DM/day at 25C) — calibrated for Hemming (2019) winter
GROWTH_RESP = 0.25      # Growth respiration fraction
Q10 = 2.0               # Temperature sensitivity

# CO2 parameters
CO2_STOMATAL_CONDUCTANCE = 0.002  # m/s
GREENHOUSE_VOLUME = 4.0    # m3 per m2 floor (typical 4m height)
GREENHOUSE_HEIGHT = 4.0    # m
CO2_MOLAR_MASS = 44.01     # g/mol
PPM_TO_G_M3 = 1.83e-3      # 1 ppm CO2 = 1.83e-3 g/m3 at 20C
CO2_INJ_SCALE = 0.5       # ppm CO2 rise per injection unit per hour
CO2_UPTAKE_SCALE = 1.5     # g CO2 / m2 / h at P_rate=1, LAI=1
CO2_INITIAL = 600.0        # Initial greenhouse CO2 (ppm) - typical enriched

# Ventilation parameters (Boulard & Baille 1993)
VENT_COEFF_WIND = 0.05     # Wind-driven ventilation coefficient
VENT_COEFF_BUOYANCY = 0.02 # Buoyancy-driven coefficient (m/s-K^0.5)
VENT_LEAKAGE = 0.03         # Base leakage rate (1/h)

# Transpiration (Stanghellini 1987)
TRANS_COEFF_RAD = 0.6      # Radiation-driven transpiration factor (g/J)
TRANS_COEFF_VPD = 25.0     # VPD-driven transpiration (g/m2-h-kPa)

# HVAC COP curves (heat pump)
COP_BASE = 4.0             # Base COP at deltaT=5C
COP_DEGRADATION = 0.08     # COP loss per degree deltaT above 5C

# CONFIGURATION


@dataclass
class GreenhouseConfig:
    """Enhanced greenhouse environment configuration."""

    # ---- Crop physiology ----
    T_opt: float = 25.0
    T_min: float = 10.0
    T_max: float = 40.0
    T_sigma: float = 5.0
    L_sat: float = 400.0
    L_half: float = 120.0
    C_ref: float = 400.0
    C_sat: float = 1000.0
    H_opt: float = 70.0
    H_sigma: float = 15.0

    # Crop growth parameters
    P_max_ref: float = P_MAX
    LAI_initial: float = 2.5       # Initial LAI
    DM_initial: float = 100.0  # Consistent: leafDM=LAI/SLA=100g, total=100*(1/f_leaf)=250g approximated      # Initial dry matter (g/m2)
    lai_growth_rate: float = 0.08  # LAI growth per day at optimal T

    # ---- Energy cost (per control unit, calibrated to kWh) ----
    cost_heating: float = 0.008    # kWh per (C * m2) heating
    cost_cooling: float = 0.010    # kWh per (C * m2) cooling
    cost_lighting: float = 0.001   # kWh per W/m2 supplemental light
    cost_co2: float = 0.003        # kWh per unit CO2 injection
    cost_humidification: float = 0.002  # kWh per % humidity adjustment

    # ---- Outdoor climate ----
    T_out_mean: float = 27.0       # Singapore mean
    T_out_amp: float = 3.5
    RH_out: float = 83.0
    CO2_out: float = 420.0         # Ambient CO2 (ppm)
    wind_speed: float = 2.0        # Average wind speed (m/s)

    # ---- Hard safety bounds ----
    T_lower: float = 15.0
    T_upper: float = 38.0
    L_upper: float = 800.0
    C_lower: float = 300.0
    C_upper: float = 1800.0
    H_lower: float = 30.0
    H_upper: float = 95.0

    # ---- Rate-of-change soft limits ----
    max_dT: float = 6.0
    max_dL: float = 400.0       # Allow natural day/night light transitions
    max_dC: float = 500.0        # Allow CO2 enrichment ramping
    max_dH: float = 20.0

    # ---- Penalty weights ----
    lambda_bound: float = 500.0
    lambda_rate: float = 500.0
    humidity_tracking: float = 0.5  # Declared assumption: manuscript gives no kappa_H value.
    lambda_yield_weight: float = 1.0
    lambda_energy_weight: float = 1.0

    # ---- Time discretisation ----
    T_steps: int = 24

    # ---- Time-of-Use pricing (SGD/kWh, Singapore SP Group rates) ----
    tou_peak_hours: tuple = (10, 18)
    tou_mid_hours: tuple = (7, 22)
    price_peak: float = 0.30       # Peak rate SGD/kWh
    price_mid: float = 0.22         # Mid-peak
    price_offpeak: float = 0.16     # Off-peak

    # ---- Weather variability ----
    cloud_cover: float = 0.40       # Singapore average cloud cover
    cloud_variability: float = 0.15
    weather_seed: int = 0

    # ---- Model fidelity flags ----
    use_crop_dynamics: bool = True     # Enable LAI/DM state dynamics
    use_co2_balance: bool = True       # Enable CO2 mass balance
    use_ventilation: bool = True       # Enable T-H coupling via ventilation
    use_transpiration: bool = True     # Enable plant transpiration feedback
    use_hvac_cop: bool = True          # Use non-linear COP

    # ---- Water balance (soil moisture dynamics, Allen et al. 1998 FAO-56) ----
    use_water_balance: bool = True        # Enable soil water dynamics
    SWC_initial: float = 0.30             # Initial soil water content (m3/m3)
    SWC_field_capacity: float = 0.35      # Field capacity (m3/m3)
    SWC_wilting_point: float = 0.12       # Permanent wilting point (m3/m3)
    SWC_opt: float = 0.28                 # Optimal SWC for transpiration (m3/m3)
    soil_depth: float = 0.30              # Effective root zone depth (m)
    drainage_coeff: float = 0.55          # Drainage fraction above field capacity
    water_cost_per_m3: float = 0.28       # Water cost (SGD/m3, Singapore PUB rate)

    # ---- Economic profit model ----
    use_economic_model: bool = True        # Enable economic profit calculation
    crop_market_price: float = 3.50        # Tomato wholesale price (SGD/kg)
    fixed_cost_per_day: float = 0.05       # Fixed infrastructure cost (SGD/m2/day)

# ENHANCED GREENHOUSE ENVIRONMENT


class GreenhouseEnv:
    """Enhanced greenhouse simulator with coupled physics and crop dynamics.

    Parameters
    ----------
    config : GreenhouseConfig
    seed : int
    """

    def __init__(self, config=None, seed=42):
        self.config = config or GreenhouseConfig()
        self.rng = np.random.default_rng(seed)
        self._T_out = self._build_outdoor_profile()
        self._tou_prices = self._build_tou_prices()
        self._cloud = self._build_cloud_profile()
        self._solar = self._build_solar_profile()
        self._RH_out = np.full(self.config.T_steps, self.config.RH_out)
        self._co2_injection_scale = np.ones(self.config.T_steps)

    # Climate profiles


    def _build_outdoor_profile(self):
        cfg = self.config
        hours = np.arange(cfg.T_steps)
        T = cfg.T_out_mean + cfg.T_out_amp * np.sin(2 * np.pi * (hours - 8) / 24)
        return T

    def _build_tou_prices(self):
        cfg = self.config
        hours = np.arange(cfg.T_steps) % 24
        prices = np.full(cfg.T_steps, cfg.price_offpeak)
        peak_mask = (hours >= cfg.tou_peak_hours[0]) & (hours < cfg.tou_peak_hours[1])
        mid_mask = (hours >= cfg.tou_mid_hours[0]) & (hours < cfg.tou_mid_hours[1]) & ~peak_mask
        prices[peak_mask] = cfg.price_peak
        prices[mid_mask] = cfg.price_mid
        return prices

    def _build_cloud_profile(self):
        cfg = self.config
        hours = np.arange(cfg.T_steps)
        base = cfg.cloud_cover + cfg.cloud_variability * np.sin(2 * np.pi * (hours - 6) / 12)
        noise = self.rng.normal(0, cfg.cloud_variability * 0.5, cfg.T_steps)
        return np.clip(base + noise, 0.0, 1.0)

    def _build_solar_profile(self):
        cfg = self.config
        hours = np.arange(cfg.T_steps) % 24
        solar_clear = 800.0 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
        solar = solar_clear * (1.0 - self._cloud * 0.7)
        return solar

    @property
    def T_out(self): return self._T_out.copy()
    @property
    def tou_prices(self): return self._tou_prices.copy()
    @property
    def cloud_profile(self): return self._cloud.copy()
    @property
    def solar_profile(self): return self._solar.copy()

    # PSYCHROMETRIC UTILITIES


    @staticmethod
    def saturation_vapor_pressure(T_celsius):
        """Tetens equation. Returns kPa."""
        return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))

    @staticmethod
    def rh_to_ah(RH, T_celsius):
        """Relative humidity (%) -> Absolute humidity (g/m3)."""
        es = GreenhouseEnv.saturation_vapor_pressure(T_celsius)  # kPa
        ea = es * RH / 100.0
        return 2.1667 * ea / (T_celsius + 273.15) * 1000.0  # g/m3

    @staticmethod
    def ah_to_rh(AH, T_celsius):
        """Absolute humidity (g/m3) -> Relative humidity (%)."""
        es = GreenhouseEnv.saturation_vapor_pressure(T_celsius)
        ea = AH / 1000.0 * (T_celsius + 273.15) / 2.1667
        return np.clip(ea / es * 100.0, 0.0, 100.0)

    @staticmethod
    def vpd(T_celsius, RH):
        """Vapor Pressure Deficit (kPa)."""
        es = GreenhouseEnv.saturation_vapor_pressure(T_celsius)
        return es * (1.0 - RH / 100.0)

    # CROP PHYSIOLOGY


    def temperature_response(self, T):
        """Gaussian temperature response for photosynthesis."""
        cfg = self.config
        return np.exp(-0.5 * ((T - cfg.T_opt) / cfg.T_sigma) ** 2)

    def light_response(self, L):
        """Michaelis-Menten light response."""
        cfg = self.config
        L_eff = np.maximum(L, 0)
        return L_eff / (L_eff + cfg.L_half)

    def co2_response(self, C):
        """Manuscript Michaelis-Menten response C / (C + C_sat)."""
        cfg = self.config
        C_eff = np.maximum(C, 0.0)
        return C_eff / (C_eff + cfg.C_sat)

    def humidity_response(self, H):
        """Gaussian humidity response."""
        cfg = self.config
        return np.exp(-0.5 * ((H - cfg.H_opt) / cfg.H_sigma) ** 2)

    def photosynthesis_rate(self, T, PAR, CO2, H):
        """Coupled photosynthesis rate (gross, arbitrary units)."""
        fT = self.temperature_response(T)
        fL = self.light_response(PAR)
        fC = self.co2_response(CO2)
        fH = self.humidity_response(H)
        return fT * fL * fC * fH
    def water_stress_response(self, SWC):
        """Water stress factor (0-1) based on soil water content.

        FAO-56 style: linear reduction below optimal, zero at wilting point.
        """
        cfg = self.config
        if not cfg.use_water_balance:
            return 1.0
        if SWC >= cfg.SWC_opt:
            return 1.0
        if SWC <= cfg.SWC_wilting_point:
            return 0.05  # minimal survival
        return (SWC - cfg.SWC_wilting_point) / (cfg.SWC_opt - cfg.SWC_wilting_point)

    def irrigation_demand(self, trans_rate, SWC):
        """Compute irrigation needed to maintain optimal soil moisture."""
        cfg = self.config
        if not cfg.use_water_balance:
            return 0.0
        # Convert transpiration (g/m2-h) to water depth (mm/h)
        trans_mm_h = trans_rate / 1000.0  # 1 g/m2 = 0.001 mm
        # Irrigation = transpiration + deficit correction
        deficit_mm = max(0.0, (cfg.SWC_opt - SWC) * cfg.soil_depth * 1000.0)
        return trans_mm_h + deficit_mm * 0.15  # smooth deficit correction


    def crop_uptake_co2(self, P_rate, LAI):
        """CO2 consumed by crop (ppm per hour for greenhouse volume)."""
        cfg = self.config
        uptake_g_m2_h = P_rate * LAI * 5.0  # ~5 g CO2 / m2 / h at P=1
        delta_ppm = uptake_g_m2_h / (PPM_TO_G_M3 * GREENHOUSE_VOLUME)
        return delta_ppm

    # VENTILATION & COUPLING


    def ventilation_rate(self, T_in, T_out):
        """Natural ventilation rate (air changes per hour) via roof vents.

        Boulard & Baille (1993): combined wind + buoyancy driven.
        """
        cfg = self.config
        wind_component = VENT_COEFF_WIND * cfg.wind_speed
        dT = max(0.0, T_in - T_out)  # Buoyancy only when T_in > T_out
        buoyancy_component = VENT_COEFF_BUOYANCY * np.sqrt(dT)
        return max(VENT_LEAKAGE, wind_component + buoyancy_component)

    def co2_balance(self, CO2_inj, CO2_in, T_in, T_out, P_rate, LAI):
        """Update indoor CO2 concentration given injection and losses.

        Returns: new CO2_in (ppm)
        """
        cfg = self.config
        VR = self.ventilation_rate(T_in, T_out) if cfg.use_ventilation else VENT_LEAKAGE

        # Injection contribution (converts injection rate to ppm delta)
        delta_injection = CO2_inj  # ppm per step, as in the manuscript.

        # Leakage (driven by concentration difference)
        delta_leakage = VR * CO2_in

        # Crop uptake
        delta_uptake = self.crop_uptake_co2(P_rate, LAI) if cfg.use_transpiration else 0.0

        new_co2 = CO2_in + delta_injection - delta_leakage - delta_uptake
        return new_co2  # Diagnostic state; do not silently clip to control bounds.

    def greenhouse_energy_balance(self, T_sp, T_in, T_out, L_supp, H_sp, H_in, LAI, solar):
        """Compute energy needed to maintain setpoints given ventilation losses."""
        cfg = self.config
        VR = self.ventilation_rate(T_in, T_out) if cfg.use_ventilation else VENT_LEAKAGE

        # Sensible heat loss through ventilation
        Q_vent = RHO_AIR * CP_AIR * VR / 3600.0 * GREENHOUSE_VOLUME * (T_in - T_out)

        # Solar gain (reduces heating need)
        Q_solar = solar * 0.5  # 50% transmissivity

        # Heating/cooling needed
        delta_T = T_sp - T_in
        thermal_mass = 20000.0  # J/m2-K (approximate)
        Q_needed = thermal_mass * delta_T + Q_vent - Q_solar - L_supp * 0.3

        # COP curve
        if cfg.use_hvac_cop:
            dT_actual = abs(T_sp - T_out)
            cop = max(1.5, COP_BASE - COP_DEGRADATION * max(0, dT_actual - 5))
        else:
            cop = COP_BASE

        if Q_needed > 0:
            energy_heating = Q_needed / cop / 3.6e6  # Convert J to kWh
            energy_cooling = 0.0
        else:
            energy_heating = 0.0
            energy_cooling = abs(Q_needed) / cop / 3.6e6

        # Humidification/dehumidification energy
        delta_AH = GreenhouseEnv.rh_to_ah(H_sp, T_sp) - H_in if cfg.use_ventilation else 0
        energy_humidity = abs(delta_AH) * LATENT_HEAT * GREENHOUSE_VOLUME / 3.6e6 * 0.01

        return energy_heating, energy_cooling, energy_humidity

    def transpiration_rate(self, PAR, VPD, LAI):
        """Stanghellini transpiration model (g/m2-h)."""
        cfg = self.config
        if not cfg.use_transpiration or LAI <= 0:
            return 0.0
        trans_rad = TRANS_COEFF_RAD * PAR * LAI
        trans_vpd = TRANS_COEFF_VPD * VPD * LAI
        return trans_rad + trans_vpd

    # CROP STATE DYNAMICS


    def update_crop_state(self, P_rate, T, LAI, DM):
        """Update crop state (LAI, dry matter) for one timestep.

        Based on van Henten (1994) simplified tomato model.
        """
        cfg = self.config
        if not cfg.use_crop_dynamics:
            return LAI, DM, 0.0

        dt = 1.0 / 24.0  # 1 hour in days

        # Gross photosynthesis: P_MAX = 2.5 g DM / m2 / day (van Henten 1994)
        # Convert daily rate -> hourly for this 1-hour timestep
        P_gross_hourly = P_rate * LAI * P_MAX * dt

        # Maintenance respiration (temperature dependent)
        # MAINT_RESP = 0.003 g/g DM / day -> convert to hourly
        R_maint_hourly = MAINT_RESP * DM * Q10 ** ((T - 25.0) / 10.0) * dt

        # Net available for growth (hourly)
        P_net_hourly = max(0.0, P_gross_hourly - R_maint_hourly)

        # Partition to leaves vs fruit (simplified: constant fraction)
        f_leaf = 0.4 if LAI < LAI_MAX * 0.7 else 0.2

        # Update states: hourly accumulation, no extra dt
        dDM_leaf = P_net_hourly * f_leaf * (1.0 - GROWTH_RESP)
        dDM_total = P_net_hourly * (1.0 - GROWTH_RESP)

        new_LAI = min(LAI_MAX, LAI + dDM_leaf * SLA)
        new_DM = DM + dDM_total

        # Harvestable yield increment (non-leaf portion)
        harvest_increment = P_net_hourly * (1.0 - f_leaf) * (1.0 - GROWTH_RESP)

        return new_LAI, new_DM, harvest_increment

    # PENALTY BRIDGE


    def calculate_penalty(self, x, x_prev=None):
        """Smooth quadratic penalty (kept from v1 with enhancements)."""
        cfg = self.config
        x = self._reshape(x)

        penalty = 0.0
        breakdown = {"bound": 0.0, "rate": 0.0, "co2": 0.0, "humidity": 0.0}

        for t in range(cfg.T_steps):
            T, L, C, H = x[t]

            # Hard bound violations
            p_T_lo = max(0.0, cfg.T_lower - T)
            p_T_hi = max(0.0, T - cfg.T_upper)
            p_L = max(0.0, L - cfg.L_upper)
            p_L_lo = max(0.0, -L)
            p_C_lo = max(0.0, cfg.C_lower - C)
            p_C_hi = max(0.0, C - cfg.C_upper)
            p_H_lo = max(0.0, cfg.H_lower - H)
            p_H_hi = max(0.0, H - cfg.H_upper)

            bound_p = (
                (p_T_lo**2 + p_T_hi**2)/(cfg.T_upper-cfg.T_lower)**2
                + (p_L**2 + p_L_lo**2)/cfg.L_upper**2
                + (p_C_lo**2 + p_C_hi**2)/(cfg.C_upper-cfg.C_lower)**2
                + (p_H_lo**2 + p_H_hi**2)/(cfg.H_upper-cfg.H_lower)**2
            )
            penalty += cfg.lambda_bound * bound_p
            breakdown["bound"] += cfg.lambda_bound * bound_p

            # Rate-of-change violations
            if x_prev is not None or t > 0:
                x_prev_t = x_prev[t] if x_prev is not None else x[t-1]
                dT = abs(T - x_prev_t[0])
                dL = abs(L - x_prev_t[1])
                dC = abs(C - x_prev_t[2])
                dH = abs(H - x_prev_t[3])

                rate_p = (
                    (max(0.0, dT - cfg.max_dT)/(cfg.T_upper-cfg.T_lower))**2 +
                    (max(0.0, dL - cfg.max_dL)/cfg.L_upper)**2 +
                    (max(0.0, dC - cfg.max_dC)/(cfg.C_upper-cfg.C_lower))**2 +
                    (max(0.0, dH - cfg.max_dH)/(cfg.H_upper-cfg.H_lower))**2
                )
                penalty += cfg.lambda_rate * rate_p
                breakdown["rate"] += cfg.lambda_rate * rate_p

        return penalty, breakdown

    # FITNESS EVALUATION (main entry point)


    def fitness(self, x):
        """Evaluate a control vector, returning (fitness, details).

        This is the main API - with optional crop dynamics tracking.
        """
        cfg = self.config
        x = self._reshape(x)
        if not np.all(np.isfinite(x)):
            raise ValueError("controls must be finite")

        # Initialize state variables
        LAI = cfg.LAI_initial
        DM = cfg.DM_initial
        CO2_in = CO2_INITIAL
        AH_in = GreenhouseEnv.rh_to_ah(self._RH_out[0], cfg.T_out_mean)
        SWC = cfg.SWC_initial   # Soil water content (m3/m3)
        RH_state = float(self._RH_out[0])

        total_yield = 0.0
        total_energy = 0.0
        total_water = 0.0       # Total irrigation water (L/m2)
        yield_hourly = np.zeros(cfg.T_steps)
        energy_hourly = np.zeros(cfg.T_steps)
        water_hourly = np.zeros(cfg.T_steps)
        co2_realized = np.zeros(cfg.T_steps)
        rh_realized = np.zeros(cfg.T_steps)
        swc_hourly = np.zeros(cfg.T_steps)
        lai_hourly = np.zeros(cfg.T_steps)
        dry_matter_hourly = np.zeros(cfg.T_steps)
        absolute_humidity_hourly = np.zeros(cfg.T_steps)
        humidity_disturbance_hourly = np.zeros(cfg.T_steps)
        irrigation_swc_hourly = np.zeros(cfg.T_steps)
        evapotranspiration_swc_hourly = np.zeros(cfg.T_steps)

        for t in range(cfg.T_steps):
            T_sp, L_supp, CO2_inj, RH_sp = x[t]
            CO2_inj_effective = CO2_inj * self._co2_injection_scale[t]

            # Get outdoor conditions
            T_out_t = self._T_out[t]
            solar_t = self._solar[t]
            price_t = self._tou_prices[t]

            # Total PAR (solar + supplemental)
            PAR_total = solar_t * 0.5 + L_supp  # ~50% of solar is PAR

            # --- CO2 balance ---
            if cfg.use_co2_balance:
                # Temporary photosynthesis estimate for CO2 uptake
                P_temp = self.photosynthesis_rate(T_sp, PAR_total, CO2_in, RH_state)
                CO2_in = self.co2_balance(CO2_inj_effective, CO2_in, T_sp, T_out_t, P_temp, LAI)
                # Use realized CO2 in photosynthesis
                CO2_eff = CO2_in
            else:
                CO2_eff = CO2_inj  # Direct setpoint control

            # --- Transpiration & humidity coupling ---
            if cfg.use_transpiration:
                VPD = GreenhouseEnv.vpd(T_sp, RH_state)
                trans_rate = self.transpiration_rate(PAR_total, VPD, LAI)
                # Add water vapor from transpiration
                AH_add = trans_rate / (GREENHOUSE_VOLUME * 1000.0)  # g/m3

                # Ventilation humidity exchange
                if cfg.use_ventilation:
                    VR = self.ventilation_rate(T_sp, T_out_t)
                    AH_out = GreenhouseEnv.rh_to_ah(self._RH_out[t], T_out_t)
                    AH_leak = VR * (AH_in - AH_out)
                    AH_in = AH_in + AH_add - AH_leak
                else:
                    AH_in = AH_in + AH_add

                # Convert to effective RH
                RH_uncontrolled = GreenhouseEnv.ah_to_rh(AH_in, T_sp)
                disturbance = RH_uncontrolled - RH_state
                RH_eff = np.clip(RH_state + cfg.humidity_tracking * (RH_sp - RH_state)
                                 + disturbance, 0.0, 100.0)
                humidity_disturbance_hourly[t] = disturbance
            else:
                RH_eff = RH_sp
                humidity_disturbance_hourly[t] = RH_sp - RH_state - cfg.humidity_tracking * (RH_sp - RH_state)

            RH_state = float(RH_eff)
            AH_in = GreenhouseEnv.rh_to_ah(RH_eff, T_sp)


            # --- Water balance (soil moisture dynamics) ---
            if cfg.use_water_balance:
                # Compute VPD for transpiration calcs
                VPD_w = GreenhouseEnv.vpd(T_sp, RH_eff) if cfg.use_transpiration else 1.0
                act_trans = self.transpiration_rate(PAR_total, VPD_w, LAI) if cfg.use_transpiration else 0.0
                # Irrigation computed to match transpiration + maintain SWC
                W_irrig = self.irrigation_demand(act_trans, SWC)
                # Soil water balance (FAO-56)
                trans_loss_mm = act_trans / 1000.0
                drainage = max(0.0, SWC - cfg.SWC_field_capacity) * cfg.drainage_coeff
                soil_water_mm = SWC * cfg.soil_depth * 1000.0
                # Eq. water: irrigation minus ET only. No hidden drainage/clipping.
                irrigation_swc_hourly[t] = W_irrig / (cfg.soil_depth * 1000.0)
                evapotranspiration_swc_hourly[t] = trans_loss_mm / (cfg.soil_depth * 1000.0)
                SWC += irrigation_swc_hourly[t] - evapotranspiration_swc_hourly[t]
            else:
                W_irrig = 0.0
            total_water += W_irrig
            water_hourly[t] = W_irrig
            # Water stress modifies photosynthesis
            fW = self.water_stress_response(SWC)

            # --- Photosynthesis & yield ---
            P_rate = self.photosynthesis_rate(T_sp, PAR_total, CO2_eff, RH_eff) * fW

            if cfg.use_crop_dynamics:
                # Crop state dynamics with harvest tracking
                LAI, DM, harvest = self.update_crop_state(P_rate, T_sp, LAI, DM)
                hourly_yield = harvest * 200.0   # gDM/m2 -> kg/ha equivalent (tuned)  # Scale: gDM/m2 -> kg/ha equivalent
            else:
                # Static yield without crop-state updates
                hourly_yield = P_rate * 25.0     # Scaled photosynthesis -> yield  # Scale factor

            total_yield += hourly_yield
            yield_hourly[t] = hourly_yield

            # --- Energy consumption ---
            if cfg.use_ventilation:
                Q_h, Q_c, Q_hum = self.greenhouse_energy_balance(
                    T_sp, T_sp, T_out_t, L_supp, RH_sp, AH_in, LAI, solar_t
                )
            else:
                dT = T_sp - T_out_t
                Q_h = max(0, dT) * cfg.cost_heating
                Q_c = max(0, -dT) * cfg.cost_cooling
                Q_hum = abs(RH_sp - self._RH_out[t]) * cfg.cost_humidification * 0.1

            energy_lighting = L_supp * cfg.cost_lighting
            energy_co2 = CO2_inj_effective * cfg.cost_co2 if not cfg.use_co2_balance else max(0, CO2_inj_effective) * cfg.cost_co2

            hourly_energy = (Q_h + Q_c + Q_hum + energy_lighting + energy_co2) * price_t
            total_energy += hourly_energy
            energy_hourly[t] = hourly_energy
            co2_realized[t] = CO2_in if cfg.use_co2_balance else CO2_eff
            rh_realized[t] = RH_eff
            swc_hourly[t] = SWC
            lai_hourly[t] = LAI
            dry_matter_hourly[t] = DM
            absolute_humidity_hourly[t] = AH_in

        # Penalty
        penalty, p_breakdown = self.calculate_penalty(x)

        # Economic profit (SGD/m2) - for agricultural audience
        if cfg.use_economic_model:
            # Convert yield from scaled units to kg/m2
            # total_yield is in scaled units (1 unit ~~ 0.01 kg/m2 based on calibration)
            yield_kg_m2 = total_yield * 0.04  # calibration: 100 yield units ~~ 4 kg/m2
            # Revenue
            revenue = yield_kg_m2 * cfg.crop_market_price
            # Costs
            energy_cost_sgd = total_energy  # already in SGD via price multiplication
            water_cost_sgd = total_water * cfg.water_cost_per_m3 * 0.001  # L/m2 -> m3/m2 -> SGD
            fixed_cost = cfg.fixed_cost_per_day * (cfg.T_steps / 24.0)
            total_cost = energy_cost_sgd + water_cost_sgd + fixed_cost
            # Profit
            profit = revenue - total_cost
            # Profit-based fitness (can be negative, penalty pushes down)
            fitness_val = profit - penalty
        else:
            # Weighted yield-minus-energy objective
            fitness_val = cfg.lambda_yield_weight * total_yield - cfg.lambda_energy_weight * total_energy - penalty

        # Compute derived economic metrics
        yield_kg_m2 = total_yield * 0.04
        revenue = yield_kg_m2 * cfg.crop_market_price
        WUE = total_yield / max(total_water, 0.001) if total_water > 0 else 0.0

        details = {
            "total_yield": float(total_yield),
            "total_energy": float(total_energy),
            "total_water": float(total_water),
            "total_penalty": float(penalty),
            "fitness": float(fitness_val),
            "is_feasible": penalty <= 1e-6,
            "final_LAI": float(LAI),
            "final_DM": float(DM),
            "final_CO2": float(CO2_in),
            "final_SWC": float(SWC),
            "yield_kg_m2": float(yield_kg_m2),
            "revenue_sgd": float(revenue),
            "profit_sgd": float(profit if cfg.use_economic_model else 0.0),
            "water_use_efficiency": float(WUE),
            "yield_hourly": yield_hourly,
            "energy_hourly": energy_hourly,
            "water_hourly": water_hourly,
            "co2_realized": co2_realized,
            "rh_realized": rh_realized,
            "soil_water_content_hourly": swc_hourly,
            "lai_hourly": lai_hourly,
            "dry_matter_hourly": dry_matter_hourly,
            "absolute_humidity_hourly": absolute_humidity_hourly,
            "humidity_disturbance_hourly": humidity_disturbance_hourly,
            "irrigation_swc_hourly": irrigation_swc_hourly,
            "evapotranspiration_swc_hourly": evapotranspiration_swc_hourly,
            "setpoint_temperature_hourly": x[:, 0].copy(),
            "supplemental_light_hourly": x[:, 1].copy(),
            "co2_request_hourly": x[:, 2].copy(),
            "humidity_setpoint_hourly": x[:, 3].copy(),
            "penalty_breakdown": p_breakdown,
        }
        return fitness_val, details

    # UTILITY METHODS


    def _reshape(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        expected = self.config.T_steps * 4
        if x.size == expected and x.shape != (self.config.T_steps, 4):
            x = x.reshape(self.config.T_steps, 4)
        if x.shape != (self.config.T_steps, 4):
            raise ValueError(f'Expected {self.config.T_steps}x4 controls, got {x.shape}')
        return x

    def random_solution(self, smooth=True):
        cfg = self.config
        if smooth:
            # Generate smooth random trajectories via keypoint interpolation
            n_keys = max(3, cfg.T_steps // 4)
            key_hours = np.linspace(0, cfg.T_steps - 1, n_keys, dtype=int)
            key_hours[-1] = cfg.T_steps - 1
            T_keys = self.rng.uniform(cfg.T_lower + 3, cfg.T_upper - 3, n_keys)
            L_keys = self.rng.uniform(0, cfg.L_upper * 0.5, n_keys)
            C_keys = self.rng.uniform(cfg.C_lower + 100, cfg.C_upper - 300, n_keys)
            H_keys = self.rng.uniform(cfg.H_lower + 15, cfg.H_upper - 10, n_keys)
            hours = np.arange(cfg.T_steps)
            T = np.interp(hours, key_hours, T_keys)
            L = np.interp(hours, key_hours, L_keys)
            C = np.interp(hours, key_hours, C_keys)
            H = np.interp(hours, key_hours, H_keys)
        else:
            T = self.rng.uniform(cfg.T_lower + 2, cfg.T_upper - 2, cfg.T_steps)
            L = self.rng.uniform(0, cfg.L_upper * 0.6, cfg.T_steps)
            C = self.rng.uniform(cfg.C_lower + 50, cfg.C_upper - 200, cfg.T_steps)
            H = self.rng.uniform(cfg.H_lower + 10, cfg.H_upper - 10, cfg.T_steps)
        return np.column_stack([T, L, C, H])

    def bounds(self):
        cfg = self.config
        low = np.array([cfg.T_lower, 0.0, cfg.C_lower, cfg.H_lower])
        high = np.array([cfg.T_upper, cfg.L_upper, cfg.C_upper, cfg.H_upper])
        return np.tile(low, cfg.T_steps), np.tile(high, cfg.T_steps)

    @property
    def n_vars(self): return self.config.T_steps * 4

    @property
    def n_steps(self): return self.config.T_steps

    def summary(self, x):
        x = self._reshape(x)
        _, details = self.fitness(x)
        n_days = self.config.T_steps // 24
        horizon = f"{n_days}-day ({self.config.T_steps}-hour)"
        lines = [
            "=" * 70,
            f"  Enhanced Greenhouse Control Plan ({horizon})",
            "=" * 70,
            f"  Hour    Tsp(C)   Lsup(W)   CO2inj    RHsp(%)",
            "  " + "-" * 52,
        ]
        for t in range(min(6, self.config.T_steps)):
            lines.append(f"  {t:3d}    {x[t,0]:6.1f}   {x[t,1]:7.1f}   {x[t,2]:7.0f}   {x[t,3]:5.0f}")
        if self.config.T_steps > 10:
            lines.append("  ...")
            for t in range(max(10, self.config.T_steps - 4), self.config.T_steps):
                lines.append(f"  {t:3d}    {x[t,0]:6.1f}   {x[t,1]:7.1f}   {x[t,2]:7.0f}   {x[t,3]:5.0f}")
        lines.append("  " + "-" * 52)
        lines.append(f"  Total Yield:     {details['total_yield']:12.4f}")
        lines.append(f"  Total Energy:    {details['total_energy']:12.4f} kWh")
        lines.append(f"  Total Penalty:   {details['total_penalty']:12.4f}")
        lines.append(f"  Fitness:         {details['fitness']:12.4f}")
        lines.append(f"  Final LAI:       {details['final_LAI']:12.4f}")
        lines.append(f"  Final DM:        {details['final_DM']:12.2f} g/m2")
        tw = details["total_water"]
        wue = details["water_use_efficiency"]
        swc = details["final_SWC"]
        ykg = details["yield_kg_m2"]
        rev = details["revenue_sgd"]
        prof = details["profit_sgd"]
        lines.append(f"  Total Water:     {tw:12.2f} L/m2")
        lines.append(f"  Water Use Eff:   {wue:12.4f}")
        lines.append(f"  Final SWC:       {swc:12.4f} m3/m3")
        lines.append(f"  Yield (kg/m2):   {ykg:12.4f}")
        lines.append(f"  Revenue:         {rev:12.4f} SGD")
        lines.append(f"  Profit:          {prof:12.4f} SGD/m2")
        lines.append(f"  Feasible:        {str(details['is_feasible']):>12}")
        lines.append("=" * 70)
        return "\n".join(lines)

# SELF-TEST


if __name__ == "__main__":
    print("Enhanced GreenhouseEnv v2.0 - Self Test\n")

    # Test 1: Basic static mode (backward compatible)
    env = GreenhouseEnv(GreenhouseConfig(T_steps=24, use_crop_dynamics=False, use_co2_balance=False, use_ventilation=False, use_transpiration=False), seed=42)
    x = env.random_solution()
    f, d = env.fitness(x)
    print(f"Static mode: fitness={f:.2f}, yield={d['total_yield']:.3f}, energy={d['total_energy']:.1f}, penalty={d['total_penalty']:.2f}")

    # Test 2: Full physics mode
    env2 = GreenhouseEnv(GreenhouseConfig(T_steps=24, use_crop_dynamics=True, use_co2_balance=True, use_ventilation=True, use_transpiration=True), seed=42)
    x2 = env2.random_solution()
    f2, d2 = env2.fitness(x2)
    print(f"Full physics: fitness={f2:.2f}, yield={d2['total_yield']:.3f}, energy={d2['total_energy']:.1f}, penalty={d2['total_penalty']:.2f}")
    print(f"  LAI={d2['final_LAI']:.3f}, DM={d2['final_DM']:.1f}, CO2={d2['final_CO2']:.0f}ppm")

    # Test 3: 7-day with crop growth
    env7 = GreenhouseEnv(GreenhouseConfig(T_steps=168), seed=42)
    x7 = env7.random_solution()
    f7, d7 = env7.fitness(x7)
    print(f"7-day: fitness={f7:.2f}, yield={d7['total_yield']:.3f}, LAI={d7['final_LAI']:.3f}, DM={d7['final_DM']:.1f}")

    print("\n" + env2.summary(x2))
