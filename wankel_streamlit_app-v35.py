# -*- coding: utf-8 -*-
"""
Wankel Rotary Engine 0D Thermodynamic Simulator - Version 35
Specialized Research & Engineering Platform
Includes:
- Default Experimental MAP vs Throttle curve digitized from photo_2026-09-12_02-12-11.jpg
- Full Garrett Catalog Digitized Compressor & Turbine Maps (G-Series, GTX Gen II, GT-Series)
- Robust Parametric Sweeps & Turbocharger Coupling without errors
- Two-Zone Combustion, Motoring/Firing Toggle, Lambda/Equivalent Ratio Control, Multi-Chamber Coupling
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global ISA atmosphere calculation helper
def get_isa_properties(alt_ft):
    alt_m = alt_ft * 0.3048
    if alt_m <= 11000.0:
        t_amb_k = 288.15 - 0.0065 * alt_m
        p_amb_pa = 101325.0 * ((t_amb_k / 288.15) ** 5.25588)
    elif alt_m <= 20000.0:
        t_amb_k = 216.65
        p_11 = 101325.0 * ((1.0 - 0.0065 * 11000.0 / 288.15) ** 5.25588)
        p_amb_pa = p_11 * np.exp(-9.80665 * (alt_m - 11000.0) / (287.05 * 216.65))
    else:
        t_amb_k = 216.65 + 0.001 * (alt_m - 20000.0)
        p_11 = 101325.0 * ((1.0 - 0.0065 * 11000.0 / 288.15) ** 5.25588)
        p_20 = p_11 * np.exp(-9.80665 * (20000.0 - 11000.0) / (287.05 * 216.65))
        p_amb_pa = p_20 * (216.65 / t_amb_k) ** 34.163
    p_ambient_bar = p_amb_pa / 1e5
    return p_ambient_bar, t_amb_k, p_amb_pa

# Helper to interpolate throttle ratio from lookup dictionary
def get_ratio_local(t_val, lut_dict):
    if not lut_dict:
        return 0.20 + 0.80 * ((t_val / 100.0) ** 1.5)
    sorted_keys = sorted(lut_dict.keys())
    if t_val <= sorted_keys[0]:
        return lut_dict[sorted_keys[0]]
    if t_val >= sorted_keys[-1]:
        return lut_dict[sorted_keys[-1]]
    for i in range(len(sorted_keys) - 1):
        k1 = sorted_keys[i]
        k2 = sorted_keys[i+1]
        if k1 <= t_val <= k2:
            r1 = lut_dict[k1]
            r2 = lut_dict[k2]
            return r1 + (r2 - r1) * (t_val - k1) / (k2 - k1)
    return 1.0

# Digitized Default Experimental Throttle Curve from photo_2026-09-12_02-12-11.jpg
EXPERIMENTAL_THROTTLE_LUT = {
    0.0: 0.200,
    10.0: 0.503,
    20.0: 0.671,
    30.0: 0.799,
    40.0: 0.878,
    50.0: 0.938,
    60.0: 0.967,
    70.0: 0.982,
    80.0: 0.987,
    90.0: 0.992,
    100.0: 1.000
}

# --- DIGITIZED GARRETT TURBOCHARGER CATALOG LIBRARY ---
TURBO_LIBRARY = {
    # --- GARRETT G-SERIES (Next-Gen Advanced Aero & Mar-M Wheel) ---
        # --- BORGWARNER EFR SERIES ---
    "BorgWarner EFR 6258": {
        "brand": "BorgWarner EFR",
        "inducer_mm": 49.6, "exducer_mm": 62.0, "trim": 64,
        "flow_limit_lb_min": 44.0, "flow_limit": 20.0,
        "N_max": 155000.0, "pr_c_max": 3.2, "eta_c": 0.77, "eta_t": 0.72,
        "turbine_inducer_mm": 58.0, "turbine_exducer_mm": 51.0, "turbine_trim": 77,
        "ar_options": {
            "0.64": {"m_t_limit": 13.0, "k_t": 1.4},
            "0.85": {"m_t_limit": 16.5, "k_t": 1.1}
        },
        "surge_coeff": 5.0,
        "speed_lines": [60000, 90000, 120000, 155000]
    },
    "BorgWarner EFR 7163": {
        "brand": "BorgWarner EFR",
        "inducer_mm": 57.0, "exducer_mm": 71.0, "trim": 64,
        "flow_limit_lb_min": 60.0, "flow_limit": 27.2,
        "N_max": 150000.0, "pr_c_max": 3.6, "eta_c": 0.78, "eta_t": 0.73,
        "turbine_inducer_mm": 63.0, "turbine_exducer_mm": 56.0, "turbine_trim": 79,
        "ar_options": {
            "0.80": {"m_t_limit": 18.0, "k_t": 1.2},
            "0.85": {"m_t_limit": 20.5, "k_t": 1.0}
        },
        "surge_coeff": 5.8,
        "speed_lines": [60000, 90000, 120000, 150000]
    },
    "BorgWarner EFR 8374": {
        "brand": "BorgWarner EFR",
        "inducer_mm": 62.0, "exducer_mm": 83.0, "trim": 56,
        "flow_limit_lb_min": 79.0, "flow_limit": 35.8,
        "N_max": 140000.0, "pr_c_max": 4.0, "eta_c": 0.79, "eta_t": 0.74,
        "turbine_inducer_mm": 74.0, "turbine_exducer_mm": 65.0, "turbine_trim": 77,
        "ar_options": {
            "0.92": {"m_t_limit": 25.0, "k_t": 1.1},
            "1.05": {"m_t_limit": 29.5, "k_t": 0.9}
        },
        "surge_coeff": 7.0,
        "speed_lines": [50000, 80000, 110000, 140000]
    },
    "BorgWarner EFR 9180": {
        "brand": "BorgWarner EFR",
        "inducer_mm": 67.5, "exducer_mm": 91.0, "trim": 55,
        "flow_limit_lb_min": 95.0, "flow_limit": 43.0,
        "N_max": 132000.0, "pr_c_max": 4.2, "eta_c": 0.79, "eta_t": 0.74,
        "turbine_inducer_mm": 80.0, "turbine_exducer_mm": 73.0, "turbine_trim": 83,
        "ar_options": {
            "0.92": {"m_t_limit": 28.0, "k_t": 1.0},
            "1.05": {"m_t_limit": 33.0, "k_t": 0.85},
            "1.45": {"m_t_limit": 39.0, "k_t": 0.7}
        },
        "surge_coeff": 8.5,
        "speed_lines": [50000, 80000, 110000, 132000]
    },
        "Custom Operator Map": {
        "brand": "Custom Operator",
        "inducer_mm": 55.0, "exducer_mm": 70.0, "trim": 62,
        "flow_limit_lb_min": 60.0, "flow_limit": 27.2,
        "N_max": 150000.0, "pr_c_max": 3.5, "eta_c": 0.78, "eta_t": 0.74,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 52.0, "turbine_trim": 75,
        "ar_options": {
            "Custom (0.80)": {"m_t_limit": 20.0, "k_t": 1.0}
        },
        "surge_coeff": 5.5,
        "speed_lines": [60000, 90000, 120000, 150000]
    },
    "Garrett G25-550": {
        "brand": "Garrett G-Series",
        "inducer_mm": 48.0, "exducer_mm": 60.0, "trim": 62,
        "flow_limit_lb_min": 49.0, "flow_limit": 22.2,
        "N_max": 165000.0, "pr_c_max": 3.5, "eta_c": 0.79, "eta_t": 0.74,
        "turbine_inducer_mm": 54.0, "turbine_exducer_mm": 49.0, "turbine_trim": 84,
        "ar_options": {
            "0.49": {"m_t_limit": 14.0, "k_t": 1.5},
            "0.72": {"m_t_limit": 18.0, "k_t": 1.1},
            "0.92": {"m_t_limit": 21.0, "k_t": 0.9}
        },
        "surge_coeff": 5.2,
        "speed_lines": [60000, 90000, 120000, 145000, 165000]
    },
    "Garrett G25-660": {
        "brand": "Garrett G-Series",
        "inducer_mm": 54.0, "exducer_mm": 67.0, "trim": 65,
        "flow_limit_lb_min": 60.0, "flow_limit": 27.2,
        "N_max": 165000.0, "pr_c_max": 3.8, "eta_c": 0.79, "eta_t": 0.74,
        "turbine_inducer_mm": 54.0, "turbine_exducer_mm": 49.0, "turbine_trim": 84,
        "ar_options": {
            "0.49": {"m_t_limit": 15.0, "k_t": 1.4},
            "0.72": {"m_t_limit": 19.5, "k_t": 1.1},
            "0.92": {"m_t_limit": 22.5, "k_t": 0.9}
        },
        "surge_coeff": 6.2,
        "speed_lines": [60000, 90000, 120000, 145000, 165000]
    },
    "Garrett G30-660": {
        "brand": "Garrett G-Series",
        "inducer_mm": 54.0, "exducer_mm": 67.0, "trim": 65,
        "flow_limit_lb_min": 60.0, "flow_limit": 27.2,
        "N_max": 150000.0, "pr_c_max": 3.6, "eta_c": 0.80, "eta_t": 0.75,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 55.0, "turbine_trim": 84,
        "ar_options": {
            "0.61": {"m_t_limit": 17.5, "k_t": 1.4},
            "0.83": {"m_t_limit": 22.0, "k_t": 1.1},
            "1.01": {"m_t_limit": 26.0, "k_t": 0.9}
        },
        "surge_coeff": 6.5,
        "speed_lines": [50000, 80000, 110000, 130000, 150000]
    },
    "Garrett G30-770": {
        "brand": "Garrett G-Series",
        "inducer_mm": 58.0, "exducer_mm": 71.0, "trim": 67,
        "flow_limit_lb_min": 70.0, "flow_limit": 31.8,
        "N_max": 150000.0, "pr_c_max": 3.9, "eta_c": 0.80, "eta_t": 0.75,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 55.0, "turbine_trim": 84,
        "ar_options": {
            "0.61": {"m_t_limit": 18.5, "k_t": 1.4},
            "0.83": {"m_t_limit": 23.5, "k_t": 1.1},
            "1.01": {"m_t_limit": 27.5, "k_t": 0.9}
        },
        "surge_coeff": 7.5,
        "speed_lines": [50000, 80000, 110000, 130000, 150000]
    },
    "Garrett G30-900": {
        "brand": "Garrett G-Series",
        "inducer_mm": 62.0, "exducer_mm": 76.0, "trim": 67,
        "flow_limit_lb_min": 80.0, "flow_limit": 36.3,
        "N_max": 150000.0, "pr_c_max": 4.2, "eta_c": 0.80, "eta_t": 0.75,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 55.0, "turbine_trim": 84,
        "ar_options": {
            "0.61": {"m_t_limit": 19.5, "k_t": 1.3},
            "0.83": {"m_t_limit": 25.0, "k_t": 1.0},
            "1.01": {"m_t_limit": 29.5, "k_t": 0.85}
        },
        "surge_coeff": 8.5,
        "speed_lines": [50000, 80000, 110000, 130000, 150000]
    },
    "Garrett G35-900": {
        "brand": "Garrett G-Series",
        "inducer_mm": 62.0, "exducer_mm": 76.0, "trim": 67,
        "flow_limit_lb_min": 80.0, "flow_limit": 36.3,
        "N_max": 135000.0, "pr_c_max": 4.1, "eta_c": 0.81, "eta_t": 0.76,
        "turbine_inducer_mm": 68.0, "turbine_exducer_mm": 62.0, "turbine_trim": 84,
        "ar_options": {
            "0.61": {"m_t_limit": 22.0, "k_t": 1.3},
            "0.83": {"m_t_limit": 28.0, "k_t": 1.0},
            "1.01": {"m_t_limit": 33.0, "k_t": 0.8}
        },
        "surge_coeff": 8.8,
        "speed_lines": [50000, 75000, 100000, 120000, 135000]
    },
    "Garrett G35-1050": {
        "brand": "Garrett G-Series",
        "inducer_mm": 68.0, "exducer_mm": 84.0, "trim": 65,
        "flow_limit_lb_min": 95.0, "flow_limit": 43.1,
        "N_max": 135000.0, "pr_c_max": 4.3, "eta_c": 0.81, "eta_t": 0.76,
        "turbine_inducer_mm": 68.0, "turbine_exducer_mm": 62.0, "turbine_trim": 84,
        "ar_options": {
            "0.61": {"m_t_limit": 23.5, "k_t": 1.2},
            "0.83": {"m_t_limit": 29.5, "k_t": 0.95},
            "1.01": {"m_t_limit": 35.0, "k_t": 0.75}
        },
        "surge_coeff": 10.0,
        "speed_lines": [50000, 75000, 100000, 120000, 135000]
    },
    "Garrett G40-1150": {
        "brand": "Garrett G-Series",
        "inducer_mm": 71.0, "exducer_mm": 88.0, "trim": 68,
        "flow_limit_lb_min": 105.0, "flow_limit": 47.6,
        "N_max": 115000.0, "pr_c_max": 4.4, "eta_c": 0.82, "eta_t": 0.76,
        "turbine_inducer_mm": 77.0, "turbine_exducer_mm": 70.0, "turbine_trim": 84,
        "ar_options": {
            "0.85": {"m_t_limit": 32.0, "k_t": 1.1},
            "1.06": {"m_t_limit": 39.5, "k_t": 0.85},
            "1.19": {"m_t_limit": 45.0, "k_t": 0.7}
        },
        "surge_coeff": 11.2,
        "speed_lines": [40000, 60000, 80000, 100000, 115000]
    },
    "Garrett G42-1200": {
        "brand": "Garrett G-Series",
        "inducer_mm": 73.0, "exducer_mm": 91.0, "trim": 64,
        "flow_limit_lb_min": 110.0, "flow_limit": 49.9,
        "N_max": 105000.0, "pr_c_max": 4.5, "eta_c": 0.82, "eta_t": 0.77,
        "turbine_inducer_mm": 82.0, "turbine_exducer_mm": 75.0, "turbine_trim": 84,
        "ar_options": {
            "1.01": {"m_t_limit": 38.0, "k_t": 0.95},
            "1.15": {"m_t_limit": 44.0, "k_t": 0.8},
            "1.28": {"m_t_limit": 50.0, "k_t": 0.7}
        },
        "surge_coeff": 13.0,
        "speed_lines": [40000, 60000, 80000, 95000, 105000]
    },

    # --- GARRETT GTX GEN II SERIES (Forged Billet Wheel) ---
    "Garrett GTX2860R Gen II": {
        "brand": "Garrett GTX Gen II",
        "inducer_mm": 46.0, "exducer_mm": 60.0, "trim": 58,
        "flow_limit_lb_min": 42.0, "flow_limit": 19.1,
        "N_max": 145000.0, "pr_c_max": 3.3, "eta_c": 0.75, "eta_t": 0.70,
        "turbine_inducer_mm": 54.0, "turbine_exducer_mm": 47.0, "turbine_trim": 76,
        "ar_options": {
            "0.64": {"m_t_limit": 15.0, "k_t": 1.4},
            "0.86": {"m_t_limit": 19.5, "k_t": 1.0}
        },
        "surge_coeff": 5.8,
        "speed_lines": [50000, 80000, 110000, 130000, 145000]
    },
    "Garrett GTX2867R Gen II": {
        "brand": "Garrett GTX Gen II",
        "inducer_mm": 50.0, "exducer_mm": 67.0, "trim": 55,
        "flow_limit_lb_min": 48.0, "flow_limit": 21.8,
        "N_max": 145000.0, "pr_c_max": 3.5, "eta_c": 0.76, "eta_t": 0.70,
        "turbine_inducer_mm": 54.0, "turbine_exducer_mm": 47.0, "turbine_trim": 76,
        "ar_options": {
            "0.64": {"m_t_limit": 16.5, "k_t": 1.3},
            "0.86": {"m_t_limit": 21.0, "k_t": 1.0}
        },
        "surge_coeff": 6.8,
        "speed_lines": [50000, 80000, 110000, 130000, 145000]
    },
    "Garrett GTX3071R Gen II": {
        "brand": "Garrett GTX Gen II",
        "inducer_mm": 54.0, "exducer_mm": 71.0, "trim": 58,
        "flow_limit_lb_min": 58.0, "flow_limit": 26.3,
        "N_max": 135000.0, "pr_c_max": 3.8, "eta_c": 0.77, "eta_t": 0.72,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 55.0, "turbine_trim": 84,
        "ar_options": {
            "0.63": {"m_t_limit": 20.0, "k_t": 1.3},
            "0.82": {"m_t_limit": 25.5, "k_t": 1.0},
            "1.06": {"m_t_limit": 31.0, "k_t": 0.8}
        },
        "surge_coeff": 8.0,
        "speed_lines": [50000, 75000, 95000, 115000, 135000]
    },
    "Garrett GTX3076R Gen II": {
        "brand": "Garrett GTX Gen II",
        "inducer_mm": 58.0, "exducer_mm": 76.0, "trim": 58,
        "flow_limit_lb_min": 65.0, "flow_limit": 29.5,
        "N_max": 135000.0, "pr_c_max": 4.0, "eta_c": 0.78, "eta_t": 0.72,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 55.0, "turbine_trim": 84,
        "ar_options": {
            "0.63": {"m_t_limit": 22.0, "k_t": 1.3},
            "0.82": {"m_t_limit": 27.5, "k_t": 1.0},
            "1.06": {"m_t_limit": 33.5, "k_t": 0.8}
        },
        "surge_coeff": 8.5,
        "speed_lines": [50000, 75000, 95000, 115000, 135000]
    },
    "Garrett GTX3582R Gen II": {
        "brand": "Garrett GTX Gen II",
        "inducer_mm": 66.0, "exducer_mm": 82.0, "trim": 64,
        "flow_limit_lb_min": 78.0, "flow_limit": 35.4,
        "N_max": 130000.0, "pr_c_max": 4.2, "eta_c": 0.79, "eta_t": 0.73,
        "turbine_inducer_mm": 68.0, "turbine_exducer_mm": 62.0, "turbine_trim": 84,
        "ar_options": {
            "0.63": {"m_t_limit": 25.0, "k_t": 1.2},
            "0.82": {"m_t_limit": 31.0, "k_t": 0.95},
            "1.06": {"m_t_limit": 38.0, "k_t": 0.75}
        },
        "surge_coeff": 10.0,
        "speed_lines": [50000, 75000, 95000, 115000, 130000]
    },

    # --- GARRETT GT-SERIES (Classic) ---
    "Garrett GT2860RS": {
        "brand": "Garrett GT Classic",
        "inducer_mm": 47.0, "exducer_mm": 60.0, "trim": 62,
        "flow_limit_lb_min": 35.0, "flow_limit": 15.9,
        "N_max": 145000.0, "pr_c_max": 2.9, "eta_c": 0.73, "eta_t": 0.68,
        "turbine_inducer_mm": 54.0, "turbine_exducer_mm": 47.0, "turbine_trim": 76,
        "ar_options": {
            "0.64": {"m_t_limit": 14.0, "k_t": 1.5},
            "0.86": {"m_t_limit": 18.0, "k_t": 1.1}
        },
        "surge_coeff": 5.5,
        "speed_lines": [50000, 80000, 110000, 130000, 145000]
    },
    "Garrett GT3076R": {
        "brand": "Garrett GT Classic",
        "inducer_mm": 57.0, "exducer_mm": 76.0, "trim": 56,
        "flow_limit_lb_min": 53.0, "flow_limit": 24.0,
        "N_max": 130000.0, "pr_c_max": 3.5, "eta_c": 0.76, "eta_t": 0.71,
        "turbine_inducer_mm": 60.0, "turbine_exducer_mm": 55.0, "turbine_trim": 84,
        "ar_options": {
            "0.63": {"m_t_limit": 21.0, "k_t": 1.3},
            "0.82": {"m_t_limit": 26.0, "k_t": 1.0},
            "1.06": {"m_t_limit": 31.0, "k_t": 0.80}
        },
        "surge_coeff": 7.8,
        "speed_lines": [50000, 75000, 95000, 115000, 130000]
    },
    "Garrett GT3582R": {
        "brand": "Garrett GT Classic",
        "inducer_mm": 61.0, "exducer_mm": 82.0, "trim": 56,
        "flow_limit_lb_min": 65.0, "flow_limit": 29.5,
        "N_max": 125000.0, "pr_c_max": 3.8, "eta_c": 0.77, "eta_t": 0.72,
        "turbine_inducer_mm": 68.0, "turbine_exducer_mm": 62.0, "turbine_trim": 84,
        "ar_options": {
            "0.63": {"m_t_limit": 24.0, "k_t": 1.2},
            "0.82": {"m_t_limit": 30.0, "k_t": 0.95},
            "1.06": {"m_t_limit": 36.0, "k_t": 0.75}
        },
        "surge_coeff": 9.0,
        "speed_lines": [50000, 75000, 95000, 110000, 125000]
    },

    
    # --- GARRETT GTX HEAVY / DRAG RACING SERIES (GTX47 / GTX50 / GTX55 & G50 / G55) ---
    "Garrett GTX4720R Gen II (85mm)": {
        "brand": "Garrett",
        "flow_limit": 95.0,
        "N_max": 95000.0,
        "pr_c_max": 4.8,
        "eta_c": 0.77,
        "eta_t": 0.73,
        "ar_options": {
            "1.08": {"m_t_limit": 52.0, "k_t": 0.80},
            "1.23": {"m_t_limit": 58.0, "k_t": 0.70}
        },
        "surge_coeff": 16.0,
        "speed_lines": [30000, 50000, 70000, 85000, 95000]
    },
    "Garrett GTX5020R Gen II (88mm)": {
        "brand": "Garrett",
        "flow_limit": 115.0,
        "N_max": 90000.0,
        "pr_c_max": 5.0,
        "eta_c": 0.78,
        "eta_t": 0.74,
        "ar_options": {
            "1.08": {"m_t_limit": 60.0, "k_t": 0.75},
            "1.23": {"m_t_limit": 68.0, "k_t": 0.65}
        },
        "surge_coeff": 18.0,
        "speed_lines": [30000, 50000, 70000, 80000, 90000]
    },
    "Garrett GTX5533R Gen II (94mm)": {
        "brand": "Garrett",
        "flow_limit": 135.0,
        "N_max": 85000.0,
        "pr_c_max": 5.2,
        "eta_c": 0.78,
        "eta_t": 0.74,
        "ar_options": {
            "1.00": {"m_t_limit": 65.0, "k_t": 0.70},
            "1.24": {"m_t_limit": 75.0, "k_t": 0.60},
            "1.40": {"m_t_limit": 85.0, "k_t": 0.50}
        },
        "surge_coeff": 22.0,
        "speed_lines": [25000, 45000, 65000, 78000, 85000]
    },
    "Garrett GTX5544R Gen II (102mm)": {
        "brand": "Garrett",
        "flow_limit": 155.0,
        "N_max": 80000.0,
        "pr_c_max": 5.5,
        "eta_c": 0.79,
        "eta_t": 0.75,
        "ar_options": {
            "1.12": {"m_t_limit": 78.0, "k_t": 0.60},
            "1.24": {"m_t_limit": 88.0, "k_t": 0.52},
            "1.40": {"m_t_limit": 98.0, "k_t": 0.45}
        },
        "surge_coeff": 25.0,
        "speed_lines": [25000, 40000, 60000, 72000, 80000]
    },
    "Garrett G-Series G50-1700 (80mm)": {
        "brand": "Garrett",
        "flow_limit": 105.0,
        "N_max": 93000.0,
        "pr_c_max": 5.2,
        "eta_c": 0.81,
        "eta_t": 0.76,
        "ar_options": {
            "0.96": {"m_t_limit": 55.0, "k_t": 0.80},
            "1.08": {"m_t_limit": 62.0, "k_t": 0.70},
            "1.23": {"m_t_limit": 70.0, "k_t": 0.60}
        },
        "surge_coeff": 18.0,
        "speed_lines": [30000, 50000, 70000, 85000, 93000]
    },
    "Garrett G-Series G55-2450 (98mm)": {
        "brand": "Garrett",
        "flow_limit": 150.0,
        "N_max": 84000.0,
        "pr_c_max": 5.5,
        "eta_c": 0.82,
        "eta_t": 0.77,
        "ar_options": {
            "1.00": {"m_t_limit": 72.0, "k_t": 0.65},
            "1.24": {"m_t_limit": 84.0, "k_t": 0.55},
            "1.40": {"m_t_limit": 96.0, "k_t": 0.48}
        },
        "surge_coeff": 24.0,
        "speed_lines": [25000, 45000, 65000, 77000, 84000]
    },

    # --- BORGWARNER EFR SERIES ---
    "BorgWarner EFR-7670": {
        "brand": "BorgWarner EFR",
        "inducer_mm": 57.0, "exducer_mm": 76.0, "trim": 56,
        "flow_limit_lb_min": 64.0, "flow_limit": 29.0,
        "N_max": 130000.0, "pr_c_max": 3.6, "eta_c": 0.76, "eta_t": 0.71,
        "turbine_inducer_mm": 70.0, "turbine_exducer_mm": 62.0, "turbine_trim": 78,
        "ar_options": {
            "0.83": {"m_t_limit": 22.0, "k_t": 1.2},
            "1.05": {"m_t_limit": 28.0, "k_t": 0.9}
        },
        "surge_coeff": 8.0,
        "speed_lines": [50000, 75000, 95000, 115000, 130000]
    },
    "BorgWarner EFR-8374": {
        "brand": "BorgWarner EFR",
        "inducer_mm": 62.0, "exducer_mm": 83.0, "trim": 56,
        "flow_limit_lb_min": 79.0, "flow_limit": 35.8,
        "N_max": 120000.0, "pr_c_max": 3.8, "eta_c": 0.77, "eta_t": 0.72,
        "turbine_inducer_mm": 74.0, "turbine_exducer_mm": 65.0, "turbine_trim": 77,
        "ar_options": {
            "0.92": {"m_t_limit": 29.0, "k_t": 1.1},
            "1.05": {"m_t_limit": 33.0, "k_t": 0.95},
            "1.45": {"m_t_limit": 39.0, "k_t": 0.75}
        },
        "surge_coeff": 9.5,
        "speed_lines": [40000, 60000, 80000, 100000, 120000]
    },

    # --- MITSUBISHI HEAVY INDUSTRIES (MHI) ---
    "MHI TD05": {
        "brand": "MHI",
        "inducer_mm": 48.0, "exducer_mm": 68.0, "trim": 50,
        "flow_limit_lb_min": 45.0, "flow_limit": 20.4,
        "N_max": 145000.0, "pr_c_max": 3.0, "eta_c": 0.73, "eta_t": 0.68,
        "turbine_inducer_mm": 56.0, "turbine_exducer_mm": 49.0, "turbine_trim": 77,
        "ar_options": {
            "0.60": {"m_t_limit": 14.0, "k_t": 1.3},
            "0.80": {"m_t_limit": 19.0, "k_t": 1.0}
        },
        "surge_coeff": 5.8,
        "speed_lines": [50000, 80000, 105000, 125000, 145000]
    }
}

# --- DIGITIZED MAP DRAWING FUNCTIONS ---
def draw_compressor_map_axes(ax, turbo_params, matched_flow, matched_pr, stage_label):
    flow_limit = turbo_params["flow_limit"]
    N_max = turbo_params["N_max"]
    pr_c_max = turbo_params["pr_c_max"]
    surge_coeff = turbo_params["surge_coeff"]
    speed_lines = turbo_params["speed_lines"]
    eta_c_peak = turbo_params.get("eta_c", 0.78)
    
    flow_max = flow_limit * 1.18
    pr_max_val = pr_c_max * 1.15
    flow_opt = flow_limit * 0.55
    pr_opt = 1.0 + 0.45 * (pr_c_max - 1.0)
    
    flow_grid = np.linspace(0.0, flow_max, 120)
    pr_grid = np.linspace(1.0, pr_max_val, 120)
    F_g, P_g = np.meshgrid(flow_grid, pr_grid)
    
    c1_val = (eta_c_peak) / (flow_opt ** 2)
    c2_val = (eta_c_peak) / ((pr_opt - 1.0) ** 2)
    F_g_skew = F_g - 0.20 * (P_g - pr_opt) * flow_opt
    
    eff_g = eta_c_peak - 1.25 * (c1_val * (F_g_skew - flow_opt)**2 + c2_val * (P_g - pr_opt)**2)
    eff_g = np.clip(eff_g, 0.50, eta_c_peak)
    
    # Digitized efficiency islands
    contours = ax.contour(F_g, P_g, eff_g, levels=[0.60, 0.65, 0.70, 0.75, 0.77, eta_c_peak], colors='navy', alpha=0.6, linewidths=1.2)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.2f')
    
    # Surge Line
    pr_surge_vals = np.linspace(1.0, pr_c_max * 1.1, 100)
    flow_surge_vals = surge_coeff * np.sqrt(np.maximum(pr_surge_vals - 1.0, 0.0))
    ax.plot(flow_surge_vals, pr_surge_vals, 'r--', linewidth=2.0, label="Surge Line (مرز سرج)")
    
    # Choke Line
    ax.axvline(x=flow_limit, color='royalblue', linestyle=':', linewidth=1.5, label="Choke Limit")
    
    # Speed Lines
    for speed in speed_lines:
        pr_center = 1.0 + (speed / N_max)**1.8 * (pr_c_max - 1.0)
        flow_center = (speed / N_max) * flow_opt * 1.15
        
        f_min = surge_coeff * np.sqrt(max(pr_center*0.8 - 1.0, 0.0))
        f_max = flow_center * 1.45
        
        f_vals = np.linspace(f_min, min(f_max, flow_limit * 1.1), 100)
        p_vals = pr_center - 0.75 * (pr_center - 1.0) * ((f_vals - f_min) / (f_max - f_min))**2
        
        surge_flow = surge_coeff * np.sqrt(np.maximum(p_vals - 1.0, 0.0))
        valid = f_vals >= surge_flow
        
        ax.plot(f_vals[valid], p_vals[valid], 'k-', alpha=0.5, linewidth=1.1)
        if len(f_vals[valid]) > 0:
            ax.text(f_vals[valid][-1], p_vals[valid][-1], f"{speed/1000:.0f}k RPM", fontsize=6, fontweight='bold')
            
    # Plot Operating Point
    ax.plot(matched_flow, matched_pr, 'g*', markersize=14, label=f"Operating Point ({matched_flow:.1f} kg/min, PR: {matched_pr:.2f})")
    
    surge_boundary = surge_coeff * np.sqrt(max(matched_pr - 1.0, 0.0))
    if matched_flow < surge_boundary:
        ax.text(matched_flow + 0.5, matched_pr, "⚠️ SURGE ZONE", color='red', fontsize=11, fontweight='bold')
    elif matched_flow > flow_limit:
        ax.text(matched_flow - 3.0, matched_pr, "⚠️ CHOKE ZONE", color='darkblue', fontsize=11, fontweight='bold')
        
    ax.set_xlabel("Corrected Air Flow [kg/min]", fontsize=8, fontweight='bold')
    ax.set_ylabel("Compressor Pressure Ratio (PR_c)", fontsize=8, fontweight='bold')
    ax.set_title(f"{stage_label} Compressor Map ({turbo_params['brand']})", fontsize=9, fontweight='bold')
    ax.set_xlim(0, flow_max)
    ax.set_ylim(1.0, pr_max_val)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(prop={'size': 7}, loc='upper left')

def draw_turbine_map_axes(ax, limit_t, k_t_coeff, matched_pr, matched_flow_corr, stage_label):
    pr_t_vals = np.linspace(1.0, 3.8, 100)
    flow_t_curve = limit_t * (1.0 - np.exp(-k_t_coeff * (pr_t_vals - 1.0)))
    
    ax.plot(pr_t_vals, flow_t_curve, 'm-', linewidth=2.2, label="Turbine Capacity")
    ax.plot(matched_pr, matched_flow_corr, 'c*', markersize=14, label=f"Operating Point ({matched_flow_corr:.1f} kg/min, PR_t: {matched_pr:.2f})")
    
    ax.set_xlabel("Turbine Expansion Ratio (PR_t)", fontsize=8, fontweight='bold')
    ax.set_ylabel("Corrected Exhaust Flow [kg/min]", fontsize=8, fontweight='bold')
    ax.set_title(f"{stage_label} Turbine Map", fontsize=9, fontweight='bold')
    ax.set_xlim(1.0, 3.8)
    ax.set_ylim(0.0, limit_t * 1.18)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(prop={'size': 7}, loc='upper left')

# Page Config
st.set_page_config(
    page_title="Wankel Engine 0D Simulator v35",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌀 Wankel Rotary Engine 0D Thermodynamic Simulator - Version 35")
st.markdown("""
نسخه تخصصی کالیبره‌شده، مجهز به **پیش‌فرض نمودار تجربی اختصاصی (photo_2026-09-12_02-12-11.jpg)**، **کاتالوگ دیجیتالی واقعی توربوشارژرهای Garrett Performance**، **رفع کامل خطاهای تحلیل پارامتریک در حالت پرخوران** و **ماژول سوخت داینامیک با کنترل ضریب لاندا (\\(\\lambda\\))**.
""")
class WankelSimulator0D_V14:
    def __init__(self, R=0.105, e=0.015, b=0.070, r_c=9.4, N=3000.0, 
                 p_ambient_bar=1.013, T_ambient_k=288.15, T_wall=420.0,
                 LHV=44e6, AFR=14.7, V_pocket_cc=35.0, crevol_cc=0.875,
                 tipo=-530.0, tipc=-180.0, tepo=199.0, tepc=588.5,
                 ipa_cm2=12.2, epa_cm2=6.5, tspark=-30.0, arealk_cm2=0.012,
                 spark_plugs=2, comb_dur_deg=72.0, throttle_pct=100.0,
                 throttle_lut=None, p_in_override=None, T_in_override=None, p_ex_override=None):
        self.throttle_lut = throttle_lut
        self.R = R
        self.e = e
        self.b = b
        self.r_c = r_c
        self.N = N
        self.omega = N * (2 * np.pi / 60.0)
        self.throttle_pct = throttle_pct
        
        self.p_ambient_pa = p_ambient_bar * 1e5
        self.T_ambient_k = T_ambient_k
        self.T_in = T_ambient_k
        self.p_ex = p_ambient_bar * 1e5
        self.T_wall = T_wall
        self.LHV = LHV
        self.AFR = AFR
        self.spark_plugs = spark_plugs
        self.comb_dur_input = comb_dur_deg
        
        # Timing parameters (relative to combustion TDC at 540 degrees)
        self.tipo = tipo
        self.tipc = tipc
        self.tepo = tepo
        self.tepc = tepc
        self.tspark = tspark
        
        # Port areas (converted to m^2)
        self.ipa = ipa_cm2 * 1e-4
        self.epa = epa_cm2 * 1e-4
        
        # Leakage area per apex seal (converted to m^2)
        self.A_leak = arealk_cm2 * 1e-4
        
        # Pocket and crevice volumes (m^3)
        self.V_pocket = V_pocket_cc * 1e-6
        self.crevol = crevol_cc * 1e-6
        
        # Gas properties
        self.R_gas = 287.05
        self.gamma = 1.35
        self.Cv = self.R_gas / (self.gamma - 1)
        self.Cp = self.gamma * self.Cv
        
        # Derived geometry
        self.V_d = 3.0 * np.sqrt(3) * self.e * self.R * self.b
        
        # Decouple nominal compression ratio from rotor pocket volume to prevent double-counting
        # Total minimum volume (V_min) at TDC must equal V_d / (r_c - 1).
        # Since the minimum volume in get_volume is V_c + V_pocket, we set V_c = V_min - V_pocket.
        V_min_req = self.V_d / (self.r_c - 1.0)
        if self.V_pocket >= V_min_req:
            # Prevent negative or zero clearance volume if the pocket is too large
            self.V_pocket = V_min_req * 0.80
            
        self.V_c = V_min_req - self.V_pocket
        self.V_max = self.V_c + self.V_d
        
        # Override values for matched turbocharger operation
        if p_in_override is not None:
            self.p_in = p_in_override
        else:
            if self.throttle_lut is not None:
                self.p_in = self.p_ambient_pa * self.interpolate_throttle_ratio(self.throttle_pct)
            else:
                self.p_in = self.p_ambient_pa * (0.25 + 0.75 * ((self.throttle_pct / 100.0) ** 1.5))

        if T_in_override is not None:
            self.T_in = T_in_override
            
        if p_ex_override is not None:
            self.p_ex = p_ex_override

    def interpolate_throttle_ratio(self, t_val):
        if not self.throttle_lut:
            return 0.25 + 0.75 * ((t_val / 100.0) ** 1.5)
        sorted_keys = sorted(self.throttle_lut.keys())
        if t_val <= sorted_keys[0]:
            return self.throttle_lut[sorted_keys[0]]
        if t_val >= sorted_keys[-1]:
            return self.throttle_lut[sorted_keys[-1]]
        for i in range(len(sorted_keys) - 1):
            k1 = sorted_keys[i]
            k2 = sorted_keys[i+1]
            if k1 <= t_val <= k2:
                r1 = self.throttle_lut[k1]
                r2 = self.throttle_lut[k2]
                return r1 + (r2 - r1) * (t_val - k1) / (k2 - k1)
        return 1.0

    def get_volume(self, theta_deg):
        theta_rad = np.radians(theta_deg)
        V = self.V_c + (self.V_d / 2.0) * (1.0 - np.cos(2.0 * theta_rad / 3.0)) + self.V_pocket
        dV_dtheta_rad = (self.V_d / 3.0) * np.sin(2.0 * theta_rad / 3.0)
        dV_dtheta = dV_dtheta_rad * (np.pi / 180.0) # per degree
        return V, dV_dtheta

    def get_surface_area(self, V):
        A_sides = 2.0 * (V / self.b)
        A_rotor = self.b * (np.sqrt(3) * self.R)
        A_housing = self.b * (np.sqrt(3) * self.R * (1.0 + 1.5 * (self.e / self.R)))
        return A_sides + A_rotor + A_housing

    def get_port_areas(self, theta_deg):
        theta_ipo = (540.0 + self.tipo) % 1080.0
        theta_ipc = (540.0 + self.tipc) % 1080.0
        theta_epo = (540.0 + self.tepo) % 1080.0
        theta_epc = (540.0 + self.tepc) % 1080.0
        
        if theta_ipo <= theta_ipc:
            if theta_ipo <= theta_deg <= theta_ipc:
                a_in = self.ipa * (np.sin(np.pi * (theta_deg - theta_ipo) / (theta_ipc - theta_ipo)) ** 2)
            else:
                a_in = 0.0
        else:
            if theta_deg >= theta_ipo or theta_deg <= theta_ipc:
                dist_total = (1080.0 - theta_ipo) + theta_ipc
                dist_curr = (theta_deg - theta_ipo) if theta_deg >= theta_ipo else (1080.0 - theta_ipo + theta_deg)
                a_in = self.ipa * (np.sin(np.pi * dist_curr / dist_total) ** 2)
            else:
                a_in = 0.0
                
        if theta_epo <= theta_epc:
            if theta_epo <= theta_deg <= theta_epc:
                a_ex = self.epa * (np.sin(np.pi * (theta_deg - theta_epo) / (theta_epc - theta_epo)) ** 2)
            else:
                a_ex = 0.0
        else:
            if theta_deg >= theta_epo or theta_deg <= theta_epc:
                dist_total = (1080.0 - theta_epo) + theta_epc
                dist_curr = (theta_deg - theta_epo) if theta_deg >= theta_epo else (1080.0 - theta_epo + theta_deg)
                a_ex = self.epa * (np.sin(np.pi * dist_curr / dist_total) ** 2)
            else:
                a_ex = 0.0
                
        return a_in, a_ex

    def compressible_flow(self, p_up, T_up, p_down, area, Cd=0.62):
        if area <= 0.0 or p_up <= 0.0 or p_down <= 0.0:
            return 0.0
        pr = p_down / p_up
        r_crit = (2.0 / (self.gamma + 1.0)) ** (self.gamma / (self.gamma - 1.0))
        
        if pr <= r_crit: # Choked Flow
            m_dot = Cd * area * p_up * np.sqrt((self.gamma / (self.R_gas * T_up)) * \
                    ((2.0 / (self.gamma + 1.0)) ** ((self.gamma + 1.0) / (self.gamma - 1.0))))
        else: # Subsonic Flow
            m_dot = Cd * area * p_up * np.sqrt(((2.0 * self.gamma) / (self.gamma - 1.0)) / (self.R_gas * T_up) * \
                    ((pr ** (2.0 / self.gamma)) - (pr ** ((self.gamma + 1.0) / self.gamma))))
        return m_dot

    def woschni_norman_ht(self, p, T, V, N_rpm, p_motoring=None, is_combustion=False):
        B = self.R
        p_bar = p / 1e5
        V_m = np.pi * N_rpm * self.R / 90.0 # Mean rotor tip velocity
        
        if is_combustion and p_motoring is not None:
            C1 = 0.5
            C2 = 0.00324 # m/(s K)
            V_ipc, _ = self.get_volume((540.0 + self.tipc) % 1080.0)
            p_ipc = self.p_in
            
            p_diff = max(p - p_motoring, 0.0)
            w = C1 * V_m + C2 * (self.V_d * self.T_in / (p_ipc * V_ipc)) * p_diff
        else:
            w = 0.5 * V_m
            
        h_c = 3.2 * 3.26 * (B ** -0.2) * (p_bar ** 0.8) * (T ** -0.53) * (w ** 0.8)
        return h_c

    def wiebe_combustion(self, theta, theta_start=510.0):
        if self.spark_plugs == 1:
            theta_dur = self.comb_dur_input * 1.25
            a = 4.0
        elif self.spark_plugs == 3:
            theta_dur = self.comb_dur_input * 0.85
            a = 6.0
        else:
            theta_dur = self.comb_dur_input
            a = 5.0
            
        m = 2.0
        
        if theta < theta_start:
            return 0.0, 0.0
        elif theta > (theta_start + theta_dur):
            return 1.0, 0.0
        else:
            y = (theta - theta_start) / theta_dur
            x_b = 1.0 - np.exp(-a * (y ** (m + 1)))
            dx_b_dtheta = (a * (m + 1) / theta_dur) * (y ** m) * np.exp(-a * (y ** (m + 1)))
            return x_b, dx_b_dtheta

    def simulate(self, d_theta=0.5, fire=True, p_motoring_external=None):
        # 1. Run motoring trace first if Firing is enabled to get p_motoring_trace for Woschni
        p_motoring_trace = p_motoring_external
        if fire and p_motoring_trace is None:
            _, motoring_summary = self.simulate(d_theta=d_theta, fire=False)
            p_motoring_trace = motoring_summary['pressure_trace_A']
            
        thetas = np.arange(0.0, 1080.0 + d_theta, d_theta)
        N_steps = len(thetas)
        dt_dtheta = 1.0 / (6.0 * self.N)
        dt_step = d_theta * dt_dtheta
        
        p = np.zeros((3, N_steps))
        T = np.zeros((3, N_steps))
        m = np.zeros((3, N_steps))
        m_fuel = np.zeros((3, N_steps))
        
        T_u = np.zeros((3, N_steps))
        T_b = np.zeros((3, N_steps))
        V_u = np.zeros((3, N_steps))
        V_b = np.zeros((3, N_steps))
        
        cum_heat_release = np.zeros((3, N_steps))
        cum_heat_loss = np.zeros((3, N_steps))
        cum_crevice_loss = np.zeros((3, N_steps))
        
        heat_release_rate_ch0 = np.zeros(N_steps)
        heat_loss_rate_ch0 = np.zeros(N_steps)
        
        work = np.zeros(3)
        m_crev = np.zeros((3, N_steps))
        m_fuel_start_ch = np.zeros(3)
        
        # Initialize
        for ch in range(3):
            theta_init = (ch * 360.0) % 1080.0
            V_init, _ = self.get_volume(theta_init)
            
            if 0.0 <= theta_init < 270.0:
                p[ch, 0] = self.p_in
                T[ch, 0] = self.T_in
            else:
                p[ch, 0] = self.p_ambient_pa
                T[ch, 0] = self.T_wall
                
            m[ch, 0] = (p[ch, 0] * V_init) / (self.R_gas * T[ch, 0])
            m_fuel[ch, 0] = 0.0
            m_crev[ch, 0] = (p[ch, 0] * self.crevol) / (self.R_gas * self.T_wall)
            
            T_u[ch, 0] = T[ch, 0]
            T_b[ch, 0] = T[ch, 0]
            V_u[ch, 0] = V_init
            V_b[ch, 0] = 0.0
            
        p_old_deriv = np.zeros(3)
        
        # Time Loop
        for i in range(N_steps - 1):
            theta_A = thetas[i]
            
            ch_angles = np.array([
                theta_A,
                (theta_A + 360.0) % 1080.0,
                (theta_A + 720.0) % 1080.0
            ])
            
            p_curr = p[:, i]
            T_curr = T[:, i]
            m_curr = m[:, i]
            mf_curr = m_fuel[:, i]
            m_crev_curr = m_crev[:, i]
            
            V_curr = np.zeros(3)
            dV_curr = np.zeros(3)
            V_next = np.zeros(3)
            A_wall = np.zeros(3)
            A_in = np.zeros(3)
            A_ex = np.zeros(3)
            
            for ch in range(3):
                V_curr[ch], dV_curr[ch] = self.get_volume(ch_angles[ch])
                V_next[ch], _ = self.get_volume((ch_angles[ch] + d_theta) % 1080.0)
                A_wall[ch] = self.get_surface_area(V_curr[ch])
                A_in[ch], A_ex[ch] = self.get_port_areas(ch_angles[ch])
                
            # 1. PORT FLOWS
            dm_in = np.zeros(3)
            dm_ex = np.zeros(3)
            h_in = np.zeros(3)
            h_ex = np.zeros(3)
            
            for ch in range(3):
                if A_in[ch] > 0.0:
                    if self.p_in >= p_curr[ch]:
                        dm = self.compressible_flow(self.p_in, self.T_in, p_curr[ch], A_in[ch]) * dt_step
                        dm_in[ch] = dm
                        h_in[ch] = self.Cp * self.T_in
                    else:
                        dm = self.compressible_flow(p_curr[ch], T_curr[ch], self.p_in, A_in[ch]) * dt_step
                        dm_in[ch] = -dm
                        h_in[ch] = self.Cp * T_curr[ch]
                        
                if A_ex[ch] > 0.0:
                    if p_curr[ch] >= self.p_ex:
                        dm = self.compressible_flow(p_curr[ch], T_curr[ch], self.p_ex, A_ex[ch]) * dt_step
                        dm_ex[ch] = -dm
                        h_ex[ch] = self.Cp * T_curr[ch]
                    else:
                        dm = self.compressible_flow(self.p_ex, 600.0, p_curr[ch], A_ex[ch]) * dt_step
                        dm_ex[ch] = dm
                        h_ex[ch] = self.Cp * 600.0
                        
            # 2. COUPLED APEX SEAL LEAKAGE
            dm_leak_AB = 0.0
            dm_leak_BC = 0.0
            dm_leak_CA = 0.0
            h_leak_AB, h_leak_BC, h_leak_CA = 0.0, 0.0, 0.0
            
            if p_curr[0] >= p_curr[1]:
                dm_leak_AB = self.compressible_flow(p_curr[0], T_curr[0], p_curr[1], self.A_leak) * dt_step
                h_leak_AB = self.Cp * T_curr[0]
            else:
                dm_leak_AB = -self.compressible_flow(p_curr[1], T_curr[1], p_curr[0], self.A_leak) * dt_step
                h_leak_AB = self.Cp * T_curr[1]
                
            if p_curr[1] >= p_curr[2]:
                dm_leak_BC = self.compressible_flow(p_curr[1], T_curr[1], p_curr[2], self.A_leak) * dt_step
                h_leak_BC = self.Cp * T_curr[1]
            else:
                dm_leak_BC = -self.compressible_flow(p_curr[2], T_curr[2], p_curr[1], self.A_leak) * dt_step
                h_leak_BC = self.Cp * T_curr[2]
                
            if p_curr[2] >= p_curr[0]:
                dm_leak_CA = self.compressible_flow(p_curr[2], T_curr[2], p_curr[0], self.A_leak) * dt_step
                h_leak_CA = self.Cp * T_curr[2]
            else:
                dm_leak_CA = -self.compressible_flow(p_curr[0], T_curr[0], p_curr[2], self.A_leak) * dt_step
                h_leak_CA = self.Cp * T_curr[0]
                
            dm_leak_net = np.array([
                -dm_leak_AB + dm_leak_CA,
                -dm_leak_BC + dm_leak_AB,
                -dm_leak_CA + dm_leak_BC
            ])
            
            dH_leak_net = np.array([
                -abs(dm_leak_AB) * h_leak_AB if dm_leak_AB > 0 else abs(dm_leak_AB) * h_leak_AB + \
                (abs(dm_leak_CA) * h_leak_CA if dm_leak_CA < 0 else -abs(dm_leak_CA) * h_leak_CA),
                
                -abs(dm_leak_BC) * h_leak_BC if dm_leak_BC > 0 else abs(dm_leak_BC) * h_leak_BC + \
                (abs(dm_leak_AB) * h_leak_AB if dm_leak_AB < 0 else -abs(dm_leak_AB) * h_leak_AB),
                
                -abs(dm_leak_CA) * h_leak_CA if dm_leak_CA > 0 else abs(dm_leak_CA) * h_leak_CA + \
                (abs(dm_leak_BC) * h_leak_BC if dm_leak_BC < 0 else -abs(dm_leak_BC) * h_leak_BC)
            ])
            
            # 3. DYNAMIC CREVICE FLOWS
            dm_crev = np.zeros(3)
            dH_crev = np.zeros(3)
            
            for ch in range(3):
                m_crev_target = (p_curr[ch] * self.crevol) / (self.R_gas * self.T_wall)
                dm_crev[ch] = m_crev_target - m_crev_curr[ch]
                
                if dm_crev[ch] > 0.0:
                    dH_crev[ch] = -dm_crev[ch] * self.Cp * T_curr[ch]
                else:
                    dH_crev[ch] = -dm_crev[ch] * self.Cp * self.T_wall
                    
            m_crev[:, i+1] = m_crev_curr + dm_crev
            
            # 4. ENERGY & MASS SOLVER
            for ch in range(3):
                theta_local = ch_angles[ch]
                theta_spark_local = (540.0 + self.tspark) % 1080.0
                is_combustion = fire and (theta_spark_local <= theta_local < 810.0)
                
                dQ_comb = 0.0
                dm_f_burned = 0.0
                if is_combustion:
                    xb, dx_b = self.wiebe_combustion(theta_local, theta_start=theta_spark_local)
                    if theta_local == theta_spark_local or m_fuel_start_ch[ch] == 0.0:
                        m_fuel_start_ch[ch] = max(mf_curr[ch], (m_curr[ch] / (1.0 + self.AFR)))
                        
                    Q_total = m_fuel_start_ch[ch] * self.LHV * 0.88
                    dQ_comb = Q_total * dx_b * d_theta
                    dm_f_burned = m_fuel_start_ch[ch] * dx_b * d_theta
                    
                    if ch == 0:
                        heat_release_rate_ch0[i] = (Q_total * dx_b) / dt_dtheta
                    
                p_mot = p_motoring_trace[i] if p_motoring_trace is not None else None
                h_conv = self.woschni_norman_ht(p_curr[ch], T_curr[ch], V_curr[ch], self.N, p_mot, is_combustion)
                
                if is_combustion: # TWO-ZONE MODEL
                    xb_fraction, _ = self.wiebe_combustion(theta_local, theta_start=theta_spark_local)
                    xb_fraction = max(min(xb_fraction, 1.0), 0.0)
                    
                    m_b = m_curr[ch] * xb_fraction
                    m_u = m_curr[ch] * (1.0 - xb_fraction)
                    
                    if xb_fraction > 0.0:
                        A_wall_b = A_wall[ch] * (xb_fraction ** (2.0/3.0))
                        A_wall_u = A_wall[ch] - A_wall_b
                    else:
                        A_wall_b = 0.0
                        A_wall_u = A_wall[ch]
                        
                    dQ_ht_u = h_conv * A_wall_u * (T_u[ch, i] - self.T_wall) * dt_step
                    dQ_ht_b = h_conv * A_wall_b * (T_b[ch, i] - self.T_wall) * dt_step
                    dQ_ht_total = dQ_ht_u + dQ_ht_b
                    
                    if ch == 0:
                        heat_loss_rate_ch0[i] = dQ_ht_total / dt_step
                    
                    m_next = max(m_curr[ch] + dm_in[ch] + dm_ex[ch] + dm_leak_net[ch] - dm_crev[ch], 1e-8)
                    m[ch, i+1] = m_next
                    m_fuel[ch, i+1] = max(mf_curr[ch] + (dm_in[ch] / (1.0 + self.AFR) if dm_in[ch] > 0 else 0.0) - dm_f_burned, 0.0)
                    
                    if m_u > 0.0:
                        T_u_next = T_u[ch, i] * ((V_curr[ch]/V_next[ch])**(self.gamma-1.0)) - (dQ_ht_u / (m_u * self.Cp))
                        T_u_next = max(T_u_next, 300.0)
                    else:
                        T_u_next = self.T_wall
                        
                    U_curr = m_curr[ch] * self.Cv * T_curr[ch]
                    dU = dQ_comb - dQ_ht_total - p_curr[ch] * (V_next[ch] - V_curr[ch]) + \
                         dm_in[ch] * h_in[ch] + dm_ex[ch] * h_ex[ch] + dH_leak_net[ch] + dH_crev[ch]
                    U_next = max(U_curr + dU, 1e-3)
                    
                    T_avg_next = U_next / (m_next * self.Cv)
                    T[ch, i+1] = T_avg_next
                    
                    if m_b > 0.0:
                        T_b_next = (T_avg_next * m_next - m_u * T_u_next) / m_b
                        T_b_next = max(T_b_next, T_u_next)
                    else:
                        T_b_next = T_avg_next
                        
                    T_u[ch, i+1] = T_u_next
                    T_b[ch, i+1] = T_b_next
                    
                else: # SINGLE-ZONE MODEL
                    dQ_ht_total = h_conv * A_wall[ch] * (T_curr[ch] - self.T_wall) * dt_step
                    
                    if ch == 0:
                        heat_loss_rate_ch0[i] = dQ_ht_total / dt_step
                    
                    m_next = max(m_curr[ch] + dm_in[ch] + dm_ex[ch] + dm_leak_net[ch] - dm_crev[ch], 1e-8)
                    m[ch, i+1] = m_next
                    m_fuel[ch, i+1] = max(mf_curr[ch] + (dm_in[ch] / (1.0 + self.AFR) if dm_in[ch] > 0 else 0.0), 0.0)
                    
                    U_curr = m_curr[ch] * self.Cv * T_curr[ch]
                    dU = -dQ_ht_total - p_curr[ch] * (V_next[ch] - V_curr[ch]) + \
                         dm_in[ch] * h_in[ch] + dm_ex[ch] * h_ex[ch] + dH_leak_net[ch] + dH_crev[ch]
                    U_next = max(U_curr + dU, 1e-3)
                    
                    T[ch, i+1] = U_next / (m_next * self.Cv)
                    T_u[ch, i+1] = T[ch, i+1]
                    T_b[ch, i+1] = T[ch, i+1]
                    
                if is_combustion:
                    V_eff_comb = V_next[ch] + 1.20 * self.V_pocket
                    p[ch, i+1] = (m[ch, i+1] * self.R_gas * T[ch, i+1]) / V_eff_comb
                else:
                    p[ch, i+1] = (m[ch, i+1] * self.R_gas * T[ch, i+1]) / V_next[ch]
                p_old_deriv[ch] = (p[ch, i+1] - p_curr[ch]) / dt_step
                
                cum_heat_release[ch, i+1] = cum_heat_release[ch, i] + dQ_comb
                cum_heat_loss[ch, i+1] = cum_heat_loss[ch, i] + dQ_ht_total
                cum_crevice_loss[ch, i+1] = cum_crevice_loss[ch, i] + max(dH_crev[ch], 0.0)
                
                work[ch] += p_curr[ch] * (V_next[ch] - V_curr[ch])

        volumes = np.zeros(N_steps)
        for i in range(N_steps):
            volumes[i], _ = self.get_volume(thetas[i])
            
        results = pd.DataFrame({
            'theta': thetas,
            'volume_cm3': volumes * 1e6,
            'pressure_bar': p[0, :] / 1e5,
            'temperature_K': T[0, :],
            'mass_g': m[0, :] * 1e3,
            'heat_release_kW': heat_release_rate_ch0 / 1000.0,
            'heat_loss_kW': heat_loss_rate_ch0 / 1000.0,
            'cum_heat_release_J': cum_heat_release[0, :],
            'cum_heat_loss_J': cum_heat_loss[0, :],
            'cum_crevice_loss_J': cum_crevice_loss[0, :]
        })
        
        W_ind = work[0]
        P_ind = W_ind * (self.N / 60.0) / 1000.0
        eta_ind = (W_ind / cum_heat_release[0, -1]) * 100.0 if cum_heat_release[0, -1] > 0 else 0.0
        imep = (W_ind / self.V_d) / 1e5
        
        # Calculate indices
        idx_ipo = np.argmin(np.abs(thetas - ((540.0 + self.tipo) % 1080.0)))
        idx_ipc = np.argmin(np.abs(thetas - ((540.0 + self.tipc) % 1080.0)))
        idx_epc = np.argmin(np.abs(thetas - ((540.0 + self.tepc) % 1080.0)))
        idx_810 = np.argmin(np.abs(thetas - 810.0))
        
        m_chamber_0 = m[0, :]
        m_ipc = m_chamber_0[idx_ipc]
        m_fresh_trapped = max(m_ipc - m_chamber_0[0], 1e-6)
        
        rho_amb = self.p_ambient_pa / (self.R_gas * self.T_in)
        vol_eff = (m_fresh_trapped / (rho_amb * self.V_d)) * 100.0
        
        T_810 = T[0, idx_810]
        m_fuel_start_g = m_fuel_start_ch[0] * 1e3
        
        summary = {
            'V_d_cm3': self.V_d * 1e6,
            'V_c_cm3': self.V_c * 1e6,
            'compression_ratio': self.r_c,
            'indicated_work_J': W_ind,
            'indicated_power_kW': P_ind,
            'indicated_efficiency_pct': eta_ind,
            'imep_bar': imep,
            'pressure_trace_A': p[0, :],
            'total_heat_loss_J': cum_heat_loss[0, -1],
            'total_crevice_loss_J': cum_crevice_loss[0, -1],
            'volumetric_efficiency_pct': vol_eff,
            'm_fresh_trapped_kg': m_fresh_trapped,
            'T_end_expansion_K': T_810,
            'm_fuel_start_g': m_fuel_start_g
        }
        
        return results, summary

# Presets Library
presets_data = {
    "Mazda 13B NA (RX-7 - 2 Rotor 1.3L)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 9.7, "V_pocket_cc": 35.0,
        "tipo": -530.0, "tipc": -180.0, "tepo": 199.0, "tepc": 588.5, "ipa_cm2": 12.2, "epa_cm2": 6.5,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.875, "spark_plugs_count": 2
    },
    "AIE 225CS (Range Extender - 1 Rotor 225cc)": {
        "num_rotors": 1, "R_mm": 78.0, "e_mm": 11.6, "b_mm": 40.0, "r_c": 9.2, "V_pocket_cc": 14.0,
        "tipo": -535.0, "tipc": -165.0, "tepo": 205.0, "tepc": 580.0, "ipa_cm2": 5.5, "epa_cm2": 3.2,
        "tspark": -25.0, "arealk_cm2": 0.008, "crevol_cc": 0.45, "spark_plugs_count": 1
    },
    "UAV Wankel 802 / 802W (Technion UAV 1 Rotor)": {
        "num_rotors": 1, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 68.0, "r_c": 9.2, "V_pocket_cc": 36.0,
        "tipo": -530.0, "tipc": -180.0, "tepo": 199.0, "tepc": 588.5, "ipa_cm2": 12.0, "epa_cm2": 6.5,
        "tspark": -28.0, "arealk_cm2": 0.01, "crevol_cc": 0.88, "spark_plugs_count": 2
    },
    "Mazda 12A (RX-7 SA/FB - NA)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 70.0, "r_c": 9.4, "V_pocket_cc": 35.0,
        "tipo": -530.0, "tipc": -180.0, "tepo": 199.0, "tepc": 588.5, "ipa_cm2": 12.2, "epa_cm2": 6.5,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.875, "spark_plugs_count": 2
    },
    "Mazda RX-8 Renesis (13B-MSP - NA)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 10.0, "V_pocket_cc": 15.0,
        "tipo": -537.0, "tipc": -205.0, "tepo": 220.0, "tepc": 537.0, "ipa_cm2": 10.11, "epa_cm2": 15.10,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.80, "spark_plugs_count": 2
    },
    "Mazda RX-7 FC NA (13B-DEI)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 9.4, "V_pocket_cc": 35.0,
        "tipo": -508.0, "tipc": -220.0, "tepo": 195.0, "tepc": 578.0, "ipa_cm2": 12.2, "epa_cm2": 6.5,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.80, "spark_plugs_count": 2
    },
    "Mazda 10A (Cosmo Sport - NA)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 60.0, "r_c": 9.4, "V_pocket_cc": 25.0,
        "tipo": -508.0, "tipc": -230.0, "tepo": 195.0, "tepc": 578.0, "ipa_cm2": 9.5, "epa_cm2": 5.0,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.70, "spark_plugs_count": 2
    },
    "NSU Ro80 (KKM 612 - NA)": {
        "num_rotors": 2, "R_mm": 100.0, "e_mm": 14.0, "b_mm": 67.0, "r_c": 9.0, "V_pocket_cc": 18.0,
        "tipo": -515.0, "tipc": -225.0, "tepo": 200.0, "tepc": 580.0, "ipa_cm2": 10.0, "epa_cm2": 5.5,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.60, "spark_plugs_count": 1
    },
    "Curtiss-Wright RC2-60 (Aviation - NA)": {
        "num_rotors": 2, "R_mm": 116.8, "e_mm": 16.5, "b_mm": 76.2, "r_c": 8.5, "V_pocket_cc": 42.0,
        "tipo": -525.0, "tipc": -230.0, "tepo": 200.0, "tepc": 575.0, "ipa_cm2": 15.0, "epa_cm2": 10.0,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 1.0, "spark_plugs_count": 2
    }
}

# 1. Initialize session states
if "R_mm" not in st.session_state:
    default_vals = presets_data["Mazda 12A (RX-7 SA/FB - NA)"]
    for k, v in default_vals.items():
        st.session_state[k] = v

if "preset_select" in st.session_state and st.session_state.preset_select != "Custom (تنظیمات دستی)":
    current_preset = st.session_state.preset_select
    preset_vals = presets_data[current_preset]
    for k, v in preset_vals.items():
        if k in st.session_state and st.session_state[k] != v:
            st.session_state.preset_select = "Custom (تنظیمات دستی)"
            break

def on_preset_change():
    preset = st.session_state.preset_select
    if preset != "Custom (تنظیمات دستی)":
        selected_preset_vals = presets_data[preset]
        for k, v in selected_preset_vals.items():
            st.session_state[k] = v

# Sidebar - Templates
st.sidebar.subheader("Select Engine Template (انتخاب پیش‌فرض موتور)")
engine_preset = st.sidebar.selectbox("Preset Templates", [
    "Custom (تنظیمات دستی)",
    "Mazda 12A (RX-7 SA/FB - NA)",
    "Mazda RX-8 Renesis (13B-MSP - NA)",
    "Mazda RX-7 FC NA (13B-DEI)",
    "Mazda 10A (Cosmo Sport - NA)",
    "NSU Ro80 (KKM 612 - NA)",
    "Curtiss-Wright RC2-60 (Aviation - NA)"
], key="preset_select", on_change=on_preset_change)

st.sidebar.header("🌀 Engine Basic Configuration")
num_rotors = st.sidebar.number_input("Number of Rotors (i)", min_value=1, max_value=8, step=1, key="num_rotors")

# Run Configuration (پیکربندی شبیه‌سازی)
st.sidebar.header("🔥 Run Configuration (پیکربندی شبیه‌سازی)")
sim_mode = st.sidebar.radio("Simulation Mode (حالت شبیه‌سازی)", ["Firing (همراه با احتراق)", "Motoring (بدون احتراق / موتورینگ)"], index=0)
enable_firing = (sim_mode == "Firing (همراه با احتراق)")

# Fuel Selection
fuel_type = st.sidebar.selectbox("Fuel Type (نوع سوخت)", ["Gasoline (بنزین معمولی)", "Premium Gasoline (بنزین سوپر)", "Aviation Gas (100LL)", "Hydrogen (هیدروژن خالص)", "Custom Fuel (سوخت سفارشی)"])

if fuel_type == "Gasoline (بنزین معمولی)":
    lhv_base = 44.0e6
    afr_stoich = 14.7
elif fuel_type == "Premium Gasoline (بنزین سوپر)":
    lhv_base = 44.5e6
    afr_stoich = 14.7
elif fuel_type == "Aviation Gas (100LL)":
    lhv_base = 43.5e6
    afr_stoich = 14.5
elif fuel_type == "Hydrogen (هیدروژن خالص)":
    lhv_base = 120.0e6
    afr_stoich = 34.3
else: # Custom Fuel
    lhv_base = st.sidebar.number_input("Custom Lower Heating Value LHV (MJ/kg)", min_value=1.0, max_value=200.0, value=44.0, step=1.0) * 1e6
    afr_stoich = st.sidebar.number_input("Custom Stoichiometric AFR", min_value=1.0, max_value=100.0, value=14.7, step=0.1)

# Excess Air Ratio (Lambda) / Equivalence Ratio (Phi)
lambda_val = st.sidebar.slider("Excess Air Ratio (ضریب هوای اضافی) [λ]", min_value=0.5, max_value=2.5, value=1.0, step=0.05, help="Lambda > 1 means Lean (رقیق), Lambda < 1 means Rich (غنی)")
phi_val = 1.0 / lambda_val
st.sidebar.markdown(f"📊 **Equivalence Ratio (نسبت هم‌ارزی) [Φ]:** `{phi_val:.2f}` ({'Rich/غنی' if phi_val > 1.05 else 'Lean/رقیق' if phi_val < 0.95 else 'Stoichiometric/استوکیومتریک'})")

# Compute actual AFR and LHV
afr_val = lambda_val * afr_stoich
lhv_val = lhv_base

spark_plugs_count = st.sidebar.selectbox("Spark Plugs Per Rotor (تعداد شمع)", [1, 2, 3], key="spark_plugs_count")

st.sidebar.header("🔧 Engine Geometry")
R_mm = st.sidebar.number_input("Rotor Generating Radius R (mm)", min_value=72.0, max_value=300.0, step=1.0, key="R_mm")
e_mm = st.sidebar.number_input("Eccentricity e (mm)", min_value=5.0, max_value=72.0, step=0.1, key="e_mm")
b_mm = st.sidebar.number_input("Rotor Width b (mm)", min_value=10.0, max_value=250.0, step=1.0, key="b_mm")
r_c = st.sidebar.number_input("Compression Ratio r_c (-)", min_value=3.0, max_value=20.0, step=0.1, key="r_c")
V_pocket_cc = st.sidebar.number_input("Rotor Pocket Volume (cc)", min_value=0.0, max_value=200.0, step=1.0, key="V_pocket_cc")

# Throttle settings
st.sidebar.header("🕹️ Throttle & Ambient (بارگذاری و تراتل)")
throttle_pct = st.sidebar.number_input("Throttle Opening (درصد دریچه گاز)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)

throttle_mode = st.sidebar.selectbox(
    "Throttle Model (نوع مدل تراتل)", 
    [
        "Experimental Test Data (نمودار تجربی photo_2026-09-12_02-12-11 - پیش‌فرض)", 
        "Analytical Formula (رابطه تحلیلی یاماموتو)", 
        "Custom File / Interactive Table (وارد کردن یا بارگذاری فایل اختصاصی)"
    ], 
    index=0,
    key="throttle_mode"
)

throttle_lut = None

if throttle_mode == "Experimental Test Data (نمودار تجربی photo_2026-09-12_02-12-11 - پیش‌فرض)":
    throttle_lut = EXPERIMENTAL_THROTTLE_LUT
    st.sidebar.success("✅ نمودار تجربی کالیبره‌شده (photo_2026-09-12_02-12-11) لود شد!")
    with st.sidebar.expander("📊 مشاهده داده‌های تجربی MAP vs Throttle"):
        df_exp = pd.DataFrame(list(EXPERIMENTAL_THROTTLE_LUT.items()), columns=["Throttle (%)", "Pressure Ratio (MAP/P_amb)"])
        df_exp["MAP (mbar)"] = df_exp["Pressure Ratio (MAP/P_amb)"] * 1013.25
        st.dataframe(df_exp, use_container_width=True)

elif throttle_mode == "Analytical Formula (رابطه تحلیلی یاماموتو)":
    throttle_lut = None
    st.sidebar.info("ℹ️ محاسبه بر اساس فرمول تحلیلی یاماموتو:\nP_in/P_amb = 0.25 + 0.75*(Throttle/100)^1.5")

elif throttle_mode == "Custom File / Interactive Table (وارد کردن یا بارگذاری فایل اختصاصی)":
    st.sidebar.markdown("### 📥 بارگذاری فایل یا ویرایش دستی داده‌ها")
    uploaded_throttle_file = st.sidebar.file_uploader("Upload Throttle Curve (Excel or CSV)", type=["xlsx", "xls", "csv"])
    
    if uploaded_throttle_file is not None:
        try:
            if uploaded_throttle_file.name.endswith(".csv"):
                df_lut = pd.read_csv(uploaded_throttle_file)
            else:
                df_lut = pd.read_excel(uploaded_throttle_file)
            
            col_throttle = None
            col_ratio = None
            for col in df_lut.columns:
                col_lower = str(col).lower()
                if "throttle" in col_lower or "تراتل" in col_lower or "%" in col_lower:
                    col_throttle = col
                if "ratio" in col_lower or "map" in col_lower or "نسبت" in col_lower or "p_in" in col_lower or "pressure" in col_lower:
                    col_ratio = col
            
            if col_throttle is not None and col_ratio is not None:
                lut_df = df_lut[[col_throttle, col_ratio]].dropna()
                lut_df[col_throttle] = pd.to_numeric(lut_df[col_throttle])
                lut_df[col_ratio] = pd.to_numeric(lut_df[col_ratio])
                
                max_val = lut_df[col_ratio].max()
                if max_val > 100: 
                    if max_val > 10000: 
                        lut_df[col_ratio] = lut_df[col_ratio] / 101325.0
                    else: 
                        lut_df[col_ratio] = lut_df[col_ratio] / 1013.25
                elif max_val > 1.5: 
                    lut_df[col_ratio] = lut_df[col_ratio] / 1.01325
                
                throttle_lut = dict(zip(lut_df[col_throttle], lut_df[col_ratio]))
                st.sidebar.success("✅ فایل اختصاصی بارگذاری شد!")
                st.sidebar.dataframe(lut_df, use_container_width=True)
            else:
                st.sidebar.error("ستون‌های فایل شناسایی نشد. باید ستون‌ها 'Throttle (%)' و 'Pressure Ratio' باشند.")
        except Exception as e:
            st.sidebar.error(f"خطا در خواندن فایل: {str(e)}")
    else:
        st.sidebar.info("💡 می‌توانید فایل آپلود کنید یا داده‌ها را مستقیماً در جدول زیر ویرایش کنید:")
        initial_custom_df = pd.DataFrame({
            "Throttle (%)": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "Pressure Ratio": [0.20, 0.503, 0.671, 0.799, 0.878, 0.938, 0.967, 0.982, 0.987, 0.992, 1.000]
        })
        edited_df = st.sidebar.data_editor(initial_custom_df, num_rows="dynamic", key="custom_throttle_editor")
        if edited_df is not None:
            try:
                t_col = edited_df.iloc[:, 0]
                r_col = edited_df.iloc[:, 1]
                throttle_lut = dict(zip(pd.to_numeric(t_col), pd.to_numeric(r_col)))
            except Exception:
                throttle_lut = None

    template_data = {
        "Throttle (%)": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Pressure Ratio": [0.20, 0.503, 0.671, 0.799, 0.878, 0.938, 0.967, 0.982, 0.987, 0.992, 1.000]
    }
    df_template = pd.DataFrame(template_data)
    csv_template = df_template.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "📥 دانلود الگوی نمونه فایل (CSV)", 
        data=csv_template, 
        file_name="wankel_throttle_curve_template.csv", 
        mime="text/csv"
    )

ambient_mode = st.sidebar.radio("Ambient Conditions Mode", ["Manual (دستی)", "Altitude-based (ارتفاع از سطح دریا)"])

if ambient_mode == "Manual (دستی)":
    p_ambient_bar = st.sidebar.number_input("Ambient Pressure (bar)", min_value=0.1, max_value=2.0, value=1.013, step=0.01)
    t_ambient_c = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-50.0, max_value=60.0, value=15.0, step=1.0)
    p_amb_pa = p_ambient_bar * 1e5
    t_amb_k = t_ambient_c + 273.15
else:
    altitude_ft = st.sidebar.number_input("Altitude (ft)", min_value=0.0, max_value=80000.0, value=0.0, step=1000.0)
    p_ambient_bar, t_amb_k, p_amb_pa = get_isa_properties(altitude_ft)
    t_ambient_c = t_amb_k - 273.15
    st.sidebar.info(f"Calculated ISA P_amb: {p_ambient_bar:.3f} bar, T_amb: {t_ambient_c:.1f} °C")

# --- TURBOCHARGER CONFIGURATION SECTION ---
st.sidebar.header("🌀 Turbocharger Configuration")
enable_turbo = st.sidebar.checkbox("Enable Turbocharger (فعالسازی توربوشارژر)", value=False)

if enable_turbo:
    turbo_stages = st.sidebar.selectbox("Turbocharger Stages (تعداد مراحل پرخوران)", ["Single-Stage (تک مرحله ای)", "Dual-Stage (دو مرحله ای)"])
    
    if turbo_stages == "Single-Stage (تک مرحله ای)":
        turbo_brand = st.sidebar.selectbox("Turbocharger Brand (برند توربوشارژر)", ["Garrett", "BorgWarner EFR", "MHI", "Custom Operator (ورود مپ اختصاصی)"])
        if turbo_brand == "Custom Operator (ورود مپ اختصاصی)":
            st.sidebar.markdown("### 🛠️ ورود یا بارگذاری مپ اختصاصی کمپرسور/توربین")
            custom_flow_limit = st.sidebar.number_input("Max Compressor Flow Limit (kg/min)", min_value=5.0, max_value=200.0, value=28.0, step=1.0)
            custom_pr_max = st.sidebar.number_input("Max Compressor Pressure Ratio (PR_c max)", min_value=1.1, max_value=6.0, value=3.5, step=0.1)
            custom_eta_c = st.sidebar.number_input("Peak Compressor Efficiency", min_value=0.50, max_value=0.90, value=0.78, step=0.01)
            custom_surge_coeff = st.sidebar.number_input("Surge Line Coefficient", min_value=1.0, max_value=15.0, value=5.5, step=0.5)
            custom_m_t_limit = st.sidebar.number_input("Turbine Flow Limit (kg/min)", min_value=5.0, max_value=150.0, value=20.0, step=1.0)
            
            uploaded_map_file = st.sidebar.file_uploader("Upload Custom Compressor Map (CSV/Excel)", type=["csv", "xlsx", "xls"], key="custom_turbo_map_file")
            if uploaded_map_file is not None:
                try:
                    df_map = pd.read_csv(uploaded_map_file) if uploaded_map_file.name.endswith(".csv") else pd.read_excel(uploaded_map_file)
                    st.sidebar.success("✅ مپ اختصاصی با موفقیت بارگذاری شد!")
                    st.sidebar.dataframe(df_map.head(), use_container_width=True)
                except Exception as e_map:
                    st.sidebar.error(f"خطا در خواندن فایل مپ: {str(e_map)}")
            
            TURBO_LIBRARY["Custom Operator Map"]["flow_limit"] = custom_flow_limit
            TURBO_LIBRARY["Custom Operator Map"]["pr_c_max"] = custom_pr_max
            TURBO_LIBRARY["Custom Operator Map"]["eta_c"] = custom_eta_c
            TURBO_LIBRARY["Custom Operator Map"]["surge_coeff"] = custom_surge_coeff
            TURBO_LIBRARY["Custom Operator Map"]["ar_options"]["Custom (0.80)"]["m_t_limit"] = custom_m_t_limit
            available_models = ["Custom Operator Map"]
        else:
            available_models = [k for k, v in TURBO_LIBRARY.items() if v["brand"] == turbo_brand]

        turbo_type = st.sidebar.selectbox("Turbocharger Model (مدل توربوشارژر)", available_models)
        
        ar_options = list(TURBO_LIBRARY[turbo_type]["ar_options"].keys())
        selected_ar = st.sidebar.selectbox("Turbine Housing A/R Ratio (نسبت A/R توربین)", ar_options, index=1 if len(ar_options) > 1 else 0)
        target_boost_bar = st.sidebar.slider("Target Boost Pressure (فشار بوست هدف) [bar gauge]", min_value=0.1, max_value=2.5, value=0.0, step=0.1)
        ic_eff = st.sidebar.slider("Intercooler Efficiency (راندمان اینترکولر)", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
        
        hp_turbo_type = turbo_type
        hp_selected_ar = selected_ar
        lp_turbo_type = "NA"
        lp_selected_ar = "NA"
    else: # Dual-Stage
        st.sidebar.subheader("Stage 1: High Pressure (HP) [مرحله فشار بالا]")
        hp_brand = st.sidebar.selectbox("HP Turbo Brand", ["Garrett", "BorgWarner EFR", "MHI"], key="hp_brand")
        hp_available = [k for k, v in TURBO_LIBRARY.items() if v["brand"] == hp_brand]
        hp_turbo_type = st.sidebar.selectbox("HP Turbo Model", hp_available, key="hp_model")
        hp_ar_options = list(TURBO_LIBRARY[hp_turbo_type]["ar_options"].keys())
        hp_selected_ar = st.sidebar.selectbox("HP Turbine A/R Ratio", hp_ar_options, index=1 if len(hp_ar_options) > 1 else 0, key="hp_ar")
        
        st.sidebar.subheader("Stage 2: Low Pressure (LP) [مرحله فشار ضعیف]")
        lp_brand = st.sidebar.selectbox("LP Turbo Brand", ["Garrett", "BorgWarner EFR", "MHI"], key="lp_brand")
        lp_available = [k for k, v in TURBO_LIBRARY.items() if v["brand"] == lp_brand]
        lp_turbo_type = st.sidebar.selectbox("LP Turbo Model", lp_available, key="lp_model")
        lp_ar_options = list(TURBO_LIBRARY[lp_turbo_type]["ar_options"].keys())
        lp_selected_ar = st.sidebar.selectbox("LP Turbine A/R Ratio", lp_ar_options, index=1 if len(lp_ar_options) > 1 else 0, key="lp_ar")
        
        target_boost_bar = st.sidebar.slider("Target Boost Pressure (فشار بوست هدف) [bar gauge]", min_value=0.1, max_value=3.5, value=1.2, step=0.1)
        ic_eff = st.sidebar.slider("Intercooler Efficiency (راندمان اینترکولر)", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
        
        turbo_type = hp_turbo_type
        selected_ar = hp_selected_ar
else:
    turbo_stages = "Single-Stage (تک مرحله ای)"
    turbo_type = "NA"
    selected_ar = "NA"
    hp_turbo_type = "NA"
    hp_selected_ar = "NA"
    lp_turbo_type = "NA"
    lp_selected_ar = "NA"
    target_boost_bar = 0.0
    ic_eff = 0.0

p_in_bar = p_amb_pa / 1e5
p_in_pa = p_amb_pa
t_in_k = t_amb_k
p_ex_pa = p_amb_pa

N_rpm = st.sidebar.number_input("Engine Speed N (RPM)", min_value=500, max_value=15000, value=3000, step=100)
T_wall_k = st.sidebar.number_input("Wall Temperature T_wall (K)", min_value=250, max_value=800, value=420, step=10)
A_leak_mm2 = st.sidebar.number_input("Equivalent Leakage Area (mm²)", min_value=0.0, max_value=20.0, value=1.2, step=0.1)

tipo = st.sidebar.number_input("Intake Port Open Angle (TIPO) [deg BTDC]", step=5.0, key="tipo")
tipc = st.sidebar.number_input("Intake Port Close Angle (TIPC) [deg BTDC]", step=5.0, key="tipc")
tepo = st.sidebar.number_input("Exhaust Port Open Angle (TEPO) [deg ATDC]", step=5.0, key="tepo")
tepc = st.sidebar.number_input("Exhaust Port Close Angle (TEPC) [deg ATDC]", step=5.0, key="tepc")

ipa_cm2 = st.sidebar.number_input("Intake Port Area (IPA) [cm²]", min_value=0.1, step=0.1, key="ipa_cm2")
epa_cm2 = st.sidebar.number_input("Exhaust Port Area (EPA) [cm²]", min_value=0.1, step=0.1, key="epa_cm2")

tspark = st.sidebar.number_input("Spark Timing (TSPARK) [deg BTDC]", step=1.0, key="tspark")
arealk_cm2 = st.sidebar.number_input("Apex Seal Leak Area (AREALK) [cm²]", min_value=0.0, step=0.001, key="arealk_cm2")
crevol_cc = st.sidebar.number_input("Crevice Volume (CREVOL) [cc]", min_value=0.0, step=0.05, key="crevol_cc")
comb_dur_deg = st.sidebar.number_input("Nominal Combustion Duration (°)", min_value=5.0, max_value=180.0, value=72.0, step=1.0)

R = R_mm * 1e-3
e = e_mm * 1e-3
b = b_mm * 1e-3
A_leak = A_leak_mm2 * 1e-6

# --- MATRICES & MATCHING SOLVER ---
matched_pr_c = 1.0
matched_pr_t = 1.0
matched_airflow_kg_min = 0.0
matched_egt_k = 300.0
matched_turbo_rpm = 0.0
is_wastegate_open = False

matched_pr_c_lp = 1.0
matched_pr_c_hp = 1.0
matched_pr_t_lp = 1.0
matched_pr_t_hp = 1.0
matched_turbo_rpm_lp = 0.0
matched_turbo_rpm_hp = 0.0

if enable_turbo:
    if turbo_stages == "Single-Stage (تک مرحله ای)":
        # 1D Matching Solver
        turbo_params = TURBO_LIBRARY[turbo_type]
        flow_limit = turbo_params["flow_limit"]
        N_max = turbo_params["N_max"]
        pr_c_max = turbo_params["pr_c_max"]
        eta_c = turbo_params["eta_c"]
        eta_t = turbo_params["eta_t"]
        
        # Selected A/R options
        ar_params = turbo_params["ar_options"][selected_ar]
        m_t_limit = ar_params["m_t_limit"]
        k_t = ar_params["k_t"]
            
        eta_mech = 0.95
        Cp_gas = 1005.0
        Cp_ex = 1150.0
        gamma_gas = 1.4
        gamma_ex = 1.33
        
        # Matching Iteration loop to find stable MAP and EMP
        pr_c_iter = 1.0
        pr_t_iter = 1.0
        t_in_iter = t_amb_k
        p_ex_iter = p_amb_pa
        
        # Temporary simulation instance for motoring reference
        temp_sim = WankelSimulator0D_V14(
            R=R, e=e, b=b, r_c=r_c, N=N_rpm,
            p_ambient_bar=p_ambient_bar, T_ambient_k=t_amb_k, T_wall=T_wall_k,
            LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
            tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
            ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
            spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=throttle_pct,
            throttle_lut=throttle_lut
        )
        _, temp_mot_summary = temp_sim.simulate(fire=False)
        p_motoring_trace = temp_mot_summary['pressure_trace_A']
        
        for matching_step in range(5):
            # 1. Compressor out temp
            t_c_out = t_amb_k * (1.0 + (pr_c_iter ** ((gamma_gas - 1.0) / gamma_gas) - 1.0) / eta_c)
            # Intercooling
            t_in_iter = t_c_out - ic_eff * (t_c_out - (t_amb_k + 10.0))
            t_in_iter = max(t_in_iter, t_amb_k)
            
            # Manifold pressure
            p_c_out = p_amb_pa * pr_c_iter
            if throttle_lut is not None:
                p_ratio = temp_sim.interpolate_throttle_ratio(throttle_pct)
                p_in_iter = p_c_out * p_ratio
            else:
                p_in_iter = p_c_out * (0.25 + 0.75 * ((throttle_pct / 100.0) ** 1.5))
                
            p_ex_iter = p_amb_pa * pr_t_iter
            
            # Run 0D simulation using these boundary conditions
            iter_sim = WankelSimulator0D_V14(
                R=R, e=e, b=b, r_c=r_c, N=N_rpm,
                p_ambient_bar=p_ambient_bar, T_ambient_k=t_amb_k, T_wall=T_wall_k,
                LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
                tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
                ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
                spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=throttle_pct,
                throttle_lut=throttle_lut, p_in_override=p_in_iter, T_in_override=t_in_iter, p_ex_override=p_ex_iter
            )
            
            _, iter_summary = iter_sim.simulate(fire=True, p_motoring_external=p_motoring_trace)
            
            # Extract results
            m_dot_air = iter_summary['m_fresh_trapped_kg'] * N_rpm * (num_rotors / 60.0) # kg/s
            m_dot_ex = m_dot_air * (1.0 + 1.0 / afr_val)
            T_egt = iter_summary['T_end_expansion_K'] * 0.88
            
            # Convert to kg/min for map matching
            m_dot_air_min = m_dot_air * 60.0
            m_dot_ex_min = m_dot_ex * 60.0
            m_dot_ex_corr = m_dot_ex_min * np.sqrt(T_egt / 288.15) / (p_amb_pa / 101325.0)
            
            # Solve for Matched expansion ratio
            flow_ratio = min(m_dot_ex_corr / m_t_limit, 0.96)
            pr_t_matched = 1.0 - (1.0 / k_t) * np.log(1.0 - flow_ratio)
            
            # Turbine power and Compressor matched capacity
            W_dot_t = m_dot_ex * Cp_ex * T_egt * (1.0 - (1.0 / pr_t_matched) ** ((gamma_ex - 1.0) / gamma_ex)) * eta_t
            W_dot_c = W_dot_t * eta_mech
            
            T_c_out_matched = t_amb_k + W_dot_c / (max(m_dot_air, 1e-4) * Cp_gas)
            pr_c_matched = (1.0 + eta_c * (T_c_out_matched - t_amb_k) / t_amb_k) ** (gamma_gas / (gamma_gas - 1.0))
            
            # Target limit (Boost control via wastegate)
            # Cap manifold pressure at 990 mbar (99,000 Pa absolute) as requested
            p_in_target_pa = min(p_amb_pa + target_boost_bar * 1e5, 99000.0)
            pr_c_target = p_in_target_pa / p_amb_pa
            
            if pr_c_matched > pr_c_target:
                is_wastegate_open = True
                pr_c_iter = pr_c_target
                # Adjust turbine expansion ratio based on compressor work requirement
                W_dot_c_req = m_dot_air * Cp_gas * (t_amb_k * (1.0 + (pr_c_iter ** ((gamma_gas - 1.0) / gamma_gas) - 1.0) / eta_c) - t_amb_k)
                W_dot_t_req = W_dot_c_req / eta_mech
                
                term = W_dot_t_req / (max(m_dot_ex, 1e-4) * Cp_ex * T_egt * eta_t)
                term = min(max(term, 0.0), 0.9)
                pr_t_iter = (1.0 / (1.0 - term)) ** (gamma_ex / (gamma_ex - 1.0))
            else:
                is_wastegate_open = False
                pr_c_iter = pr_c_matched
                pr_t_iter = pr_t_matched
                
        # Stabilized Matched values
        matched_pr_c = pr_c_iter
        matched_pr_t = pr_t_iter
        p_in_pa = p_in_iter
        t_in_k = t_in_iter
        p_ex_pa = p_ex_iter
        matched_airflow_kg_min = m_dot_air * 60.0
        matched_egt_k = T_egt
        
        # Speed estimation
        speed_factor = (matched_pr_c ** 0.286 - 1.0) / (pr_c_max ** 0.286 - 1.0)
        matched_turbo_rpm = N_max * np.sqrt(max(speed_factor, 0.0))
        # Cap speed
        matched_turbo_rpm = min(matched_turbo_rpm, N_max * 1.1)
        
        matched_pr_c_lp = matched_pr_c
        matched_pr_c_hp = 1.0
        matched_pr_t_lp = matched_pr_t
        matched_pr_t_hp = 1.0
        matched_turbo_rpm_lp = matched_turbo_rpm
        matched_turbo_rpm_hp = 0.0
    else: # Dual-Stage
        hp_params = TURBO_LIBRARY[hp_turbo_type]
        hp_flow_limit = hp_params["flow_limit"]
        hp_N_max = hp_params["N_max"]
        hp_pr_c_max = hp_params["pr_c_max"]
        hp_eta_c = hp_params["eta_c"]
        hp_eta_t = hp_params["eta_t"]
        hp_ar_params = hp_params["ar_options"][hp_selected_ar]
        hp_m_t_limit = hp_ar_params["m_t_limit"]
        hp_k_t = hp_ar_params["k_t"]
        
        lp_params = TURBO_LIBRARY[lp_turbo_type]
        lp_flow_limit = lp_params["flow_limit"]
        lp_N_max = lp_params["N_max"]
        lp_pr_c_max = lp_params["pr_c_max"]
        lp_eta_c = lp_params["eta_c"]
        lp_eta_t = lp_params["eta_t"]
        lp_ar_params = lp_params["ar_options"][lp_selected_ar]
        lp_m_t_limit = lp_ar_params["m_t_limit"]
        lp_k_t = lp_ar_params["k_t"]
        
        eta_mech = 0.95
        Cp_gas = 1005.0
        Cp_ex = 1150.0
        gamma_gas = 1.4
        gamma_ex = 1.33
        
        # We target pressure ratio total split equally as starting point
        p_in_target_pa = min(p_amb_pa + target_boost_bar * 1e5, 99000.0)
        pr_c_total_target = p_in_target_pa / p_amb_pa
        
        pr_c_lp_iter = np.sqrt(pr_c_total_target)
        pr_c_hp_iter = np.sqrt(pr_c_total_target)
        
        pr_t_hp_iter = 1.25
        pr_t_lp_iter = 1.25
        
        t_in_iter = t_amb_k
        p_ex_iter = p_amb_pa
        
        # Temporary simulation instance for motoring reference
        temp_sim = WankelSimulator0D_V14(
            R=R, e=e, b=b, r_c=r_c, N=N_rpm,
            p_ambient_bar=p_ambient_bar, T_ambient_k=t_amb_k, T_wall=T_wall_k,
            LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
            tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
            ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
            spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=throttle_pct,
            throttle_lut=throttle_lut
        )
        _, temp_mot_summary = temp_sim.simulate(fire=False)
        p_motoring_trace = temp_mot_summary['pressure_trace_A']
        
        for matching_step in range(5):
            # 1. LP compressor out temp
            t_c_out_lp = t_amb_k * (1.0 + (pr_c_lp_iter ** ((gamma_gas - 1.0) / gamma_gas) - 1.0) / lp_eta_c)
            # Interstage cooler efficiency = ic_eff
            t_in_hp = t_c_out_lp - ic_eff * (t_c_out_lp - (t_amb_k + 10.0))
            t_in_hp = max(t_in_hp, t_amb_k)
            
            # 2. HP compressor out temp
            t_c_out_hp = t_in_hp * (1.0 + (pr_c_hp_iter ** ((gamma_gas - 1.0) / gamma_gas) - 1.0) / hp_eta_c)
            # Final intercooler
            t_in_iter = t_c_out_hp - ic_eff * (t_c_out_hp - (t_amb_k + 10.0))
            t_in_iter = max(t_in_iter, t_amb_k)
            
            # Total compressor outlet pressure
            p_c_out_total = p_amb_pa * pr_c_lp_iter * pr_c_hp_iter
            if throttle_lut is not None:
                p_ratio = temp_sim.interpolate_throttle_ratio(throttle_pct)
                p_in_iter = p_c_out_total * p_ratio
            else:
                p_in_iter = p_c_out_total * (0.25 + 0.75 * ((throttle_pct / 100.0) ** 1.5))
                
            p_ex_iter = p_amb_pa * pr_t_lp_iter * pr_t_hp_iter
            
            # Run simulation
            iter_sim = WankelSimulator0D_V14(
                R=R, e=e, b=b, r_c=r_c, N=N_rpm,
                p_ambient_bar=p_ambient_bar, T_ambient_k=t_amb_k, T_wall=T_wall_k,
                LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
                tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
                ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
                spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=throttle_pct,
                throttle_lut=throttle_lut, p_in_override=p_in_iter, T_in_override=t_in_iter, p_ex_override=p_ex_iter
            )
            _, iter_summary = iter_sim.simulate(fire=True, p_motoring_external=p_motoring_trace)
            
            m_dot_air = iter_summary['m_fresh_trapped_kg'] * N_rpm * (num_rotors / 60.0)
            m_dot_ex = m_dot_air * (1.0 + 1.0 / afr_val)
            T_egt = iter_summary['T_end_expansion_K'] * 0.88
            T_egt = max(T_egt, 350.0)
            
            m_dot_air_min = m_dot_air * 60.0
            m_dot_ex_min = m_dot_ex * 60.0
            
            # Corrected flow for HP turbine (inlet is exhaust port EMP: p_ex_iter)
            m_dot_ex_corr_hp = m_dot_ex_min * np.sqrt(T_egt / 288.15) / (p_ex_iter / 101325.0)
            flow_ratio_hp = min(m_dot_ex_corr_hp / hp_m_t_limit, 0.96)
            pr_t_matched_hp = 1.0 - (1.0 / hp_k_t) * np.log(1.0 - flow_ratio_hp)
            pr_t_matched_hp = max(pr_t_matched_hp, 1.05)
            
            # LP turbine inlet temp
            T_egt_lp = T_egt * (1.0 - hp_eta_t * (1.0 - (1.0 / pr_t_matched_hp) ** ((gamma_ex - 1.0) / gamma_ex)))
            T_egt_lp = max(T_egt_lp, 300.0)
            p_ex_mid = p_amb_pa * pr_t_lp_iter
            m_dot_ex_corr_lp = m_dot_ex_min * np.sqrt(T_egt_lp / 288.15) / (p_ex_mid / 101325.0)
            flow_ratio_lp = min(m_dot_ex_corr_lp / lp_m_t_limit, 0.96)
            pr_t_matched_lp = 1.0 - (1.0 / lp_k_t) * np.log(1.0 - flow_ratio_lp)
            pr_t_matched_lp = max(pr_t_matched_lp, 1.05)
            
            # HP compressor power
            W_dot_t_hp = m_dot_ex * Cp_ex * T_egt * (1.0 - (1.0 / pr_t_matched_hp) ** ((gamma_ex - 1.0) / gamma_ex)) * hp_eta_t
            W_dot_c_hp = W_dot_t_hp * eta_mech
            T_c_out_matched_hp = t_in_hp + W_dot_c_hp / (max(m_dot_air, 1e-4) * Cp_gas)
            pr_c_matched_hp = (1.0 + hp_eta_c * (T_c_out_matched_hp - t_in_hp) / t_in_hp) ** (gamma_gas / (gamma_gas - 1.0))
            pr_c_matched_hp = max(pr_c_matched_hp, 1.0)
            
            # LP compressor power
            W_dot_t_lp = m_dot_ex * Cp_ex * T_egt_lp * (1.0 - (1.0 / pr_t_matched_lp) ** ((gamma_ex - 1.0) / gamma_ex)) * lp_eta_t
            W_dot_c_lp = W_dot_t_lp * eta_mech
            T_c_out_matched_lp = t_amb_k + W_dot_c_lp / (max(m_dot_air, 1e-4) * Cp_gas)
            pr_c_matched_lp = (1.0 + lp_eta_c * (T_c_out_matched_lp - t_amb_k) / t_amb_k) ** (gamma_gas / (gamma_gas - 1.0))
            pr_c_matched_lp = max(pr_c_matched_lp, 1.0)
            
            pr_c_matched_total = pr_c_matched_lp * pr_c_matched_hp
            
            if pr_c_matched_total > pr_c_total_target:
                is_wastegate_open = True
                # Limit total PR achieved by wastegate
                pr_c_total_target = min(pr_c_total_target, lp_pr_c_max * hp_pr_c_max)
                pr_c_lp_sw = min(np.sqrt(pr_c_total_target), lp_pr_c_max)
                pr_c_hp_sw = pr_c_total_target / pr_c_lp_sw
                
                # HP Turbine required expansion
                W_dot_c_hp_req = m_dot_air * Cp_gas * (t_in_hp * (1.0 + (pr_c_hp_sw ** ((gamma_gas - 1.0) / gamma_gas) - 1.0) / hp_eta_c) - t_in_hp)
                W_dot_t_hp_req = W_dot_c_hp_req / eta_mech
                term_hp = W_dot_t_hp_req / (max(m_dot_ex, 1e-4) * Cp_ex * T_egt * hp_eta_t)
                term_hp = min(max(term_hp, 0.0), 0.9)
                pr_t_hp_iter = (1.0 / (1.0 - term_hp)) ** (gamma_ex / (gamma_ex - 1.0))
                pr_t_hp_iter = max(pr_t_hp_iter, 1.05)
                
                # LP Turbine required expansion
                T_egt_lp_req = T_egt * (1.0 - hp_eta_t * (1.0 - (1.0 / pr_t_hp_iter) ** ((gamma_ex - 1.0) / gamma_ex)))
                W_dot_c_lp_req = m_dot_air * Cp_gas * (t_amb_k * (1.0 + (pr_c_lp_sw ** ((gamma_gas - 1.0) / gamma_gas) - 1.0) / lp_eta_c) - t_amb_k)
                W_dot_t_lp_req = W_dot_c_lp_req / eta_mech
                term_lp = W_dot_t_lp_req / (max(m_dot_ex, 1e-4) * Cp_ex * T_egt_lp_req * lp_eta_t)
                term_lp = min(max(term_lp, 0.0), 0.9)
                pr_t_lp_iter = (1.0 / (1.0 - term_lp)) ** (gamma_ex / (gamma_ex - 1.0))
                pr_t_lp_iter = max(pr_t_lp_iter, 1.05)
            else:
                is_wastegate_open = False
                pr_c_lp_iter = pr_c_matched_lp
                pr_c_hp_iter = pr_c_matched_hp
                pr_t_lp_iter = pr_t_matched_lp
                pr_t_hp_iter = pr_t_matched_hp
                
        # Matched values
        matched_pr_c_lp = pr_c_lp_iter
        matched_pr_c_hp = pr_c_hp_iter
        matched_pr_t_lp = pr_t_lp_iter
        matched_pr_t_hp = pr_t_hp_iter
        
        matched_pr_c = pr_c_lp_iter * pr_c_hp_iter
        matched_pr_t = pr_t_lp_iter * pr_t_hp_iter
        
        p_in_pa = p_in_iter
        t_in_k = t_in_iter
        p_ex_pa = p_ex_iter
        matched_airflow_kg_min = m_dot_air * 60.0
        matched_egt_k = T_egt
        
        # LP Speed
        speed_factor_lp = (matched_pr_c_lp ** 0.286 - 1.0) / (lp_pr_c_max ** 0.286 - 1.0)
        matched_turbo_rpm_lp = lp_N_max * np.sqrt(max(speed_factor_lp, 0.0))
        matched_turbo_rpm_lp = min(matched_turbo_rpm_lp, lp_N_max * 1.1)
        
        # HP Speed
        speed_factor_hp = (matched_pr_c_hp ** 0.286 - 1.0) / (hp_pr_c_max ** 0.286 - 1.0)
        matched_turbo_rpm_hp = hp_N_max * np.sqrt(max(speed_factor_hp, 0.0))
        matched_turbo_rpm_hp = min(matched_turbo_rpm_hp, hp_N_max * 1.1)
        
        matched_turbo_rpm = matched_turbo_rpm_lp # For generic metrics

# Final Sim Instance Call
sim = WankelSimulator0D_V14(
    R=R, e=e, b=b, r_c=r_c, N=N_rpm,
    p_ambient_bar=p_ambient_bar, T_ambient_k=t_amb_k, T_wall=T_wall_k,
    LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
    tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
    ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
    spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=throttle_pct,
    throttle_lut=throttle_lut, p_in_override=p_in_pa, T_in_override=t_in_k, p_ex_override=p_ex_pa
)

results, summary = sim.simulate(fire=enable_firing)

# ----------------- POST-PROCESSING & SHAFT METRICS -----------------
fmep_bar = 0.35 + 0.12 * (N_rpm / 1000.0) + 0.015 * ((N_rpm / 1000.0) ** 2)
fmep_pa = fmep_bar * 1e5

W_ind = summary['indicated_work_J']
W_fric = fmep_pa * sim.V_d
W_brake = W_ind - W_fric

is_motoring = not enable_firing

if is_motoring:
    total_indicated_power_kW = summary['indicated_power_kW'] * num_rotors
    total_brake_power_kW = (W_brake * (N_rpm / 60.0) / 1000.0) * num_rotors
    omega = N_rpm * (2 * np.pi / 60.0)
    total_brake_torque_Nm = (total_brake_power_kW * 1000.0 / omega) if omega > 0 else 0.0
    dot_m_air_kgh = summary['m_fresh_trapped_kg'] * N_rpm * 60.0 * num_rotors
    dot_m_fuel_kgh = 0.0
    bsfc_g_kwh = float('inf')
else:
    total_indicated_power_kW = summary['indicated_power_kW'] * num_rotors
    total_brake_power_kW = (W_brake * (N_rpm / 60.0) / 1000.0) * num_rotors
    if total_brake_power_kW < 0:
        total_brake_power_kW = 0.0
    omega = N_rpm * (2 * np.pi / 60.0)
    total_brake_torque_Nm = (total_brake_power_kW * 1000.0 / omega) if omega > 0 else 0.0
    dot_m_air_kgh = summary['m_fresh_trapped_kg'] * N_rpm * 60.0 * num_rotors
    dot_m_fuel_kgh = dot_m_air_kgh / afr_val
    if total_brake_power_kW > 0.05:
        bsfc_g_kwh = (dot_m_fuel_kgh * 1000.0) / total_brake_power_kW
    else:
        bsfc_g_kwh = float('inf')

# EGT
t_egt_c = (summary['T_end_expansion_K'] * 0.88) - 273.15
total_displacement_cc = 2.0 * (sim.V_d * 1e6) * num_rotors

# ----------------- TABS LAYOUT -----------------
tab_dash, tab_turbo, tab_sweeps, tab_data, tab_theory = st.tabs([
    "📊 Performance Dashboard", 
    "🌀 Turbocharger Matching Map (نقشه تطبیق توربوشارژر)",
    "📈 Parametric Sweeps (تحلیل پارامتریک)", 
    "📂 Raw Simulation Data", 
    "📝 Theoretical Models"
])

with tab_dash:
    if is_motoring:
        st.subheader("Motoring Performance Metrics (شاخص‌های عملکرد موتورینگ - بدون احتراق)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Motoring Power Required (قدرت موتورینگ جذبی)", f"{abs(total_brake_power_kW):.2f} kW / {abs(total_brake_power_kW) * 1.34102:.1f} hp", help="Power required to spin the engine at this speed under motoring conditions")
        col2.metric("Motoring Torque Required (گشتاور موتورینگ)", f"{abs(total_brake_torque_Nm):.1f} N.m")
        col3.metric("Air Flow Rate (دبی هوای ورودی)", f"{dot_m_air_kgh:.2f} kg/h")
        col4.metric("Fuel Flow Rate (دبی جرمی سوخت)", "0.000 kg/h")
        col5.metric("BSFC (مصرف ویژه سوخت)", "N/A (Motoring)")

        st.markdown("---")
        st.subheader("In-Cylinder & Environmental Indicators")
        col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns(5)
        col2_1.metric("Pumping & Heat Loss (Indicated)", f"{total_indicated_power_kW:.2f} kW", help="Negative indicated power due to pumping and heat transfer losses")
        col2_2.metric("Estimated EGT (دمای اگزوز)", f"{t_egt_c:.1f} °C")
        col2_3.metric("Manifold Pressure (MAP)", f"{sim.p_in/1e5:.3f} bar")
        col2_4.metric("Thermal Efficiency", "N/A (Motoring)")
        col2_5.metric("Displacement", f"{total_displacement_cc:.1f} cc")
    else:
        st.subheader("Brake Performance Metrics (شاخص‌های عملکرد سر شفت / ترموزی)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Brake Power (توان سر شفت)", f"{total_brake_power_kW:.2f} kW / {total_brake_power_kW * 1.34102:.1f} hp", help="Brake power delivered to eccentric shaft")
        col2.metric("Brake Torque (گشتاور ترمزی)", f"{total_brake_torque_Nm:.1f} N.m")
        col3.metric("Air Flow Rate (دبی هوای ورودی)", f"{dot_m_air_kgh:.2f} kg/h")
        col4.metric("Fuel Flow Rate (دبی جرمی سوخت)", f"{dot_m_fuel_kgh:.3f} kg/h")
        if bsfc_g_kwh != float('inf'):
            col5.metric("BSFC (مصرف ویژه سوخت)", f"{bsfc_g_kwh:.1f} g/kWh")
        else:
            col5.metric("BSFC (مصرف ویژه سوخت)", "N/A (Stalled)")

        st.markdown("---")
        st.subheader("In-Cylinder & Environmental Indicators")
        col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns(5)
        col2_1.metric("Indicated Power", f"{total_indicated_power_kW:.2f} kW")
        col2_2.metric("Estimated EGT (دمای اگزوز)", f"{t_egt_c:.1f} °C")
        col2_3.metric("Manifold Pressure (MAP)", f"{sim.p_in/1e5:.3f} bar")
        col2_4.metric("Thermal Efficiency", f"{summary['indicated_efficiency_pct']:.2f} %")
        col2_5.metric("Displacement", f"{total_displacement_cc:.1f} cc")

    # Single-Run Summary Data export
    dash_summary_data = {
        "Parameter Name (نام پارامتر)": [
            "Brake Power (توان سر شفت)", "Brake Torque (گشتاور ترمزی)", "Air Flow Rate (دبی هوای ورودی)",
            "Fuel Flow Rate (دبی سوخت ورودی)", "BSFC (مصرف ویژه سوخت)", "Indicated Power (توان اندیکاتوری)",
            "Estimated EGT (دمای تخمینی اگزوز)", "Manifold Pressure downstream of throttle (MAP)",
            "Thermal Efficiency (راندمان حرارتی)", "Engine Displacement (حجم جابجایی کل موتور)",
            "Single Chamber Swept Volume (حجم محفظه کاری)", "Single Chamber Clearance Volume (حجم کلیرنس مرده)",
            "Compression Ratio (نسبت تراکم)", "Engine Speed (RPM)", "Throttle Opening (%)", "Number of Rotors",
            "Turbocharger Active", "Matched Compressor Pressure Ratio", "Matched Turbine Expansion Ratio"
        ],
        "Value (مقدار)": [
            f"{total_brake_power_kW:.2f} kW / {total_brake_power_kW * 1.34102:.1f} hp",
            f"{total_brake_torque_Nm:.1f} N.m", f"{dot_m_air_kgh:.2f} kg/h", f"{dot_m_fuel_kgh:.3f} kg/h",
            f"{bsfc_g_kwh:.1f} g/kWh" if bsfc_g_kwh != float('inf') else "N/A (Stalled)",
            f"{total_indicated_power_kW:.2f} kW", f"{t_egt_c:.1f} °C", f"{sim.p_in/1e5:.3f} bar",
            f"{summary['indicated_efficiency_pct']:.2f} %" if not is_motoring else "N/A (Motoring)", f"{total_displacement_cc:.1f} cc",
            f"{sim.V_d*1e6:.1f} cc", f"{sim.V_c*1e6:.1f} cc", f"{sim.r_c:.1f}", f"{N_rpm:.0f} RPM",
            f"{throttle_pct:.1f} %", f"{num_rotors:.0f}",
            "Yes" if enable_turbo else "No",
            f"{matched_pr_c:.2f}" if enable_turbo else "N/A",
            f"{matched_pr_t:.2f}" if enable_turbo else "N/A"
        ]
    }
    df_dash_summary = pd.DataFrame(dash_summary_data)
    csv_dash = df_dash_summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Single-Run Performance Summary (CSV/Excel)",
        data=csv_dash,
        file_name="wankel_single_run_performance_summary.csv",
        mime="text/csv"
    )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Pressure-Volume (P-V) Diagram")
        log_scale = st.checkbox("Logarithmic Scale")
        
        fig_pv, ax_pv = plt.subplots(figsize=(6, 4))
        ax_pv.plot(results['volume_cm3'], results['pressure_bar'], 'r-', linewidth=2.5)
        ax_pv.set_xlabel("Chamber Volume ($cm^3$)")
        ax_pv.set_ylabel("Pressure (bar)")
        ax_pv.grid(True, linestyle="--", alpha=0.5)
        
        if log_scale:
            ax_pv.set_xscale('log')
            ax_pv.set_yscale('log')
            ax_pv.set_title("Log P - Log V Cycle Diagram", fontweight="bold")
        else:
            ax_pv.set_title("Thermodynamic P-V Cycle Diagram", fontweight="bold")
        st.pyplot(fig_pv)

    with col_right:
        st.subheader("Energy Balance & Heat Release")
        fig_en, ax_en = plt.subplots(figsize=(6, 4))
        ax_en.plot(results['theta'], results['cum_heat_release_J'], 'm-', label='Heat Released', linewidth=2)
        ax_en.plot(results['theta'], results['cum_heat_loss_J'], 'k--', label='Heat Loss to Walls', linewidth=2)
        ax_en.set_xlabel("Crank Angle (deg)")
        ax_en.set_ylabel("Energy (J)")
        ax_en.set_title("Cumulative Heat Release & Wall Heat Loss", fontweight="bold")
        ax_en.legend()
        ax_en.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_en)

    st.markdown("---")
    st.subheader("In-Cylinder Parameter Variations vs. Crank Angle (Per Chamber)")
    fig_multi, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    axs[0, 0].plot(results['theta'], results['pressure_bar'], 'b-', linewidth=2)
    axs[0, 0].set_title("In-Cylinder Pressure", fontweight="bold")
    axs[0, 0].set_xlabel("Crank Angle (deg)")
    axs[0, 0].set_ylabel("Pressure (bar)")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    axs[0, 1].plot(results['theta'], results['temperature_K'], 'r-', linewidth=2)
    axs[0, 1].set_title("In-Cylinder Temperature", fontweight="bold")
    axs[0, 1].set_xlabel("Crank Angle (deg)")
    axs[0, 1].set_ylabel("Temperature (K)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    axs[1, 0].plot(results['theta'], results['volume_cm3'], 'g-', linewidth=2)
    axs[1, 0].set_title("Chamber Volume", fontweight="bold")
    axs[1, 0].set_xlabel("Crank Angle (deg)")
    axs[1, 0].set_ylabel("Volume ($cm^3$)")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    axs[1, 1].plot(results['theta'], results['mass_g'], 'y-', linewidth=2)
    axs[1, 1].set_title("In-Cylinder Gas Mass", fontweight="bold")
    axs[1, 1].set_xlabel("Crank Angle (deg)")
    axs[1, 1].set_ylabel("Gas Mass (g)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig_multi)


with tab_turbo:
    st.subheader("🌀 Turbocharger Matching & Performance Map analysis")
    
    if not enable_turbo:
        st.subheader("💡 سامانه هوشمند پیشنهاد و سایزینگ توربوشارژر (Turbocharger Smart Sizing & Recommendation)")
        st.markdown("""
        **ابزار کمکی طراحی پیشرانه:**  
        در صورتی که موتور به صورت تنفس طبیعی (تنفس طبیعی در سایدبار) کالیبره شده باشد، می‌توانید با استفاده از این ماژول هوشمند فیزیک‌محور، بهترین مدل توربوشارژر و نسبت هوزینگ اگزوز (A/R) متناسب با ارتفاع پروازی و بوست هدف خود را استخراج کنید. این ماژول بر اساس راندمان حجمی واقعی موتور شما در شرایط کنونی و موازنه اگزرژی گازهای داغ اگزوز، پیشنهادات بهینه را اولویت‌بندی می‌کند.
        """)
        
        # User input for sizing target
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            req_alt_ft = st.number_input("Target Flight Altitude (ارتفاع پروازی درخواستی) [ft]", min_value=0.0, max_value=80000.0, value=20000.0, step=1000.0)
        with col_s2:
            req_boost_bar = st.slider("Target Boost Pressure (فشار بوست درخواستی) [bar gauge]", min_value=0.1, max_value=3.5, value=1.2, step=0.1)
        with col_s3:
            req_rpm = st.number_input("Target Engine Speed (دور موتور درخواستی) [RPM]", min_value=1500, max_value=8000, value=int(N_rpm), step=100)
        with col_s4:
            target_stage_mode = st.selectbox("Sizing Mode (نوع پیکربندی درخواستی جهت سایزینگ)", ["Single-Stage (تک مرحله ای)", "Dual-Stage (دو مرحله ای)"])
            
        btn_suggest = st.button("🔍 آنالیز و پیشنهاد بهترین توربوشارژر (Find Optimal Turbocharger)")
        
        if btn_suggest:
            # Let's perform the sizing analysis!
            p_amb_local_bar, t_amb_local, p_amb_local_pa = get_isa_properties(req_alt_ft)
            
            # Target manifold pressure
            p_in_target_pa = min(p_amb_local_pa + req_boost_bar * 1e5, 99000.0)
            PR_c_req = p_in_target_pa / p_amb_local_pa
            
            # Retrieve volumetric efficiency from the NA baseline run
            vol_eff = summary['volumetric_efficiency_pct']
            
            # Pre-calculate required air flow under ambient conditions
            eta_c_est = 0.74
            t_c_out_est = t_amb_local * (1.0 + (PR_c_req ** 0.286 - 1.0) / eta_c_est)
            t_in_est = t_c_out_est - 0.80 * (t_c_out_est - (t_amb_local + 10.0))
            t_in_est = max(t_in_est, t_amb_local)
            
            rho_in_est = p_in_target_pa / (287.05 * t_in_est)
            m_dot_air_req_kg_s = rho_in_est * (vol_eff/100.0) * (sim.V_d * req_rpm * num_rotors / 60.0)
            m_dot_air_req_corr = m_dot_air_req_kg_s * 60.0 * np.sqrt(t_amb_local / 288.15) / (p_amb_local_pa / 101325.0)
            
            if target_stage_mode == "Single-Stage (تک مرحله ای)":
                
                T_egt_est = summary['T_end_expansion_K'] * 0.88 # K
                if T_egt_est < 500.0:
                    T_egt_est = 950.0
                    
                m_dot_ex_req_kg_s = m_dot_air_req_kg_s * (1.0 + 1.0 / afr_val)
                m_dot_ex_corr = m_dot_ex_req_kg_s * 60.0 * np.sqrt(T_egt_est / 288.15) / (p_amb_local_pa / 101325.0)
                
                st.info(f"**خلاصه پارامترهای تک مرحله‌ای درخواستی در ارتفاع {req_alt_ft:,.0f} پا:**  \n"
                        f"• نسبت فشار کل کمپرسور مورد نیاز (PR_total): `{PR_c_req:.2f}` \n"
                        f"• دبی جرمی هوای تصحیح‌شده ورودی (محیط): `{m_dot_air_req_corr:.2f} kg/min` \n"
                        f"• دمای تخمینی اگزوز (EGT): `{T_egt_est-273.15:.1f} °C`")
                
                recommendations = []
                for name, params in TURBO_LIBRARY.items():
                    brand = params["brand"]
                    flow_limit = params["flow_limit"]
                    pr_c_max = params["pr_c_max"]
                    surge_coeff = params["surge_coeff"]
                    
                    if PR_c_req > pr_c_max:
                        continue
                    if m_dot_air_req_corr > flow_limit:
                        continue
                    flow_surge = surge_coeff * np.sqrt(max(PR_c_req - 1.0, 0.0))
                    if m_dot_air_req_corr < flow_surge:
                        continue
                        
                    for ar, ar_data in params["ar_options"].items():
                        m_t_limit = ar_data["m_t_limit"]
                        k_t = ar_data["k_t"]
                        
                        if m_dot_ex_corr >= m_t_limit * 0.98:
                            continue
                        
                        comp_flow_ratio = m_dot_air_req_corr / flow_limit
                        comp_score = 100.0 - abs(comp_flow_ratio - 0.60) * 100.0
                        
                        turb_flow_ratio = m_dot_ex_corr / m_t_limit
                        turb_score = 100.0 - abs(turb_flow_ratio - 0.75) * 100.0
                        
                        total_score = 0.5 * comp_score + 0.5 * turb_score
                        total_score = max(min(total_score, 100.0), 0.0)
                        
                        if turb_flow_ratio > 0.90:
                            suitability = "قابل استفاده با بک‌پرشر بالا (High EMP Backpressure)"
                            total_score -= 20.0
                        elif turb_flow_ratio < 0.40:
                            suitability = "عدم پاسخگویی سریع / تاخیر بوست بالا (Laggy)"
                            total_score -= 20.0
                        else:
                            suitability = "انطباق عالی و راندمان بهینه (Highly Recommended)"
                            
                        recommendations.append({
                            "name": name,
                            "brand": brand,
                            "ar": ar,
                            "score": total_score,
                            "turb_flow_ratio": turb_flow_ratio,
                            "comp_flow_ratio": comp_flow_ratio,
                            "suitability": suitability
                        })
                        
                if len(recommendations) == 0:
                    st.error("❌ متاسفانه هیچ توربوشارژری در کتابخانه فعلی با این شرایط مرزی منطبق نیست.")
                else:
                    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)
                    st.subheader("🏆 پیشنهادات بهینه اولویت‌بندی شده برای سیستم تک مرحله‌ای:")
                    for idx, rec in enumerate(recommendations[:3]):
                        st.markdown(f"### رتبه {idx+1}: {rec['name']} (Housing A/R: {rec['ar']})")
                        st.write(f"• **درصد تطبیق فیزیکی کل سیستم:** `{rec['score']:.1f}%`")
                        st.markdown("---")
            else: # Dual-Stage Sizing logic
                PR_c_stage_req = np.sqrt(PR_c_req)
                
                # Estimate HP inlet pressure and temp for corrected flow calculation
                p_mid_est_pa = p_amb_local_pa * PR_c_stage_req
                t_c_out_lp_est = t_amb_local * (1.0 + (PR_c_stage_req ** 0.286 - 1.0) / 0.74)
                t_in_hp_est = t_c_out_lp_est - 0.80 * (t_c_out_lp_est - (t_amb_local + 10.0))
                t_in_hp_est = max(t_in_hp_est, t_amb_local)
                
                # Corrected flow at LP inlet (ambient)
                m_dot_air_lp_corr = m_dot_air_req_corr
                
                # Corrected flow at HP inlet (mid-pressure)
                m_dot_air_hp_corr = (p_in_target_pa / (287.05 * t_in_hp_est)) * (vol_eff/100.0) * (sim.V_d * req_rpm * num_rotors / 60.0) * 60.0 * np.sqrt(t_in_hp_est / 288.15) / (p_mid_est_pa / 101325.0)
                
                T_egt_est = summary['T_end_expansion_K'] * 0.88 # K
                if T_egt_est < 500.0:
                    T_egt_est = 950.0
                
                recommendations = []
                for lp_name, lp_p in TURBO_LIBRARY.items():
                    if PR_c_stage_req > lp_p["pr_c_max"] or m_dot_air_lp_corr > lp_p["flow_limit"]:
                        continue
                    lp_flow_surge = lp_p["surge_coeff"] * np.sqrt(max(PR_c_stage_req - 1.0, 0.0))
                    if m_dot_air_lp_corr < lp_flow_surge:
                        continue
                        
                    for hp_name, hp_p in TURBO_LIBRARY.items():
                        if lp_p["flow_limit"] < hp_p["flow_limit"]: # LP must be larger or equal to HP
                            continue
                        if PR_c_stage_req > hp_p["pr_c_max"] or m_dot_air_hp_corr > hp_p["flow_limit"]:
                            continue
                        hp_flow_surge = hp_p["surge_coeff"] * np.sqrt(max(PR_c_stage_req - 1.0, 0.0))
                        if m_dot_air_hp_corr < hp_flow_surge:
                            continue
                            
                        # Evaluate LP and HP combinations over A/R options
                        for lp_ar, lp_ar_data in lp_p["ar_options"].items():
                            for hp_ar, hp_ar_data in hp_p["ar_options"].items():
                                lp_flow_ratio = m_dot_air_lp_corr / lp_p["flow_limit"]
                                lp_comp_score = 100.0 - abs(lp_flow_ratio - 0.60) * 100.0
                                
                                hp_flow_ratio = m_dot_air_hp_corr / hp_p["flow_limit"]
                                hp_comp_score = 100.0 - abs(hp_flow_ratio - 0.60) * 100.0
                                
                                total_score = 0.5 * lp_comp_score + 0.5 * hp_comp_score
                                total_score = max(min(total_score, 100.0), 0.0)
                                
                                recommendations.append({
                                    "lp_name": lp_name,
                                    "lp_ar": lp_ar,
                                    "hp_name": hp_name,
                                    "hp_ar": hp_ar,
                                    "score": total_score,
                                    "lp_flow_ratio": lp_flow_ratio,
                                    "hp_flow_ratio": hp_flow_ratio
                                })
                                
                if len(recommendations) == 0:
                    st.error("❌ متاسفانه هیچ زوج توربوشارژری در کتابخانه فعلی با این شرایط مرزی منطبق نیست.")
                else:
                    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)
                    st.subheader("🏆 پیشنهادات بهینه اولویت‌بندی شده برای سیستم دو مرحله‌ای:")
                    for idx, rec in enumerate(recommendations[:3]):
                        st.markdown(f"### رتبه {idx+1}: HP: {rec['hp_name']} (A/R: {rec['hp_ar']}) & LP: {rec['lp_name']} (A/R: {rec['lp_ar']})")
                        st.write(f"• **درصد تطبیق فیزیکی کل سیستم:** `{rec['score']:.1f}%`")
                        st.write(f"  * **تحلیل عملکرد توربو LP (کم فشار):** نقطه کاری در دبی `{m_dot_air_lp_corr:.2f} kg/min` معادل `{rec['lp_flow_ratio']*100:.1f}%` ظرفیت LP کمپرسور است.")
                        st.write(f"  * **تحلیل عملکرد توربو HP (پر فشار):** نقطه کاری در دبی `{m_dot_air_hp_corr:.2f} kg/min` معادل `{rec['hp_flow_ratio']*100:.1f}%` ظرفیت HP کمپرسور است.")
                        st.markdown("---")
                        
    else:
        st.subheader("📋 Garrett Performance Catalog Specifications")
        if turbo_type in TURBO_LIBRARY:
            tp = TURBO_LIBRARY[turbo_type]
            spec_df = pd.DataFrame({
                "Parameter Name (پارامتر)": [
                    "Turbocharger Model", "Brand / Series", "Compressor Inducer Diameter",
                    "Compressor Exducer Diameter", "Compressor Wheel Trim", "Max Airflow Capacity",
                    "Max Speed Rating", "Peak Compressor Efficiency", "Turbine Wheel Inducer",
                    "Turbine Wheel Exducer", "Selected Housing A/R", "Available A/R Options"
                ],
                "Catalog Specification (مقدار کاتالوگ)": [
                    turbo_type, tp.get("brand", "Garrett"), f"{tp.get('inducer_mm', 'N/A')} mm",
                    f"{tp.get('exducer_mm', 'N/A')} mm", f"{tp.get('trim', 'N/A')}",
                    f"{tp.get('flow_limit_lb_min', 'N/A')} lb/min ({tp.get('flow_limit', 'N/A')} kg/min)",
                    f"{tp.get('N_max', 0.0)/1000:.0f}k RPM", f"{tp.get('eta_c', 0.78)*100:.0f}%",
                    f"{tp.get('turbine_inducer_mm', 'N/A')} mm", f"{tp.get('turbine_exducer_mm', 'N/A')} mm",
                    f"{selected_ar} A/R", f"{', '.join(tp.get('ar_options', {}).keys())} A/R"
                ]
            })
            st.table(spec_df)
        st.markdown("---")
        st.subheader("Brake & Turbocharger Matched State Indicators")
        if turbo_stages == "Dual-Stage (دو مرحله ای)":
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
            col_t1.metric("HP Compressor PR", f"{matched_pr_c_hp:.2f}")
            col_t2.metric("HP Turbine PR", f"{matched_pr_t_hp:.2f}")
            col_t3.metric("LP Compressor PR", f"{matched_pr_c_lp:.2f}")
            col_t4.metric("LP Turbine PR", f"{matched_pr_t_lp:.2f}")
            col_t5.metric("Corrected Flow (m_corr)", f"{matched_airflow_kg_min:.2f} kg/min")
            
            col_t1_b, col_t2_b, col_t3_b, col_t4_b, col_t5_b = st.columns(5)
            col_t1_b.metric("HP Turbo Speed", f"{matched_turbo_rpm_hp/1000:.1f}k RPM")
            col_t2_b.metric("LP Turbo Speed", f"{matched_turbo_rpm_lp/1000:.1f}k RPM")
            col_t5_b.metric("Wastegate State", "Open (دریچه باز)" if is_wastegate_open else "Closed (بسته)")
        else:
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
            col_t1.metric("Compressor PR", f"{matched_pr_c:.2f}")
            col_t2.metric("Turbine Expansion Ratio", f"{matched_pr_t:.2f}")
            col_t3.metric("Corrected Flow (m_corr)", f"{matched_airflow_kg_min:.2f} kg/min")
            col_t4.metric("Matched Turbo Speed", f"{matched_turbo_rpm/1000:.1f}k RPM")
            col_t5.metric("Wastegate State", "Open (دریچه باز)" if is_wastegate_open else "Closed (بسته)")
            
        st.markdown("---")
        
        # Draw maps
        if turbo_stages == "Single-Stage (تک مرحله ای)":
            col_m_l, col_m_r = st.columns(2)
            with col_m_l:
                fig_c, ax_c = plt.subplots(figsize=(7, 5))
                draw_compressor_map_axes(ax_c, TURBO_LIBRARY[turbo_type], matched_airflow_kg_min, matched_pr_c, "Single-Stage")
                st.pyplot(fig_c)
            with col_m_r:
                fig_t, ax_t = plt.subplots(figsize=(7, 5))
                matched_ex_corr_flow = matched_airflow_kg_min * (1.0 + 1.0/afr_val) * np.sqrt(matched_egt_k / 288.15) / (p_amb_pa / 101325.0)
                draw_turbine_map_axes(ax_t, TURBO_LIBRARY[turbo_type]["ar_options"][selected_ar]["m_t_limit"], TURBO_LIBRARY[turbo_type]["ar_options"][selected_ar]["k_t"], matched_pr_t, matched_ex_corr_flow, "Single-Stage")
                st.pyplot(fig_t)
        else: # Dual-Stage Maps Rendering
            col_l1, col_r1 = st.columns(2)
            with col_l1:
                fig_c_hp, ax_c_hp = plt.subplots(figsize=(7, 5))
                draw_compressor_map_axes(ax_c_hp, TURBO_LIBRARY[hp_turbo_type], matched_airflow_kg_min, matched_pr_c_hp, "Stage 1 (HP)")
                st.pyplot(fig_c_hp)
            with col_r1:
                fig_c_lp, ax_c_lp = plt.subplots(figsize=(7, 5))
                draw_compressor_map_axes(ax_c_lp, TURBO_LIBRARY[lp_turbo_type], matched_airflow_kg_min, matched_pr_c_lp, "Stage 2 (LP)")
                st.pyplot(fig_c_lp)
                
            col_l2, col_r2 = st.columns(2)
            with col_l2:
                fig_t_hp, ax_t_hp = plt.subplots(figsize=(7, 5))
                # HP Turbine corrected flow
                m_dot_ex_corr_hp_final = matched_airflow_kg_min * (1.0 + 1.0/afr_val) * np.sqrt(matched_egt_k / 288.15) / (p_ex_pa / 101325.0)
                draw_turbine_map_axes(ax_t_hp, TURBO_LIBRARY[hp_turbo_type]["ar_options"][hp_selected_ar]["m_t_limit"], TURBO_LIBRARY[hp_turbo_type]["ar_options"][hp_selected_ar]["k_t"], matched_pr_t_hp, m_dot_ex_corr_hp_final, "Stage 1 (HP)")
                st.pyplot(fig_t_hp)
            with col_r2:
                fig_t_lp, ax_t_lp = plt.subplots(figsize=(7, 5))
                # LP Turbine corrected flow
                T_egt_lp_final = matched_egt_k * (1.0 - TURBO_LIBRARY[hp_turbo_type]["eta_t"] * (1.0 - (1.0 / matched_pr_t_hp) ** (0.33/1.33)))
                p_ex_mid_final = p_amb_pa * matched_pr_t_lp
                m_dot_ex_corr_lp_final = matched_airflow_kg_min * (1.0 + 1.0/afr_val) * np.sqrt(T_egt_lp_final / 288.15) / (p_ex_mid_final / 101325.0)
                draw_turbine_map_axes(ax_t_lp, TURBO_LIBRARY[lp_turbo_type]["ar_options"][lp_selected_ar]["m_t_limit"], TURBO_LIBRARY[lp_turbo_type]["ar_options"][lp_selected_ar]["k_t"], matched_pr_t_lp, m_dot_ex_corr_lp_final, "Stage 2 (LP)")
                st.pyplot(fig_t_lp)
                
        st.markdown("---")
        st.subheader("⚠️ تحلیل پویای مرز فروپاشی بالانس انرژی توربوشارژر (Turbo Enthalpy Collapse Boundary)")
        st.markdown("""
        **چالش فیزیکی کسری آنتالپی اگزوز (دود بی‌جون):**  
        یکی از بزرگ‌ترین نقاط ضعف شبیه‌سازهای تجاری متداول، عدم توانایی در پیش‌بینی دقیق مرزهایی است که بالانس انرژی روی شفت توربوشارژر به دلیل افت شدید آنتالپی اگزوز («دود بی‌جون») فرو می‌پاشد. در این سناریو، ممکن است موتور در ارتفاع بالا (مانند ۳۰,۰۰۰ پا) قرار داشته باشد و شما تراتل را باز کنید تا توان بیشتری بگیرید، اما دور توربوشارژر افزایش نمی‌یابد، زیرا دبی جرمی جریان اگزوز و دمای منیفولد خروجی توانایی غلبه بر کار مورد نیاز کمپرسور برای بوست هدف را ندارند.
        
        کد پیشرفته حاضر این پدیده را با فرمول‌بندی **نسبت بالانس انرژی توربوشارژر (TEBR)** مدلسازی می‌کند:
        $$TEBR = \\frac{\\dot{W}_{t, max\\_avail} \\cdot \\eta_{mech}}{\\dot{W}_{c, required}}$$
        
        *   **اگر $TEBR \\ge 1.0$ (محدوده پایدار):** کار کمپرسور تامین شده و شیر ویست‌گیت برای قفل کردن فشار منیفولد در **۹۹۰ میلی‌بار** (یا فشار هدف شما) باز می‌شود.
        *   **اگر $TEBR < 1.0$ (محدوده فروپاشی بالانس / دود بی‌جون):** توربین دچار کسری کار شده و ویست‌گیت کاملاً بسته است، اما دور توربو افزایش نمی‌یابد. این ناحیه مرزهای واقعی کارکرد موتور پرخوران را مشخص می‌کند.
        """)
        
        with st.spinner("در حال تحلیل و نقشه‌برداری مرزهای بالانس انرژی توربو..."):
            alts_ft = np.linspace(0.0, 75000.0, 15)
            throttles = np.linspace(10.0, 100.0, 15)
            tebr_mat = np.zeros((len(throttles), len(alts_ft)))
            wg_mat = np.zeros((len(throttles), len(alts_ft)))
            
            for i, t_val in enumerate(throttles):
                for j, alt_val in enumerate(alts_ft):
                    p_amb_local_bar, t_amb_local, p_amb_local_pa = get_isa_properties(alt_val)
                    
                    p_in_target_pa = min(p_amb_local_pa + target_boost_bar * 1e5, 99000.0)
                    pr_c_total_target_sw = p_in_target_pa / p_amb_local_pa
                    
                    if turbo_stages == "Single-Stage (تک مرحله ای)":
                        pr_c_sw = 1.0
                        pr_t_sw = 1.0
                        t_in_sw = t_amb_local
                        
                        for step in range(3):
                            t_c_out = t_amb_local * (1.0 + (pr_c_sw ** 0.286 - 1.0) / eta_c)
                            t_in_sw = t_c_out - ic_eff * (t_c_out - (t_amb_local + 10.0))
                            t_in_sw = max(t_in_sw, t_amb_local)
                            
                            p_c_out = p_amb_local_pa * pr_c_sw
                            if throttle_lut is not None:
                                p_ratio = get_ratio_local(t_val, throttle_lut)
                                p_in_sw = p_c_out * p_ratio
                            else:
                                p_in_sw = p_c_out * (0.25 + 0.75 * ((t_val / 100.0) ** 1.5))
                            p_ex_sw = p_amb_local_pa * pr_t_sw
                            
                            iter_sim_sw = WankelSimulator0D_V14(
                                R=R, e=e, b=b, r_c=r_c, N=N_rpm,
                                p_ambient_bar=p_amb_local_bar, T_ambient_k=t_amb_local, T_wall=T_wall_k,
                                LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
                                tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
                                ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
                                spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=t_val,
                                throttle_lut=throttle_lut, p_in_override=p_in_sw, T_in_override=t_in_sw, p_ex_override=p_ex_sw
                            )
                            _, iter_sum_sw = iter_sim_sw.simulate(fire=True)
                            m_dot_air = iter_sum_sw['m_fresh_trapped_kg'] * N_rpm * (num_rotors / 60.0)
                            m_dot_ex = m_dot_air * (1.0 + 1.0 / afr_val)
                            T_egt = iter_sum_sw['T_end_expansion_K'] * 0.88
                            T_egt = max(T_egt, 350.0)
                            m_dot_ex_min = m_dot_ex * 60.0
                            m_dot_ex_corr = m_dot_ex_min * np.sqrt(T_egt / 288.15) / (p_amb_local_pa / 101325.0)
                            flow_ratio = min(m_dot_ex_corr / m_t_limit, 0.96)
                            pr_t_matched_sw = 1.0 - (1.0 / k_t) * np.log(1.0 - flow_ratio)
                            W_dot_t = m_dot_ex * Cp_ex * T_egt * (1.0 - (1.0 / pr_t_matched_sw) ** (0.248)) * eta_t
                            W_dot_c = W_dot_t * eta_mech
                            T_c_out_matched = t_amb_local + W_dot_c / (max(m_dot_air, 1e-4) * Cp_gas)
                            pr_c_matched_sw = (1.0 + eta_c * (T_c_out_matched - t_amb_local) / t_amb_local) ** 3.5
                            if pr_c_matched_sw > pr_c_target_sw:
                                pr_c_sw = pr_c_target_sw
                                W_dot_c_req = m_dot_air * Cp_gas * (t_amb_local * (1.0 + (pr_c_sw ** 0.286 - 1.0) / eta_c) - t_amb_local)
                                W_dot_t_req = W_dot_c_req / eta_mech
                                term = W_dot_t_req / (max(m_dot_ex, 1e-4) * Cp_ex * T_egt * eta_t)
                                term = min(max(term, 0.0), 0.9)
                                pr_t_sw = (1.0 / (1.0 - term)) ** 4.03
                            else:
                                pr_c_sw = pr_c_matched_sw
                                pr_t_sw = pr_t_matched_sw
                        W_dot_c_req_final = m_dot_air * Cp_gas * (t_amb_local * (1.0 + (pr_c_target_sw ** 0.286 - 1.0) / eta_c) - t_amb_local)
                        W_dot_t_max_avail = m_dot_ex * Cp_ex * T_egt * (1.0 - (1.0 / pr_t_matched_sw) ** 0.248) * eta_t
                        tebr_val = (W_dot_t_max_avail * eta_mech) / max(W_dot_c_req_final, 1.0)
                    else: # Dual-Stage
                        pr_c_lp_sw = np.sqrt(pr_c_total_target_sw)
                        pr_c_hp_sw = np.sqrt(pr_c_total_target_sw)
                        pr_t_lp_sw = 1.25
                        pr_t_hp_sw = 1.25
                        t_in_sw = t_amb_local
                        
                        for step in range(3):
                            t_c_out_lp = t_amb_local * (1.0 + (pr_c_lp_sw ** 0.286 - 1.0) / lp_eta_c)
                            t_in_hp_sw = t_c_out_lp - ic_eff * (t_c_out_lp - (t_amb_local + 10.0))
                            t_in_hp_sw = max(t_in_hp_sw, t_amb_local)
                            
                            t_c_out_hp = t_in_hp_sw * (1.0 + (pr_c_hp_sw ** 0.286 - 1.0) / hp_eta_c)
                            t_in_sw = t_c_out_hp - ic_eff * (t_c_out_hp - (t_amb_local + 10.0))
                            t_in_sw = max(t_in_sw, t_amb_local)
                            
                            p_c_out_total = p_amb_local_pa * pr_c_lp_sw * pr_c_hp_sw
                            if throttle_lut is not None:
                                p_ratio = get_ratio_local(t_val, throttle_lut)
                                p_in_sw = p_c_out_total * p_ratio
                            else:
                                p_in_sw = p_c_out_total * (0.25 + 0.75 * ((t_val / 100.0) ** 1.5))
                            p_ex_sw = p_amb_local_pa * pr_t_lp_sw * pr_t_hp_sw
                            
                            iter_sim_sw = WankelSimulator0D_V14(
                                R=R, e=e, b=b, r_c=r_c, N=N_rpm,
                                p_ambient_bar=p_amb_local_bar, T_ambient_k=t_amb_local, T_wall=T_wall_k,
                                LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
                                tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
                                ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
                                spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=t_val,
                                throttle_lut=throttle_lut, p_in_override=p_in_sw, T_in_override=t_in_sw, p_ex_override=p_ex_sw
                            )
                            _, iter_sum_sw = iter_sim_sw.simulate(fire=True)
                            
                            m_dot_air = iter_sum_sw['m_fresh_trapped_kg'] * N_rpm * (num_rotors / 60.0)
                            m_dot_ex = m_dot_air * (1.0 + 1.0 / afr_val)
                            T_egt = iter_sum_sw['T_end_expansion_K'] * 0.88
                            T_egt = max(T_egt, 350.0)
                            m_dot_ex_min = m_dot_ex * 60.0
                            
                            m_dot_ex_corr_hp = m_dot_ex_min * np.sqrt(T_egt / 288.15) / (p_ex_sw / 101325.0)
                            flow_ratio_hp = min(m_dot_ex_corr_hp / hp_m_t_limit, 0.96)
                            pr_t_matched_hp = 1.0 - (1.0 / hp_k_t) * np.log(1.0 - flow_ratio_hp)
                            pr_t_matched_hp = max(pr_t_matched_hp, 1.05)
                            
                            T_egt_lp = T_egt * (1.0 - hp_eta_t * (1.0 - (1.0 / pr_t_matched_hp) ** (0.33/1.33)))
                            T_egt_lp = max(T_egt_lp, 300.0)
                            p_ex_mid = p_amb_local_pa * pr_t_lp_sw
                            m_dot_ex_corr_lp = m_dot_ex_min * np.sqrt(T_egt_lp / 288.15) / (p_ex_mid / 101325.0)
                            flow_ratio_lp = min(m_dot_ex_corr_lp / lp_m_t_limit, 0.96)
                            pr_t_matched_lp = 1.0 - (1.0 / lp_k_t) * np.log(1.0 - flow_ratio_lp)
                            pr_t_matched_lp = max(pr_t_matched_lp, 1.05)
                            
                            W_dot_t_hp = m_dot_ex * Cp_ex * T_egt * (1.0 - (1.0 / pr_t_matched_hp) ** (0.33/1.33)) * hp_eta_t
                            W_dot_c_hp = W_dot_t_hp * eta_mech
                            T_c_out_matched_hp = t_in_hp_sw + W_dot_c_hp / (max(m_dot_air, 1e-4) * Cp_gas)
                            pr_c_matched_hp = (1.0 + hp_eta_c * (T_c_out_matched_hp - t_in_hp_sw) / t_in_hp_sw) ** 3.5
                            
                            W_dot_t_lp = m_dot_ex * Cp_ex * T_egt_lp * (1.0 - (1.0 / pr_t_matched_lp) ** (0.33/1.33)) * lp_eta_t
                            W_dot_c_lp = W_dot_t_lp * eta_mech
                            T_c_out_matched_lp = t_amb_local + W_dot_c_lp / (max(m_dot_air, 1e-4) * Cp_gas)
                            pr_c_matched_lp = (1.0 + lp_eta_c * (T_c_out_matched_lp - t_amb_local) / t_amb_local) ** 3.5
                            
                            pr_c_matched_total = pr_c_matched_lp * pr_c_matched_hp
                            
                            if pr_c_matched_total > pr_c_total_target_sw:
                                pr_c_total_target_sw = min(pr_c_total_target_sw, lp_pr_c_max * hp_pr_c_max)
                                pr_c_lp_sw = min(np.sqrt(pr_c_total_target_sw), lp_pr_c_max)
                                pr_c_hp_sw = pr_c_total_target_sw / pr_c_lp_sw
                                
                                W_dot_c_hp_req = m_dot_air * Cp_gas * (t_in_hp_sw * (1.0 + (pr_c_hp_sw ** 0.286 - 1.0) / hp_eta_c) - t_in_hp_sw)
                                W_dot_t_hp_req = W_dot_c_hp_req / eta_mech
                                term_hp = W_dot_t_hp_req / (max(m_dot_ex, 1e-4) * Cp_ex * T_egt * hp_eta_t)
                                term_hp = min(max(term_hp, 0.0), 0.9)
                                pr_t_hp_sw = (1.0 / (1.0 - term_hp)) ** 4.03
                                pr_t_hp_sw = max(pr_t_hp_sw, 1.05)
                                
                                T_egt_lp_req = T_egt * (1.0 - hp_eta_t * (1.0 - (1.0 / pr_t_hp_sw) ** (0.33/1.33)))
                                W_dot_c_lp_req = m_dot_air * Cp_gas * (t_amb_local * (1.0 + (pr_c_lp_sw ** 0.286 - 1.0) / lp_eta_c) - t_amb_local)
                                W_dot_t_lp_req = W_dot_c_lp_req / eta_mech
                                term_lp = W_dot_t_lp_req / (max(m_dot_ex, 1e-4) * Cp_ex * T_egt_lp_req * lp_eta_t)
                                term_lp = min(max(term_lp, 0.0), 0.9)
                                pr_t_lp_sw = (1.0 / (1.0 - term_lp)) ** 4.03
                                pr_t_lp_sw = max(pr_t_lp_sw, 1.05)
                            else:
                                pr_c_lp_sw = pr_c_matched_lp
                                pr_c_hp_sw = pr_c_matched_hp
                                pr_t_lp_sw = pr_t_matched_lp
                                pr_t_hp_sw = pr_t_matched_hp
                        
                        W_dot_c_lp_req_f = m_dot_air * Cp_gas * (t_amb_local * (1.0 + (min(np.sqrt(p_in_target_pa/p_amb_local_pa), lp_pr_c_max) ** 0.286 - 1.0) / lp_eta_c) - t_amb_local)
                        t_in_hp_req_f = t_amb_local * (1.0 + (min(np.sqrt(p_in_target_pa/p_amb_local_pa), lp_pr_c_max) ** 0.286 - 1.0) / lp_eta_c)
                        t_in_hp_req_f = t_in_hp_req_f - ic_eff * (t_in_hp_req_f - (t_amb_local + 10.0))
                        W_dot_c_hp_req_f = m_dot_air * Cp_gas * (t_in_hp_req_f * (1.0 + (max(1.0, (p_in_target_pa/p_amb_local_pa)/min(np.sqrt(p_in_target_pa/p_amb_local_pa), lp_pr_c_max)) ** 0.286 - 1.0) / hp_eta_c) - t_in_hp_req_f)
                        W_dot_c_req_final_total = W_dot_c_lp_req_f + W_dot_c_hp_req_f
                        
                        W_dot_t_hp_max = m_dot_ex * Cp_ex * T_egt * (1.0 - (1.0 / pr_t_matched_hp) ** (0.33/1.33)) * hp_eta_t
                        W_dot_t_lp_max = m_dot_ex * Cp_ex * T_egt_lp * (1.0 - (1.0 / pr_t_matched_lp) ** (0.33/1.33)) * lp_eta_t
                        W_dot_t_max_avail_total = (W_dot_t_hp_max + W_dot_t_lp_max) * eta_mech
                        tebr_val = W_dot_t_max_avail_total / max(W_dot_c_req_final_total, 1.0)
                    
                    tebr_mat[i, j] = tebr_val
            
            # Plot the TEBR Map
            fig_env, ax_env = plt.subplots(figsize=(9, 5.5))
            A_grid, T_grid = np.meshgrid(alts_ft, throttles)
            
            regime = np.zeros_like(tebr_mat)
            regime[tebr_mat >= 1.0] = 2.0
            regime[(tebr_mat < 1.0) & (tebr_mat >= 0.7)] = 1.0
            regime[tebr_mat < 0.7] = 0.0
            
            cp_env = ax_env.contourf(A_grid, T_grid, tebr_mat, levels=[0.0, 0.7, 1.0, 2.5, 10.0], 
                                     colors=['#ffcccc', '#ffe5cc', '#d4edda', '#c3e6cb'], alpha=0.9)
            
            boundary = ax_env.contour(A_grid, T_grid, tebr_mat, levels=[1.0], colors='darkgreen', linewidths=3.0)
            ax_env.clabel(boundary, inline=True, fontsize=10, fmt={1.0: "Energy Balance Limit"})
            
            collapse = ax_env.contour(A_grid, T_grid, tebr_mat, levels=[0.7], colors='darkred', linestyles='--', linewidths=2.0)
            ax_env.clabel(collapse, inline=True, fontsize=10, fmt={0.7: "Starvation Limit"})
            
            ax_env.text(2000, 80, "Regulated Boost Zone\n(ناحیه بوست پایدار - بوست ۹۹۰ میلی‌بار)", color='darkgreen', fontsize=10, fontweight='bold')
            ax_env.text(35000, 40, "Transient Stagnation Zone\n(ناحیه افت دور توربو - بوست کاهش می‌یابد)", color='brown', fontsize=9, fontweight='bold')
            ax_env.text(55000, 15, "Enthalpy Collapse Zone\n(فروپاشی کامل بالانس انرژی)", color='darkred', fontsize=9, fontweight='bold')
            
            # Current State calculation
            if turbo_stages == "Single-Stage (تک مرحله ای)":
                current_p_in_target_pa = min(p_amb_pa + target_boost_bar * 1e5, 99000.0)
                current_pr_c_target = current_p_in_target_pa / p_amb_pa
                current_W_dot_c_req = matched_airflow_kg_min/60.0 * Cp_gas * (t_amb_k * (1.0 + (current_pr_c_target ** 0.286 - 1.0)/eta_c) - t_amb_k)
                current_matched_ex_flow = matched_airflow_kg_min/60.0 * (1.0 + 1.0/afr_val)
                current_W_dot_t_max = current_matched_ex_flow * Cp_ex * matched_egt_k * (1.0 - (1.0 / matched_pr_t) ** 0.248) * eta_t
                current_tebr = (current_W_dot_t_max * eta_mech) / max(current_W_dot_c_req, 1.0)
            else: # Dual-Stage
                current_p_in_target_pa = min(p_amb_pa + target_boost_bar * 1e5, 99000.0)
                current_pr_c_target = current_p_in_target_pa / p_amb_pa
                current_W_dot_c_lp_req = matched_airflow_kg_min/60.0 * Cp_gas * (t_amb_k * (1.0 + (min(np.sqrt(current_pr_c_target), lp_pr_c_max) ** 0.286 - 1.0)/lp_eta_c) - t_amb_k)
                t_in_hp_curr = t_amb_k * (1.0 + (min(np.sqrt(current_pr_c_target), lp_pr_c_max) ** 0.286 - 1.0)/lp_eta_c)
                t_in_hp_curr = t_in_hp_curr - ic_eff * (t_in_hp_curr - (t_amb_k + 10.0))
                current_W_dot_c_hp_req = matched_airflow_kg_min/60.0 * Cp_gas * (t_in_hp_curr * (1.0 + (max(1.0, current_pr_c_target/min(np.sqrt(current_pr_c_target), lp_pr_c_max)) ** 0.286 - 1.0)/hp_eta_c) - t_in_hp_curr)
                current_W_dot_c_req_total = current_W_dot_c_lp_req + current_W_dot_c_hp_req
                
                current_matched_ex_flow = matched_airflow_kg_min/60.0 * (1.0 + 1.0/afr_val)
                W_dot_t_hp_max = current_matched_ex_flow * Cp_ex * matched_egt_k * (1.0 - (1.0 / matched_pr_t_hp) ** (0.33/1.33)) * hp_eta_t
                T_egt_lp_curr = matched_egt_k * (1.0 - hp_eta_t * (1.0 - (1.0 / matched_pr_t_hp) ** (0.33/1.33)))
                W_dot_t_lp_max = current_matched_ex_flow * Cp_ex * T_egt_lp_curr * (1.0 - (1.0 / matched_pr_t_lp) ** (0.33/1.33)) * lp_eta_t
                current_W_dot_t_max_total = (W_dot_t_hp_max + W_dot_t_lp_max) * eta_mech
                current_tebr = current_W_dot_t_max_total / max(current_W_dot_c_req_total, 1.0)
                
            current_alt_ft = altitude_ft if ambient_mode == "Altitude-based (ارتفاع از سطح دریا)" else 0.0
            ax_env.plot(current_alt_ft, throttle_pct, 'g*', markersize=16, label=f"Current State (TEBR: {current_tebr:.2f})")
            
            ax_env.set_xlabel("Flight Altitude (ft)")
            ax_env.set_ylabel("Throttle Opening (%)")
            ax_env.set_title(f"Turbocharger Exergy Balance Map at {N_rpm:.0f} RPM", fontweight="bold", fontsize=11)
            ax_env.grid(True, linestyle=":", alpha=0.6)
            ax_env.legend(loc='lower left')
            st.pyplot(fig_env)
            
            if current_tebr >= 1.0:
                st.success(f"✅ **پایداری بالانس انرژی:** موتور در شرایط کاری کنونی دارای نسبت بالانس انرژی {current_tebr:.2f} است. اگزوز توانایی عالی برای راندن توربو را دارد و شیر ویست‌گیت برای حفظ فشار منیفولد در محدوده مجاز ۹۹۰ میلی‌بار به میزان {((current_tebr-1.0)/current_tebr*100):.1f} درصد باز خواهد شد.")
            elif current_tebr >= 0.7:
                st.warning(f"⚠️ **آستانه خفگی ترمودینامیکی (Stagnation Zone):** نسبت بالانس انرژی به {current_tebr:.2f} کاهش یافته است. توربین در بیشترین توان خود قرار دارد (ویست‌گیت کاملاً بسته است)، اما برای حفظ منیفولد در فشار هدف کسری توان دارد. افزایش تراتل در این ناحیه دور توربو را بالا نخواهد برد و موتور دچار تاخیر عملکردی یا افت دور توربو خواهد شد.")
            else:
                st.error(f"❌ **فروپاشی کامل بالانس انرژی توربو (Enthalpy Collapse):** نسبت بالانس انرژی به {current_tebr:.2f} سقوط کرده است! در این شرایط کاری («دود موتور بی‌جون است»)، توربو کاملاً استال/قفل شده و سامانه مانند یک موتور تنفس طبیعی ضعیف رفتار خواهد کرد.")



with tab_sweeps:
    st.subheader("📊 Parametric Sweeps & Engine Mapping (تحلیل پارامتریک و نقشه‌برداری متغیرها)")
    st.markdown("پیکربندی هوشمند و تعیین بازه‌های دلخواه محاسباتی جهت استخراج منحنی‌های عملکردی پیشرانه:")
    
    sweep_type = st.selectbox("Select Sweep Type (نوع تحلیل پارامتریک)", [
        "1D Throttle Sweep (تحلیل یک بعدی دریچه گاز)",
        "1D Altitude Sweep (تحلیل یک بعدی ارتفاع)",
        "1D Engine Speed Sweep (تحلیل یک بعدی دور موتور - RPM)",
        "2D Combined Sweep (تحلیل دو بعدی ارتفاع و دریچه گاز)",
        "RPM-Throttle Map at Fixed Altitude (نقشه دور و دریچه گاز در ارتفاع ثابت)"
    ])
    
    st.markdown("---")
    
    # Range Selectors based on sweep_type
    if sweep_type == "1D Throttle Sweep (تحلیل یک بعدی دریچه گاز)":
        c_sw1, c_sw2, c_sw3 = st.columns(3)
        with c_sw1:
            sw_throttle_min = st.number_input("کمترین درصد تراتل (%)", min_value=0.0, max_value=100.0, value=10.0, step=5.0)
        with c_sw2:
            sw_throttle_max = st.number_input("بیشترین درصد تراتل (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)
        with c_sw3:
            sw_throttle_pts = st.slider("تعداد نقاط گام محاسباتی", min_value=3, max_value=25, value=10)
            
    elif sweep_type == "1D Altitude Sweep (تحلیل یک بعدی ارتفاع)":
        c_sw1, c_sw2, c_sw3 = st.columns(3)
        with c_sw1:
            sw_alt_min = st.number_input("کمترین ارتفاع (ft)", min_value=0.0, max_value=80000.0, value=0.0, step=1000.0)
        with c_sw2:
            sw_alt_max = st.number_input("بیشترین ارتفاع (ft)", min_value=0.0, max_value=80000.0, value=50000.0, step=1000.0)
        with c_sw3:
            sw_alt_pts = st.slider("تعداد نقاط ارتفاعی", min_value=3, max_value=25, value=11)
            
    elif sweep_type == "1D Engine Speed Sweep (تحلیل یک بعدی دور موتور - RPM)":
        c_sw1, c_sw2, c_sw3 = st.columns(3)
        with c_sw1:
            sw_rpm_min = st.number_input("کمترین دور موتور (RPM)", min_value=500, max_value=15000, value=1500, step=100)
        with c_sw2:
            sw_rpm_max = st.number_input("بیشترین دور موتور (RPM)", min_value=500, max_value=15000, value=8000, step=500)
        with c_sw3:
            sw_rpm_step = st.selectbox("گام دور موتور (RPM Step)", [100, 250, 500, 1000], index=2)
            
    elif sweep_type == "2D Combined Sweep (تحلیل دو بعدی ارتفاع و دریچه گاز)":
        c_sw1, c_sw2, c_sw3, c_sw4 = st.columns(4)
        with c_sw1:
            sw_alt_min_2d = st.number_input("کمترین ارتفاع (ft)", min_value=0.0, max_value=80000.0, value=0.0, step=1000.0)
        with c_sw2:
            sw_alt_max_2d = st.number_input("بیشترین ارتفاع (ft)", min_value=0.0, max_value=80000.0, value=50000.0, step=5000.0)
        with c_sw3:
            sw_th_min_2d = st.number_input("کمترین تراتل (%)", min_value=10.0, max_value=100.0, value=20.0, step=10.0)
        with c_sw4:
            sw_th_max_2d = st.number_input("بیشترین تراتل (%)", min_value=10.0, max_value=100.0, value=100.0, step=10.0)
            
    else: # RPM-Throttle Map
        c_sw1, c_sw2, c_sw3, c_sw4 = st.columns(4)
        with c_sw1:
            sw_rpm_min_map = st.number_input("کمترین دور (RPM)", min_value=1000, max_value=12000, value=2000, step=500)
        with c_sw2:
            sw_rpm_max_map = st.number_input("بیشترین دور (RPM)", min_value=1000, max_value=12000, value=7000, step=500)
        with c_sw3:
            sw_th_min_map = st.number_input("کمترین تراتل (%)", min_value=10.0, max_value=100.0, value=20.0, step=10.0)
        with c_sw4:
            sw_th_max_map = st.number_input("بیشترین تراتل (%)", min_value=10.0, max_value=100.0, value=100.0, step=10.0)

    btn_run_sweep = st.button("▶️ اجرای شبیه‌سازی و ترسیم تحلیل پارامتریک (Run Parametric Sweep)", type="primary")
    
    # Sweep calculation helper using V14 logic with active/inactive turbocharger
    def run_sweep_point_v14(p_amb_local_bar, t_amb_local, throttle_local, rpm_local=N_rpm):
        p_amb_local_pa = p_amb_local_bar * 1e5
        
        if enable_turbo:
            eff_pr_c = max(matched_pr_c, 1.0)
            eff_pr_t = max(matched_pr_t, 1.0)
            
            p_c_out = p_amb_local_pa * eff_pr_c
            t_c_out = t_amb_local * (1.0 + (eff_pr_c ** 0.286 - 1.0) / 0.75)
            t_in_active = t_c_out - ic_eff * (t_c_out - (t_amb_local + 10.0))
            t_in_active = max(t_in_active, t_amb_local)
            
            if throttle_mode == "Analytical Formula (رابطه تحلیلی یاماموتو)":
                p_ratio_loc = 0.25 + 0.75 * ((throttle_local / 100.0) ** 1.5)
            elif throttle_lut is not None:
                p_ratio_loc = get_ratio_local(throttle_local, throttle_lut)
            else:
                p_ratio_loc = get_ratio_local(throttle_local, EXPERIMENTAL_THROTTLE_LUT)
                
            p_in_active = p_c_out * p_ratio_loc
            p_ex_active = p_amb_local_pa * eff_pr_t
        else:
            if throttle_mode == "Analytical Formula (رابطه تحلیلی یاماموتو)":
                p_ratio_loc = 0.25 + 0.75 * ((throttle_local / 100.0) ** 1.5)
            elif throttle_lut is not None:
                p_ratio_loc = get_ratio_local(throttle_local, throttle_lut)
            else:
                p_ratio_loc = get_ratio_local(throttle_local, EXPERIMENTAL_THROTTLE_LUT)
                
            p_in_active = p_amb_local_pa * p_ratio_loc
            t_in_active = t_amb_local
            p_ex_active = p_amb_local_pa

        sweep_sim = WankelSimulator0D_V14(
            R=R, e=e, b=b, r_c=r_c, N=rpm_local,
            p_ambient_bar=p_amb_local_bar, T_ambient_k=t_amb_local, T_wall=T_wall_k,
            LHV=lhv_val, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=crevol_cc,
            tipo=tipo, tipc=tipc, tepo=tepo, tepc=tepc,
            ipa_cm2=ipa_cm2, epa_cm2=epa_cm2, tspark=tspark, arealk_cm2=arealk_cm2,
            spark_plugs=spark_plugs_count, comb_dur_deg=comb_dur_deg, throttle_pct=throttle_local,
            throttle_lut=throttle_lut, p_in_override=p_in_active, T_in_override=t_in_active, p_ex_override=p_ex_active
        )
        
        _, sw_sum = sweep_sim.simulate(fire=True)
        
        local_fmep_bar = 0.35 + 0.12 * (rpm_local / 1000.0) + 0.015 * ((rpm_local / 1000.0) ** 2)
        local_fmep_pa = local_fmep_bar * 1e5
        
        sw_W_ind = sw_sum['indicated_work_J']
        sw_W_fric = local_fmep_pa * sweep_sim.V_d
        sw_W_brake = sw_W_ind - sw_W_fric
        
        sw_brake_power_kW = (sw_W_brake * (rpm_local / 60.0) / 1000.0) * num_rotors
        if sw_brake_power_kW < 0:
            sw_brake_power_kW = 0.0
            
        local_omega = rpm_local * (2 * np.pi / 60.0)
        sw_brake_torque_Nm = (sw_brake_power_kW * 1000.0 / local_omega) if local_omega > 0 else 0.0
        
        sw_air_kg_hr = sw_sum['m_fresh_trapped_kg'] * rpm_local * 60.0 * num_rotors
        sw_dot_m_air = sw_air_kg_hr
        sw_dot_m_fuel = sw_dot_m_air / afr_val
        
        sw_bsfc = (sw_dot_m_fuel * 1000.0) / sw_brake_power_kW if sw_brake_power_kW > 0.05 else float('nan')
        load_ratio_loc = p_in_active / p_amb_local_pa
        factor_egt = 0.52 + 0.222 * (load_ratio_loc ** 0.5) * ((rpm_local / 6500.0) ** 0.12)
        sw_egt_c = (sw_sum['T_end_expansion_K'] * factor_egt) - 273.15
        
        return {
            'Brake Power (kW)': sw_brake_power_kW,
            'Brake Torque (N.m)': sw_brake_torque_Nm,
            'BSFC (g/kWh)': sw_bsfc,
            'Air Flow (kg/h)': sw_dot_m_air,
            'EGT (°C)': sw_egt_c,
            'MAP (bar)': p_in_active / 1e5
        }

    if btn_run_sweep:
        with st.spinner("در حال محاسبه شبیه‌سازی پارامتریک در بازه درخواستی شما..."):
            if sweep_type == "1D Throttle Sweep (تحلیل یک بعدی دریچه گاز)":
                throttles = np.linspace(sw_throttle_min, sw_throttle_max, int(sw_throttle_pts))
                p_list, t_list, bsfc_list, egt_list = [], [], [], []
                p_bar_amb = p_amb_pa / 1e5
                
                for t_val in throttles:
                    res_s = run_sweep_point_v14(p_bar_amb, t_amb_k, t_val)
                    p_list.append(res_s['Brake Power (kW)'])
                    t_list.append(res_s['Brake Torque (N.m)'])
                    bsfc_list.append(res_s['BSFC (g/kWh)'])
                    egt_list.append(res_s['EGT (°C)'])
                    
                fig_sw1, axs_sw1 = plt.subplots(2, 2, figsize=(10, 6))
                axs_sw1[0, 0].plot(throttles, p_list, 'r-o', linewidth=2)
                axs_sw1[0, 0].set_title("Brake Power vs. Throttle")
                axs_sw1[0, 0].set_xlabel("Throttle Opening (%)")
                axs_sw1[0, 0].set_ylabel("Power (kW)")
                axs_sw1[0, 0].grid(True, alpha=0.5)
                
                axs_sw1[0, 1].plot(throttles, t_list, 'b-s', linewidth=2)
                axs_sw1[0, 1].set_title("Brake Torque vs. Throttle")
                axs_sw1[0, 1].set_xlabel("Throttle Opening (%)")
                axs_sw1[0, 1].set_ylabel("Torque (N.m)")
                axs_sw1[0, 1].grid(True, alpha=0.5)
                
                axs_sw1[1, 0].plot(throttles, bsfc_list, 'g-^', linewidth=2)
                axs_sw1[1, 0].set_title("BSFC vs. Throttle")
                axs_sw1[1, 0].set_xlabel("Throttle Opening (%)")
                axs_sw1[1, 0].set_ylabel("BSFC (g/kWh)")
                axs_sw1[1, 0].grid(True, alpha=0.5)
                
                axs_sw1[1, 1].plot(throttles, egt_list, 'm-v', linewidth=2)
                axs_sw1[1, 1].set_title("EGT vs. Throttle")
                axs_sw1[1, 1].set_xlabel("Throttle Opening (%)")
                axs_sw1[1, 1].set_ylabel("EGT (°C)")
                axs_sw1[1, 1].grid(True, alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig_sw1)

            elif sweep_type == "1D Altitude Sweep (تحلیل یک بعدی ارتفاع)":
                altitudes = np.linspace(sw_alt_min, sw_alt_max, int(sw_alt_pts))
                p_list, t_list, bsfc_list, egt_list = [], [], [], []
                
                for alt_ft in altitudes:
                    p_amb_local_bar, t_amb_local, p_amb_local_pa = get_isa_properties(alt_ft)
                    res_s = run_sweep_point_v14(p_amb_local_bar, t_amb_local, throttle_pct)
                    p_list.append(res_s['Brake Power (kW)'])
                    t_list.append(res_s['Brake Torque (N.m)'])
                    bsfc_list.append(res_s['BSFC (g/kWh)'])
                    egt_list.append(res_s['EGT (°C)'])
                    
                fig_sw2, axs_sw2 = plt.subplots(2, 2, figsize=(10, 6))
                axs_sw2[0, 0].plot(altitudes, p_list, 'r-o', linewidth=2)
                axs_sw2[0, 0].set_title("Brake Power vs. Altitude")
                axs_sw2[0, 0].set_xlabel("Altitude (ft)")
                axs_sw2[0, 0].set_ylabel("Power (kW)")
                axs_sw2[0, 0].grid(True, alpha=0.5)
                
                axs_sw2[0, 1].plot(altitudes, t_list, 'b-s', linewidth=2)
                axs_sw2[0, 1].set_title("Brake Torque vs. Altitude")
                axs_sw2[0, 1].set_xlabel("Altitude (ft)")
                axs_sw2[0, 1].set_ylabel("Torque (N.m)")
                axs_sw2[0, 1].grid(True, alpha=0.5)
                
                axs_sw2[1, 0].plot(altitudes, bsfc_list, 'g-^', linewidth=2)
                axs_sw2[1, 0].set_title("BSFC vs. Altitude")
                axs_sw2[1, 0].set_xlabel("Altitude (ft)")
                axs_sw2[1, 0].set_ylabel("BSFC (g/kWh)")
                axs_sw2[1, 0].grid(True, alpha=0.5)
                
                axs_sw2[1, 1].plot(altitudes, egt_list, 'm-v', linewidth=2)
                axs_sw2[1, 1].set_title("EGT vs. Altitude")
                axs_sw2[1, 1].set_xlabel("Altitude (ft)")
                axs_sw2[1, 1].set_ylabel("EGT (°C)")
                axs_sw2[1, 1].grid(True, alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig_sw2)

            elif sweep_type == "1D Engine Speed Sweep (تحلیل یک بعدی دور موتور - RPM)":
                rpms = np.arange(sw_rpm_min, sw_rpm_max + 10, sw_rpm_step)
                p_list, t_list, bsfc_list, egt_list = [], [], [], []
                p_bar_amb = p_amb_pa / 1e5
                
                for rpm_val in rpms:
                    res_s = run_sweep_point_v14(p_bar_amb, t_amb_k, throttle_pct, rpm_local=rpm_val)
                    p_list.append(res_s['Brake Power (kW)'])
                    t_list.append(res_s['Brake Torque (N.m)'])
                    bsfc_list.append(res_s['BSFC (g/kWh)'])
                    egt_list.append(res_s['EGT (°C)'])
                    
                fig_sw_rpm, axs_rpm = plt.subplots(2, 2, figsize=(10, 6))
                axs_rpm[0, 0].plot(rpms, p_list, 'r-o', linewidth=2)
                axs_rpm[0, 0].set_title("Brake Power vs. Engine Speed (RPM)")
                axs_rpm[0, 0].set_xlabel("Engine Speed (RPM)")
                axs_rpm[0, 0].set_ylabel("Power (kW)")
                axs_rpm[0, 0].grid(True, alpha=0.5)
                
                axs_rpm[0, 1].plot(rpms, t_list, 'b-s', linewidth=2)
                axs_rpm[0, 1].set_title("Brake Torque vs. Engine Speed (RPM)")
                axs_rpm[0, 1].set_xlabel("Engine Speed (RPM)")
                axs_rpm[0, 1].set_ylabel("Torque (N.m)")
                axs_rpm[0, 1].grid(True, alpha=0.5)
                
                axs_rpm[1, 0].plot(rpms, bsfc_list, 'g-^', linewidth=2)
                axs_rpm[1, 0].set_title("BSFC vs. Engine Speed (RPM)")
                axs_rpm[1, 0].set_xlabel("Engine Speed (RPM)")
                axs_rpm[1, 0].set_ylabel("BSFC (g/kWh)")
                axs_rpm[1, 0].grid(True, alpha=0.5)
                
                axs_rpm[1, 1].plot(rpms, egt_list, 'm-v', linewidth=2)
                axs_rpm[1, 1].set_title("EGT vs. Engine Speed (RPM)")
                axs_rpm[1, 1].set_xlabel("Engine Speed (RPM)")
                axs_rpm[1, 1].set_ylabel("EGT (°C)")
                axs_rpm[1, 1].grid(True, alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig_sw_rpm)

            elif sweep_type == "2D Combined Sweep (تحلیل دو بعدی ارتفاع و دریچه گاز)":
                throttles = np.linspace(sw_th_min_2d, sw_th_max_2d, 5)
                altitudes = np.linspace(sw_alt_min_2d, sw_alt_max_2d, 5)
                power_mat = np.zeros((len(throttles), len(altitudes)))
                
                for i, t_val in enumerate(throttles):
                    for j, alt_ft in enumerate(altitudes):
                        p_amb_local_bar, t_amb_local, p_amb_local_pa = get_isa_properties(alt_ft)
                        res_s = run_sweep_point_v14(p_amb_local_bar, t_amb_local, t_val)
                        power_mat[i, j] = res_s['Brake Power (kW)']
                        
                fig_sw3, ax_sw3 = plt.subplots(figsize=(8, 5))
                for i in range(len(throttles)):
                    ax_sw3.plot(altitudes, power_mat[i, :], '-o', label=f"Throttle {throttles[i]:.0f}%", linewidth=2)
                    
                ax_sw3.set_title("Brake Power vs. Altitude at Various Throttles", fontweight="bold")
                ax_sw3.set_xlabel("Altitude (ft)")
                ax_sw3.set_ylabel("Power (kW)")
                ax_sw3.grid(True, alpha=0.5)
                ax_sw3.legend()
                st.pyplot(fig_sw3)

            else: # RPM-Throttle map
                rpms = np.linspace(sw_rpm_min_map, sw_rpm_max_map, 5)
                throttles = np.linspace(sw_th_min_map, sw_th_max_map, 5)
                power_mat = np.zeros((len(throttles), len(rpms)))
                
                for i, t_val in enumerate(throttles):
                    for j, rpm_val in enumerate(rpms):
                        res_s = run_sweep_point_v14(p_ambient_bar, t_amb_k, t_val, rpm_local=rpm_val)
                        power_mat[i, j] = res_s['Brake Power (kW)']
                        
                fig_sw4, ax_sw4 = plt.subplots(figsize=(8, 5))
                for i in range(len(throttles)):
                    ax_sw4.plot(rpms, power_mat[i, :], '-s', label=f"Throttle {throttles[i]:.0f}%", linewidth=2)
                    
                ax_sw4.set_title("Brake Power Engine Map vs. Speed (RPM)", fontweight="bold")
                ax_sw4.set_xlabel("Engine Speed (RPM)")
                ax_sw4.set_ylabel("Power (kW)")
                ax_sw4.grid(True, alpha=0.5)
                ax_sw4.legend()
                st.pyplot(fig_sw4)
    else:
        st.info("💡 برای اجرای تحلیل پارامتریک، بازه متغیرهای مد نظر خود را فوقاً تعیین نموده و دکمه ▶️ را فشار دهید.")


with tab_data:
    st.subheader("📂 Complete Raw Simulation Results")
    st.dataframe(results, use_container_width=True)
    
    csv_results = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Raw Simulation Data (CSV)",
        data=csv_results,
        file_name="wankel_engine_simulation_raw_results.csv",
        mime="text/csv"
    )


with tab_theory:
    st.subheader("📝 Theoretical Background and Custom Coupling Models")
    st.markdown("""
### ۱. سینماتیک و تغییرات حجم لحظه‌ای (Epitrochoidal Volume)
تغییرات حجم محفظه ونکل به صورت دقیق از رابطه سینماتیکی کسینوسی کالیبره شده با نقطه مرگ بالا (TDC) در زاویه صفر درجه مدل‌سازی شده است:
$$V(\\theta) = V_c + \\frac{V_d}{2} \\left[ 1 - \\cos\\left(\\frac{2}{3}\\theta\\right) \\right] + V_{pocket}$$

### ۲. اتصال همزمان چندمحفظه‌ای و نشت اپکس‌سیل‌ها (Multi-Chamber Apex Leakage)
بقای جرم محفظه‌ها به صورت زنجیره‌ای حل شده و جریان نشت Blow-by از روی اپکس‌سیل‌های پیشرو و پسرو به صورت شبه‌یک‌بعدی ایزنتروپیک اریفییس کواپل شده است:
$$\\frac{dm_i}{dt} = \\dot{m}_{in, i} - \\dot{m}_{ex, i} - (\\dot{m}_{lead} + \\dot{m}_{lag})_{leak, i} - \\dot{m}_{crev, i}$$

### ۳. مدل انتقال حرارت اصلاح‌شده وشنی (Woschni Coupled Heat Transfer)
سرعت جریان گاز در داخل سیلندر به صورت پویا با رهایش حرارتی و راندمان کار موتورینگ جفت شده است:
$$h_c = 3.26 B^{-0.2} p^{0.8} T^{-0.53} w^{0.8}$$
$$w = C_1 V_m + C_2 \\frac{V_s T_{ipc}}{p_{ipc} V_{ipc}} (p - p_{motoring})$$

### ۴. فرار پویای جرم به درزها (Dynamic Crevice Flow)
جریان رفت و برگشتی جرم به داخل ظرفیت‌های خازنی مرده (Crevices) به صورت برخط با نرخ فشار حل می‌شود:
$$\\dot{m}_{crevice} = \\frac{V_{crev}}{R_g T_{wall}} \\frac{dp}{dt}$$
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("v14 Developed with Garrett GTX series turbocharger integration.")
