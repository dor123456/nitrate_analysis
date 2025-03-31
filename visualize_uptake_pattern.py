# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 15:14:07 2025

@author: Owner
"""
import numpy as np
import matplotlib.pyplot as plt

def estimate_nitrate_uptake_parabolic(nitrate_amount, total_days, a=-1, b=0, c=1, shift=0):
    """
    Spreads uptake **up and down like a customizable parabola**.
    - a, b, c: Coefficients for the quadratic function ax^2 + bx + c.
    - shift: Shifts the peak of the parabola left or right.
    """
    days = np.linspace(-1, 1, total_days) # the parabule is centered around 0 for comfort
    uptake_pattern = a * (days - shift)**2 + b * (days - shift) + c  # Customizable parabolic shape
    uptake_pattern = np.maximum(uptake_pattern, 0)  # Ensure non-negative values
    daily_uptake = uptake_pattern / np.sum(uptake_pattern) * nitrate_amount  # Normalize to sum to nitrate_amount
    return daily_uptake


nitrate_amount = 100
total_days = 30
x = np.arange(total_days)
y = estimate_nitrate_uptake_parabolic(nitrate_amount, total_days, a=-1, b=0, c=0.7)

plt.plot(x,y, '-o')
print(np.sum(y))
