"""Recovering a brake-disc surface heat flux from one buried thermocouple.

Deliberately import-light: nothing here pulls in torch, so `import thermalpinn` stays
cheap for the NumPy-only half of the repo (solver, classical baselines, metrics).
"""

__version__ = "0.1.0"
