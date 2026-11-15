"""Simulation toolchain for millimetre-scale modular robot cubes.

Modules:
  magnetics   magpylib force/torque for magnets and planar coils, image-method steel plate
  fem         NGSolve axisymmetric magnetostatics (steel pole piece case)
  coil        planar spiral coil geometry, resistance, inductance, thermal model
  circuit     storage element + H-bridge + coil RL pulse simulation (ODE)
  mjcoupling  MuJoCo coupling (pairwise magnetic wrenches, lookup tables, surface forces)
  opt         parallel multi-objective search and Monte Carlo tolerance analysis
  validation  validation rows against published data
"""
