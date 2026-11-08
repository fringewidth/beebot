"""A module design and the physical capabilities it implies.

A module is a cube of side L. Its volume is shared between a fixed shell with electronics, a fixed
battery for the actuators, face connectors, an actuator and, optionally, a flight system (rotors
inside the cube's footprint, their motors and a flight battery). Every capability is derived from
those shares with first-order physics and constants from real parts (HARDWARE.md lists the
sources), then expressed in the module's own units: forces in module weights (m g), torques in
m g L. Those dimensionless numbers are all the simulator sees.

    from beebot.hardware.design import Design
    Design(side=0.03, connector="magnet", connector_share=0.05, actuator_share=0.1).capability()
"""

import math
from dataclasses import dataclass

G = 9.81
AIR = 1.2  # kg/m^3

SHELL_SHARE, SHELL_DENSITY = 0.15, 1200.0  # plastic shell and boards
BATTERY_SHARE, BATTERY_DENSITY = 0.10, 1800.0  # LiPo for actuators and compute

MAGNET_DENSITY = 7500.0  # NdFeB
MAGNET_PRESSURE = 9.0e4  # Pa of face area, magnet pair at thickness / width 0.25, derated 0.6 for gap and misalignment
MAGNET_ASPECT = 0.25  # thickness / width of each face magnet
EPM_PRESSURE_FACTOR = 0.5  # only the switchable half (AlNiCo) of an electropermanent magnet can be turned off
EPM_DENSITY = 8000.0  # magnets plus the switching coil

ACTUATOR_TORQUE = 8.0e4  # N m per m^3: geared micro motor, gearbox-limited (about 0.25 N m from 3.1 cm^3)
ACTUATOR_DENSITY = 3000.0

FLIGHT_DENSITY = 2000.0  # flight motors and flight battery
FLIGHT_POWER = 1000.0  # W per kg of flight motors and battery (electrical)
FLIGHT_EFFICIENCY = 0.3  # motor efficiency times rotor figure of merit
ROTOR_AREA = 0.6  # rotor disc area per L^2 (inside the footprint, so modules still pack)
ROTOR_SHARE, ROTOR_DENSITY = 0.15, 100.0  # ducted rotor volume, mostly air

OBJECT_DENSITY = 1000.0  # the 2 x 2 x 2 "cheese" the swarm pushes and carries
OBJECT_SIDE = 2


@dataclass(frozen=True)
class Capability:
    """A design in the simulator's units."""

    strength: float  # one face connector's pull-off force, in module weights
    torque: float  # actuator torque, in m g L
    thrust: float  # flight thrust, in module weights; below 1 the module cannot fly
    cheese: float  # the object's weight, in module weights
    switchable: bool  # connectors turn off before a move (electropermanent)

    @property
    def flies(self):
        return self.thrust >= 1.0


@dataclass(frozen=True)
class Design:
    side: float  # L in metres
    connector: str  # "magnet" (permanent) or "epm" (electropermanent)
    connector_share: float  # volume share of all six face magnets
    actuator_share: float
    flight_share: float = 0.0  # flight motors and battery; the rotor volume comes on top when > 0

    def free_share(self):
        rotor = ROTOR_SHARE if self.flight_share > 0 else 0.0
        return 1.0 - SHELL_SHARE - BATTERY_SHARE - self.connector_share - self.actuator_share - self.flight_share - rotor

    def feasible(self):
        return self.free_share() >= 0 and min(self.connector_share, self.actuator_share, self.flight_share) >= 0

    def density(self):
        """Mean density of the module in kg/m^3 (empty volume counts as air, about zero)."""
        magnet = EPM_DENSITY if self.connector == "epm" else MAGNET_DENSITY
        rotor = ROTOR_SHARE * ROTOR_DENSITY if self.flight_share > 0 else 0.0
        return SHELL_SHARE * SHELL_DENSITY + BATTERY_SHARE * BATTERY_DENSITY + self.connector_share * magnet + self.actuator_share * ACTUATOR_DENSITY + self.flight_share * FLIGHT_DENSITY + rotor

    def mass(self):
        return self.density() * self.side**3

    def pull(self):
        """Pull-off force of one face connector in newtons. Each of six faces carries a magnet of
        area a and thickness MAGNET_ASPECT * sqrt(a), so a = (share L^3 / (6 aspect))^(2/3),
        capped at the face."""
        if self.connector_share <= 0:
            return 0.0
        area = min((self.connector_share * self.side**3 / (6 * MAGNET_ASPECT)) ** (2 / 3), self.side**2)
        pressure = MAGNET_PRESSURE * (EPM_PRESSURE_FACTOR if self.connector == "epm" else 1.0)
        return pressure * area

    def thrust(self):
        """Hover-capable thrust in newtons from momentum theory: T = (2 rho A)^(1/3) (eta P)^(2/3)."""
        if self.flight_share <= 0:
            return 0.0
        power = FLIGHT_POWER * FLIGHT_DENSITY * self.flight_share * self.side**3
        area = ROTOR_AREA * self.side**2
        return (2 * AIR * area) ** (1 / 3) * (FLIGHT_EFFICIENCY * power) ** (2 / 3)

    def capability(self):
        weight = self.mass() * G
        return Capability(
            strength=self.pull() / weight,
            torque=ACTUATOR_TORQUE * self.actuator_share * self.side**3 / (weight * self.side),
            thrust=self.thrust() / weight,
            cheese=OBJECT_DENSITY * OBJECT_SIDE**3 * self.side**3 / self.mass(),
            switchable=self.connector == "epm",
        )

    def label(self):
        return f"L{self.side * 100:g}cm-{self.connector}-c{self.connector_share:g}-a{self.actuator_share:g}-f{self.flight_share:g}"


def describe(design):
    cap = design.capability()
    return {
        "design": design.label(),
        "mass_g": round(design.mass() * 1000, 3),
        "density": round(design.density()),
        "strength": round(cap.strength, 2),
        "torque": round(cap.torque, 2),
        "thrust": round(cap.thrust, 2),
        "cheese": round(cap.cheese, 2),
        "cantilever": math.isqrt(int(max(cap.strength, 0))),
    }
