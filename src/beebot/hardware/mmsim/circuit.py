"""Pulse power circuit: storage element -> H-bridge -> coil (R, L, optional back EMF), plain ODE.

ngspice was not installable in this environment (no brew or conda; PySpice needs libngspice), so the
circuit is a small ODE solved with scipy. Topology limits: ideal switches with Rds_on, fixed dead time
neglected, freewheel through the low-side switches (body diode drop optional).

State: q (charge drawn from storage, C), i (coil current, A), e_drawn (J drawn from storage terminals).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Storage:
    """Thevenin storage element.

    kind 'cap': ideal capacitance c_f with series esr, optional bias derating (MLCC):
        c_eff = c_f * (1 - derate * (v / v_rated)^2) clipped at 0.2 c_f.
    kind 'cell': solid-state or thin-film cell, open-circuit voltage linear in state of charge from
        v_empty to v_full, capacity_c in coulombs, series resistance esr.
    """

    kind: str
    esr: float
    v0: float = 0.0
    c_f: float = 0.0
    v_rated: float = 1.0
    derate: float = 0.0
    capacity_c: float = 0.0
    v_full: float = 0.0
    v_empty: float = 0.0
    soc0: float = 1.0

    def ocv(self, q_drawn: float) -> float:
        if self.kind == "cap":
            v = self.v0
            # integrate the nonlinear capacitance: dV/dq = -1/c_eff(V); handled in the ODE for caps
            return v - q_drawn / self.c_f
        soc = self.soc0 - q_drawn / self.capacity_c
        soc = min(max(soc, 0.0), 1.0)
        return self.v_empty + (self.v_full - self.v_empty) * soc


def mlcc_bank(n: int, c_each: float, esr_each: float, v0: float, v_rated: float, derate: float = 0.0) -> Storage:
    return Storage("cap", esr_each / n, v0=v0, c_f=n * c_each, v_rated=v_rated, derate=derate)


def supercap(c_f: float, esr: float, v0: float) -> Storage:
    return Storage("cap", esr, v0=v0, c_f=c_f, v_rated=max(v0, 1e-9))


# TDK CeraCharge B73180A0101M062 class (EIA 1812): 100 uAh, 1.5 V typical, 0 to 1.6 V window (TDK datasheet via
# novapart.co copy). 200 ohm internal resistance is from a third-party listing for a sibling ordering code
# (datasheets.com B73180A0101M199) and is unconfirmed.
def ceracharge(soc0: float = 1.0, r_int: float = 200.0) -> Storage:
    return Storage("cell", r_int, capacity_c=100e-6 * 3600, v_full=1.6, v_empty=1.0, soc0=soc0)


@dataclass(frozen=True)
class Driver:
    rds_on: float = 0.1  # per switch, two in series during drive
    v_diode: float = 0.0
    r_wiring: float = 0.0


def simulate_pulse(storage: Storage, r_coil: float, l_coil: float, t_on: float, driver: Driver = Driver(),
                   t_total: float | None = None, back_emf: float = 0.0, reverse: bool = False,
                   n_eval: int = 2000):
    """Drive the coil for t_on seconds, then freewheel until t_total.

    Returns dict with t, i, v_term (storage terminal voltage), v_open (storage open-circuit voltage),
    and scalars: i_peak, droop_peak (open-circuit minus terminal at the instant of peak current relative
    to start), v_end, e_stored_drawn (J out of the storage open-circuit source), e_delivered_terminal
    (J through storage terminals), e_coil_loss, e_magnetic_end, e_switch_loss, and efficiency terms.
    """
    t_total = t_total if t_total is not None else 6 * t_on
    sign = -1.0 if reverse else 1.0
    r_path = r_coil + 2 * driver.rds_on + driver.r_wiring
    r_free = r_coil + 2 * driver.rds_on + driver.r_wiring

    def c_eff(v):
        if storage.derate == 0.0:
            return storage.c_f
        return storage.c_f * max(1 - storage.derate * (v / storage.v_rated) ** 2, 0.2)

    # state: v_oc (cap) or q_drawn (cell) -> unify as x; i; plus accumulators
    def v_open(x):
        return max(x, 0.0) if storage.kind == "cap" else storage.ocv(x)

    def rhs(t, y):
        # j is the coil current in the drive direction (>= 0); i = sign * j is reported
        x, j, e_term, e_coil, e_sw, e_src = y
        j = max(j, 0.0) if j > 1e-12 else 0.0
        if t < t_on:
            vo = v_open(x)
            r_tot = r_path + storage.esr
            dj = (vo - j * r_tot - back_emf) / l_coil
            dx = -j / c_eff(vo) if storage.kind == "cap" else j
            de_term = (vo - storage.esr * j) * j
            de_src = vo * j
            vdl = 0.0
        else:
            dj = (-j * r_free - driver.v_diode) / l_coil if j > 1e-12 else 0.0
            dx = de_term = de_src = 0.0
            vdl = abs(j) * driver.v_diode
        de_coil = j * j * r_coil
        de_sw = j * j * (2 * driver.rds_on + driver.r_wiring) + vdl
        return [dx, dj, de_term, de_coil, de_sw, de_src]

    x0 = storage.v0 if storage.kind == "cap" else 0.0
    ts = np.linspace(0, t_total, n_eval)
    sol = solve_ivp(rhs, (0, t_total), [x0, 0.0, 0, 0, 0, 0], t_eval=ts, max_step=t_on / 200,
                    method="LSODA", rtol=1e-8, atol=1e-12)
    t, x, i = sol.t, sol.y[0], sign * sol.y[1]
    v_o = np.array([v_open(xx) for xx in x])
    v_term = v_o - storage.esr * np.abs(i) * (t < t_on)
    k = int(np.argmax(np.abs(i)))
    start = v_open(x0)
    return {
        "t": t,
        "i": i,
        "v_term": v_term,
        "v_open": v_o,
        "i_peak": float(np.abs(i).max()),
        "droop_at_peak": float(start - v_term[k]),
        "v_end": float(v_o[-1]),
        "droop_open_end": float(start - v_o[-1]),
        "e_source": float(sol.y[5][-1]),
        "e_terminal": float(sol.y[2][-1]),
        "e_coil_loss": float(sol.y[3][-1]),
        "e_switch_loss": float(sol.y[4][-1]),
        "e_magnetic_peak": float(0.5 * l_coil * np.abs(i).max() ** 2),
        "q_drawn": float(abs(x0 - x[-1]) * storage.c_f) if storage.kind == "cap" else float(x[-1]),
    }


def energy_per_pulse(res) -> float:
    """Energy leaving the storage terminals per pulse, joules."""
    return res["e_terminal"]


def pulse_train(storage: Storage, r_coil: float, l_coil: float, t_on: float, period: float, n: int,
                driver: Driver = Driver(), recharge_current: float = 0.0):
    """Repeated pulses, recharging the storage at a constant current (A) between pulses.

    Returns per-pulse peak current, start voltage and energy. Caps only (recharge raises v_open).
    """
    out = []
    st = storage
    for k in range(n):
        res = simulate_pulse(st, r_coil, l_coil, t_on, driver, t_total=t_on * 3)
        out.append({"pulse": k, "v_start": st.v0, "i_peak": res["i_peak"], "e_terminal": res["e_terminal"]})
        if st.kind == "cap":
            v_next = res["v_end"] + recharge_current * (period - 3 * t_on) / st.c_f
            v_next = min(v_next, storage.v0)
            st = Storage(**{**st.__dict__, "v0": v_next})
    return out
