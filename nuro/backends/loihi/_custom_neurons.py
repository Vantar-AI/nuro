"""Custom Lava Process models for Izhikevich and AdEx neurons.

These run in **simulation only** (PyLoihiProcessModel with ``float`` tag).
They cannot be deployed to Loihi hardware — use LIF/IF for hardware targets.
"""

from __future__ import annotations


def _import_lava():
    """Lazy-import Lava primitives."""
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.process.ports.ports import InPort, OutPort
    from lava.magma.core.process.variable import Var
    from lava.magma.core.model.py.model import PyLoihiProcessModel
    from lava.magma.core.model.py.ports import PyInPort, PyOutPort
    from lava.magma.core.model.py.type import LavaPyType
    from lava.magma.core.decorator import implements, requires, tag
    from lava.magma.core.resources import CPU
    from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

    return {
        "AbstractProcess": AbstractProcess,
        "InPort": InPort,
        "OutPort": OutPort,
        "Var": Var,
        "PyLoihiProcessModel": PyLoihiProcessModel,
        "PyInPort": PyInPort,
        "PyOutPort": PyOutPort,
        "LavaPyType": LavaPyType,
        "implements": implements,
        "requires": requires,
        "tag": tag,
        "CPU": CPU,
        "LoihiProtocol": LoihiProtocol,
    }


def build_izhikevich_process(size: int, params: dict, dt: float):
    """Create an Izhikevich Process + ProcessModel pair for Lava simulation.

    Returns a Process instance with ``a_in`` (InPort) and ``s_out`` (OutPort).
    """
    import numpy as np

    lava = _import_lava()

    a = params.get("a", 0.02)
    b = params.get("b", 0.2)
    c = params.get("c", -65.0)
    d = params.get("d", 8.0)
    v_thresh = params.get("v_thresh", 30.0)
    dt_ms = dt * 1000.0

    class IzhikevichProcess(lava["AbstractProcess"]):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            shape = (size,)
            self.a_in = lava["InPort"](shape=shape)
            self.s_out = lava["OutPort"](shape=shape)
            self.v = lava["Var"](shape=shape, init=c)
            self.u = lava["Var"](shape=shape, init=b * c)
            self.a_param = lava["Var"](shape=(1,), init=a)
            self.b_param = lava["Var"](shape=(1,), init=b)
            self.c_param = lava["Var"](shape=(1,), init=c)
            self.d_param = lava["Var"](shape=(1,), init=d)
            self.vth = lava["Var"](shape=(1,), init=v_thresh)

    @lava["implements"](proc=IzhikevichProcess, protocol=lava["LoihiProtocol"])
    @lava["requires"](lava["CPU"])
    @lava["tag"]("floating_pt")
    class PyIzhikevichModel(lava["PyLoihiProcessModel"]):
        a_in: lava["PyInPort"] = lava["LavaPyType"](lava["PyInPort"].VEC_DENSE, float)
        s_out: lava["PyOutPort"] = lava["LavaPyType"](lava["PyOutPort"].VEC_DENSE, float)
        v: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        u: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        a_param: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        b_param: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        c_param: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        d_param: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        vth: np.ndarray = lava["LavaPyType"](np.ndarray, float)

        def run_spk(self):
            a_in_data = self.a_in.recv()
            I = a_in_data * 40.0

            steps = max(1, int(dt_ms / 0.5))
            for _ in range(steps):
                h = dt_ms / steps
                dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I) * h
                du = (self.a_param * (self.b_param * self.v - self.u)) * h
                self.v = self.v + dv
                self.u = self.u + du

            fired = (self.v >= self.vth).astype(float)
            self.v = np.where(fired > 0.5, self.c_param, self.v)
            self.u = np.where(fired > 0.5, self.u + self.d_param, self.u)
            self.s_out.send(fired)

    return IzhikevichProcess()


def build_adex_process(size: int, params: dict, dt: float):
    """Create an AdEx Process + ProcessModel pair for Lava simulation.

    Returns a Process instance with ``a_in`` (InPort) and ``s_out`` (OutPort).
    """
    import numpy as np

    lava = _import_lava()

    C = params.get("C", 281.0)
    g_L = params.get("g_L", 30.0)
    E_L = params.get("E_L", -70.6)
    V_T = params.get("V_T", -50.4)
    delta_T = params.get("delta_T", 2.0)
    a_adapt = params.get("a", 4.0)
    b_adapt = params.get("b", 80.5)
    tau_w = params.get("tau_w", 144.0)
    V_reset = params.get("V_reset", -70.6)
    V_cutoff = params.get("V_cutoff", -40.4)
    dt_ms = dt * 1000.0

    class AdExProcess(lava["AbstractProcess"]):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            shape = (size,)
            self.a_in = lava["InPort"](shape=shape)
            self.s_out = lava["OutPort"](shape=shape)
            self.v = lava["Var"](shape=shape, init=E_L)
            self.w = lava["Var"](shape=shape, init=0.0)

    @lava["implements"](proc=AdExProcess, protocol=lava["LoihiProtocol"])
    @lava["requires"](lava["CPU"])
    @lava["tag"]("floating_pt")
    class PyAdExModel(lava["PyLoihiProcessModel"]):
        a_in: lava["PyInPort"] = lava["LavaPyType"](lava["PyInPort"].VEC_DENSE, float)
        s_out: lava["PyOutPort"] = lava["LavaPyType"](lava["PyOutPort"].VEC_DENSE, float)
        v: np.ndarray = lava["LavaPyType"](np.ndarray, float)
        w: np.ndarray = lava["LavaPyType"](np.ndarray, float)

        def run_spk(self):
            a_in_data = self.a_in.recv()
            I = a_in_data * 500.0
            h = dt_ms

            exp_arg = (self.v - V_T) / delta_T
            exp_arg = np.clip(exp_arg, a_min=None, a_max=20.0)
            exp_term = g_L * delta_T * np.exp(exp_arg)

            dV = (-g_L * (self.v - E_L) + exp_term - self.w + I) / C * h
            self.v = self.v + dV

            dw = (a_adapt * (self.v - E_L) - self.w) / tau_w * h
            self.w = self.w + dw

            fired = (self.v >= V_cutoff).astype(float)
            self.v = np.where(fired > 0.5, V_reset, self.v)
            self.w = np.where(fired > 0.5, self.w + b_adapt, self.w)
            self.s_out.send(fired)

    return AdExProcess()
