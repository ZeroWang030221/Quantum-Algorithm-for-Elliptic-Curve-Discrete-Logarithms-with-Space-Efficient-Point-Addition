"""A tiny Qiskit-compatible circuit data model used only by the semantic test.

It is not an arithmetic model and it does not know anything about EEA.  It only
records quantum instructions and their definitions, closely matching the small
subset of Qiskit's circuit API used by this repository.  The semantic test then
executes the emitted gates with an independent sparse-state dynamic-circuit
simulator.

When real Qiskit is installed this module is not injected as ``qiskit``; the
same simulator works directly on real Qiskit circuits.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import sys
import types
from typing import Iterable, Optional, Sequence


class Bit:
    __slots__ = ("register", "index")

    def __init__(self, register, index: int):
        self.register = register
        self.index = int(index)

    def __repr__(self) -> str:
        return f"{self.register.name}[{self.index}]"

    def __hash__(self) -> int:
        return id(self)


class Qubit(Bit):
    pass


class Clbit(Bit):
    pass


class _Register:
    bit_type = Bit

    def __init__(self, size: int, name: str):
        self.size = int(size)
        self.name = str(name)
        self._bits = [self.bit_type(self, i) for i in range(self.size)]

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self._bits)

    def __getitem__(self, item):
        return self._bits[item]

    def __repr__(self):
        return f"{type(self).__name__}({self.size}, {self.name!r})"


class QuantumRegister(_Register):
    bit_type = Qubit


class ClassicalRegister(_Register):
    bit_type = Clbit


class Instruction:
    def __init__(self, name: str, num_qubits: int = 0, num_clbits: int = 0, params=None, label: Optional[str] = None):
        self.name = str(name)
        self.num_qubits = int(num_qubits)
        self.num_clbits = int(num_clbits)
        self.params = list(params or [])
        self.label = label
        self._definition = None
        self.condition = None

    @property
    def definition(self):
        return self._definition

    @definition.setter
    def definition(self, value):
        self._definition = value

    def to_mutable(self):
        out = copy.copy(self)
        out.params = list(self.params)
        return out

    def inverse(self):
        lname = self.name.lower()
        base = lname
        while base.endswith("_dg"):
            base = base[:-3]
        if base in {"x", "cx", "cnot", "ccx", "tof", "toffoli", "mcx", "swap", "h", "z", "cz", "id", "barrier"}:
            out = self.to_mutable()
            out.name = base
            # Keep primitive identity detectable by definition=None and base_name.
            out.base_name = base
            return out
        if self.definition is None:
            raise ValueError(f"Instruction {self.name!r} has no inverse definition")
        out = Instruction(f"{self.name}_dg", self.num_qubits, self.num_clbits, self.params, self.label)
        out.definition = self.definition.inverse()
        return out

    def __repr__(self):
        return f"Instruction({self.name!r}, q={self.num_qubits}, c={self.num_clbits})"


class Gate(Instruction):
    def __init__(self, name: str, num_qubits: int = 0, params=None, label: Optional[str] = None):
        super().__init__(name, num_qubits, 0, params, label)

    def inverse(self):
        lname = self.name.lower()
        base = lname
        while base.endswith("_dg"):
            base = base[:-3]
        if base in {"x", "cx", "cnot", "ccx", "tof", "toffoli", "mcx", "swap", "h", "z", "cz", "id", "barrier"}:
            out = self.to_mutable()
            out.name = base
            out.base_name = base
            return out
        if self.definition is None:
            raise ValueError(f"Gate {self.name!r} has no inverse definition")
        out = Gate(f"{self.name}_dg", self.num_qubits, self.params, self.label)
        out.definition = self.definition.inverse()
        return out


class ZGate(Gate):
    def __init__(self):
        super().__init__("z", 1, [])


class CZGate(Gate):
    def __init__(self):
        super().__init__("cz", 2, [])


@dataclass
class CircuitInstruction:
    operation: Instruction
    qubits: tuple
    clbits: tuple

    def __iter__(self):
        yield self.operation
        yield self.qubits
        yield self.clbits


@dataclass
class _FindBitResult:
    index: int
    registers: list


class QuantumCircuit:
    def __init__(self, *regs, name: str = "circuit"):
        self.name = str(name)
        self.qregs = []
        self.cregs = []
        self.qubits = []
        self.clbits = []
        self.data: list[CircuitInstruction] = []
        self._condition_stack = []
        for reg in regs:
            self.add_register(reg)

    @property
    def num_qubits(self):
        return len(self.qubits)

    @property
    def num_clbits(self):
        return len(self.clbits)

    def add_register(self, reg):
        if isinstance(reg, QuantumRegister):
            self.qregs.append(reg)
            self.qubits.extend(list(reg))
        elif isinstance(reg, ClassicalRegister):
            self.cregs.append(reg)
            self.clbits.extend(list(reg))
        else:
            raise TypeError(f"Unsupported register type: {type(reg)!r}")
        return reg

    def find_bit(self, bit):
        if isinstance(bit, Qubit):
            idx = self.qubits.index(bit)
            regs = [(bit.register, bit.index)]
        elif isinstance(bit, Clbit):
            idx = self.clbits.index(bit)
            regs = [(bit.register, bit.index)]
        else:
            raise TypeError(f"not a bit: {bit!r}")
        return _FindBitResult(idx, regs)

    def _as_qargs(self, qargs):
        if isinstance(qargs, Qubit):
            return [qargs]
        if isinstance(qargs, QuantumRegister):
            return list(qargs)
        return list(qargs)

    def _as_cargs(self, cargs):
        if cargs is None:
            return []
        if isinstance(cargs, Clbit):
            return [cargs]
        if isinstance(cargs, ClassicalRegister):
            return list(cargs)
        return list(cargs)

    def append(self, instruction, qargs, cargs=None):
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        qargs = self._as_qargs(qargs)
        cargs = self._as_cargs(cargs)
        if len(qargs) != instruction.num_qubits:
            raise ValueError(
                f"{instruction.name}: expected {instruction.num_qubits} qargs, got {len(qargs)}"
            )
        if len(cargs) != instruction.num_clbits:
            raise ValueError(
                f"{instruction.name}: expected {instruction.num_clbits} cargs, got {len(cargs)}"
            )
        op = instruction
        if self._condition_stack and getattr(op, "condition", None) is None:
            op = op.to_mutable()
            op.condition = self._condition_stack[-1]
        self.data.append(CircuitInstruction(op, tuple(qargs), tuple(cargs)))
        return self.data[-1]

    def _primitive(self, name, qargs, cargs=()):
        op = Instruction(name, len(qargs), len(cargs), [])
        op.base_name = name
        return self.append(op, qargs, cargs)

    def x(self, q):
        return self._primitive("x", [q])

    def h(self, q):
        return self._primitive("h", [q])

    def z(self, q):
        return self._primitive("z", [q])

    def cx(self, control, target):
        return self._primitive("cx", [control, target])

    cnot = cx

    def cz(self, control, target):
        return self._primitive("cz", [control, target])

    def ccx(self, c0, c1, target):
        return self._primitive("ccx", [c0, c1, target])

    def mcx(self, controls, target):
        controls = list(controls)
        return self._primitive("mcx", controls + [target])

    def swap(self, a, b):
        return self._primitive("swap", [a, b])

    def measure(self, q, c):
        return self._primitive("measure", [q], [c])

    def reset(self, q):
        return self._primitive("reset", [q])

    def barrier(self, *qargs):
        qargs = list(qargs) if qargs else list(self.qubits)
        return self._primitive("barrier", qargs)

    @contextmanager
    def if_test(self, condition):
        self._condition_stack.append(condition)
        try:
            yield
        finally:
            self._condition_stack.pop()

    def to_gate(self, label: Optional[str] = None):
        if self.num_clbits:
            raise ValueError("Circuit with classical bits cannot be converted to Gate")
        out = Gate(label or self.name, self.num_qubits, [])
        out.definition = self
        return out

    def to_instruction(self, label: Optional[str] = None):
        out = Instruction(label or self.name, self.num_qubits, self.num_clbits, [])
        out.definition = self
        return out

    def inverse(self):
        if any(item.operation.name.lower().removesuffix("_dg") in {"measure", "reset"} for item in self.data):
            raise ValueError("Cannot invert a circuit containing measure/reset")
        out = QuantumCircuit(*self.qregs, *self.cregs, name=f"{self.name}_dg")
        for item in reversed(self.data):
            out.append(item.operation.inverse(), item.qubits, item.clbits)
        return out

    def compose(self, other, qubits=None, clbits=None, inplace=False):
        target = self if inplace else self.copy()
        qmap = list(target.qubits if qubits is None else qubits)
        cmap = list(target.clbits if clbits is None else clbits)
        if len(qmap) < other.num_qubits or len(cmap) < other.num_clbits:
            raise ValueError("compose mapping too small")
        for item in other.data:
            qs = [qmap[other.find_bit(q).index] for q in item.qubits]
            cs = [cmap[other.find_bit(c).index] for c in item.clbits]
            op = item.operation
            cond = getattr(op, "condition", None)
            if cond is not None:
                lhs, expected = cond
                op = op.to_mutable()
                if isinstance(lhs, Clbit):
                    lhs = cmap[other.find_bit(lhs).index]
                else:
                    try:
                        lhs = tuple(cmap[other.find_bit(bit).index] for bit in lhs)
                    except TypeError:
                        pass
                op.condition = (lhs, expected)
            target.append(op, qs, cs)
        return target

    def copy(self):
        out = QuantumCircuit(*self.qregs, *self.cregs, name=self.name)
        out.data = list(self.data)
        return out

    def count_ops(self):
        from collections import Counter
        return Counter(item.operation.name for item in self.data)


# Minimal stubs for modules imported but not used by this semantic test.
def transpile(circuit, *args, **kwargs):
    return circuit


def qasm2_dumps(circuit):
    return f"// mini-qiskit placeholder for {circuit.name}\n"


def install_as_qiskit() -> None:
    """Register this module's API under the import names used by the codebase."""
    qiskit_mod = types.ModuleType("qiskit")
    qiskit_mod.QuantumCircuit = QuantumCircuit
    qiskit_mod.QuantumRegister = QuantumRegister
    qiskit_mod.ClassicalRegister = ClassicalRegister
    qiskit_mod.transpile = transpile

    circuit_mod = types.ModuleType("qiskit.circuit")
    for name, obj in {
        "QuantumCircuit": QuantumCircuit,
        "QuantumRegister": QuantumRegister,
        "ClassicalRegister": ClassicalRegister,
        "Instruction": Instruction,
        "Gate": Gate,
        "Qubit": Qubit,
        "Clbit": Clbit,
    }.items():
        setattr(circuit_mod, name, obj)

    library_mod = types.ModuleType("qiskit.circuit.library")
    library_mod.ZGate = ZGate
    library_mod.CZGate = CZGate

    qasm2_mod = types.ModuleType("qiskit.qasm2")
    qasm2_mod.dumps = qasm2_dumps

    transpiler_mod = types.ModuleType("qiskit.transpiler")
    class PassManager:
        def __init__(self, *args, **kwargs): pass
        def run(self, circuit): return circuit
    transpiler_mod.PassManager = PassManager

    passes_mod = types.ModuleType("qiskit.transpiler.passes")
    class TemplateOptimization:
        def __init__(self, *args, **kwargs): pass
    passes_mod.TemplateOptimization = TemplateOptimization

    sys.modules.update({
        "qiskit": qiskit_mod,
        "qiskit.circuit": circuit_mod,
        "qiskit.circuit.library": library_mod,
        "qiskit.qasm2": qasm2_mod,
        "qiskit.transpiler": transpiler_mod,
        "qiskit.transpiler.passes": passes_mod,
    })
