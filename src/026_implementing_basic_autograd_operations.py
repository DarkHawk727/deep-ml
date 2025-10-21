from __future__ import annotations

from typing import Callable, Literal


class Value:
    def __init__(self, data, _children=(), _op: Literal["+", "*", "ReLU", ""] = ""):
        self.data: float = data
        self.grad: float = 0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op: Literal["+", "*", "ReLU", ""] = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1 if out.data > 0 else 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
