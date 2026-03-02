import math


def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    PRIMITIVES = {
        "square": (lambda u: u**2, lambda u: 2 * u),
        "sin": (lambda u: math.sin(u), lambda u: math.cos(u)),
        "exp": (lambda u: math.exp(u), lambda u: math.exp(u)),
        "log": (lambda u: math.log(u), lambda u: 1.0 / u),
    }

    functions = list(reversed(functions))
    vals = [x]
    for fn in functions:
        f, _ = PRIMITIVES[fn]
        vals.append(f(vals[-1]))

    dy_dx = 1.0
    for i in reversed(range(len(functions))):
        fn = functions[i]
        _, df = PRIMITIVES[fn]
        dy_dx *= df(vals[i])

    return dy_dx


print(f"{compute_chain_rule_gradient(['sin', 'square'], 1.0):.6f}")  # 1.080605
