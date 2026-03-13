import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    Wx_np = np.array(Wx)
    Wh_np = np.array(Wh)
    b_np = np.array(b)
    h_prev = np.array(initial_hidden_state)

    for input_seq in input_sequence:
        x_t = np.array(input_seq)
        h = np.tanh(Wx_np @ x_t + Wh_np @ h_prev + b_np)
        h_prev = h

    return np.round(h, 4).tolist()


print(rnn_forward([[1.0], [2.0], [3.0]], [0.0], [[0.5]], [[0.8]], [0.0]))  # [0.9759]
