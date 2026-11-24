from collections import deque

import mlx.core as mx


class LinearForecast:
    """Extrapolate from the two latest full evaluations in sigma space."""

    def __init__(self) -> None:
        self.anchors = deque(maxlen=2)

    def update(self, sigma: float, value: mx.array) -> None:
        self.anchors.append((sigma, value))

    def predict(self, sigma: float, dtype: mx.Dtype) -> mx.array:
        last, value = self.anchors[-1]
        if len(self.anchors) == 1:
            return value.astype(dtype)
        first, previous = self.anchors[0]
        distance = (sigma - last) / (last - first)
        return (value.astype(mx.float32) + distance * (value.astype(mx.float32) - previous.astype(mx.float32))).astype(dtype)

    def arrays(self) -> list[mx.array]:
        return [value for _, value in self.anchors]
