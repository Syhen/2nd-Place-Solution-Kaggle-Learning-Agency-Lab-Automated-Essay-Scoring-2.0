import numpy as np
import pandas as pd


def round_target(target: pd.Series, digits=0.5):
    if isinstance(digits, float):
        return np.round(target / digits, 0) * digits
    return np.round(target, decimals=digits)


if __name__ == '__main__':
    x = pd.Series([2.66, 2.46, 2.1, 1.9, 2.9, 1.55, 1.45])
    print(x.values)
    print(round_target(x, 0.5))
    print(round_target(x, 0))
