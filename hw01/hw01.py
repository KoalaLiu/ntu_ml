#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @desc hw01
    @author liuxiji
    @time 2020/6/12 8:27 下午
"""

import numpy as np
import pandas as pd
from lr import LinearRegression


def main():
    """ main """
    data = pd.read_csv('./train.csv', encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()

    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        month_data[month] = sample

    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                x[month * 471 + day * 24 + hour, :] = \
                    month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("y: ", y)

    lr = LinearRegression()
    print(lr.name)

    lr.fit(x, y)
    y_p = lr.predict(x)
    print(y_p)


if __name__ == '__main__':
    main()
