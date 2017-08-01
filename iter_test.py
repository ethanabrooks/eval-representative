#! /usr/bin/env python

import itertools

x = range(20)
for i in x:
    y, z = itertools.tee(x)
    for k, j in enumerate(z):
        print(j)
        if k > 3:
            break


