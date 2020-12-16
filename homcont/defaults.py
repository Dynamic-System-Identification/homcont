"""Default tracing parameters.


HomCont:
Solving systems of nonlinear equations by homotopy continuation.

Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it
and/or modify it under the terms of the MIT License.
"""


PARAMETERS = {
    'x_tol': 1e-7,
    't_tol': 1e-7,
    'H_tol': 1e-7,
    'ds0': 0.01,
    'ds_infl': 1.2,
    'ds_defl': 0.5,
    'ds_min': 1e-9,
    'ds_max': 1000,
    'corr_steps_max': 20,
    'corr_dist_max': 0.3,
    'corr_ratio_max': 0.3,
    'detJ_change_max': 0.3,
    'bifurc_angle_min': 177.5
    }
