"""Some helper functions for path tracing.


HomCont:
Solving systems of nonlinear equations by homotopy continuation.

Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it
and/or modify it under the terms of the MIT License.
"""


import numpy as np
import sys


# %% check input to homsolve


def input_check(y0, H, J, x_transformer):
    """Check user-provided starting point and homotopy functions."""

    # check y0
    if len(y0.shape) != 1:
        raise ValueError(f'"y0" must be a flat 1D array, \
                         but has shape {y0.shape}.')

    # check H(y0)
    try:
        H0 = H(y0)
    except:
        raise ValueError('"H(y0)" cannot be evaluated.')
    if np.isnan(H0).any():
        raise ValueError('"H0(y0)" produces NaN.')
    if len(H0.shape) != 1:
        raise ValueError(f'"H(y0)" should be a 1D vector, \
                         but has shape {H0.shape}.')
    if len(H0) != len(y0)-1:
        raise ValueError(f'"H(y0)" should have length {len(y0)-1}, \
                         but has length {len(H0)}.')

    # check J(y0)
    try:
        J0 = J(y0)
    except:
        raise ValueError('"J(y0)" cannot be evaluated.')
    if np.isnan(J0).any():
        raise ValueError('"J0(y0)" produces NaN.')
    if len(J0.shape) != 2:
        raise ValueError(f'"J(y0)" should be a 2D matrix, \
                         but has shape {J0.shape}.')
    if J0.shape != (len(y0)-1, len(y0)):
        raise ValueError(f'"J(y0)" should have shape {(len(y0)-1, len(y0))}, \
                         but has shape {J0.shape}.')

    # check x_transformer(y0[:-1])
    try:
        sigma0 = x_transformer(y0[:-1])
    except:
        raise ValueError('"x_transformer(y0[:-1])" cannot be evaluated.')
    if np.isnan(sigma0).any():
        raise ValueError('"x_transformer(y0[:-1])" produces NaN.')
    if len(sigma0.shape) != 1:
        raise ValueError(f'"x_transformer(y0[:-1])" should be a 1D vector, \
                         but has shape {sigma0.shape}.')

    return


# %% tangent and pseudo inverse of Jacobian


def qr(J_y):
    """QR decomposition of evaluated Jacobian matrix."""
    Q, R = np.linalg.qr(J_y.transpose(), mode='complete')
    return Q, R


# def tangent(J_y, sign):
#     """Tangent of implicit curve via evaluated Jacobian.

#     Normalized to unit length.
#     """
#     dim = J_y.shape[1]
#     tangent = np.zeros(dim, dtype=np.float64)
#     for k in range(dim):
#         J_y_without_k = np.delete(J_y, k, axis=1)
#         tangent[k] = np.linalg.det(J_y_without_k) * (-1)**(k+1)
#     return -sign * tangent / np.linalg.norm(tangent)


def tangent_qr(Q, R, sign):
    """Tangent of implicit curve via QR decomposition of Jacobian.

    Normalized to unit length.
    """
    return sign * Q[:, -1] * np.sign(np.prod(np.diag(R)))


def jac_pinv(Q, R):
    """Pseudo inverse of Jacobian based on its QR decomposition."""
    return np.dot(Q, np.vstack((np.linalg.inv(
        np.delete(R, -1, axis=0).transpose()), np.zeros(R.shape[1]))))


# %% orientation of path


def greedy_sign(y, Q, R, t_target):
    """Greedy choice of sign in direction of target."""
    t_init = y[-1]
    sign = 1
    dtds = tangent_qr(Q=Q, R=R, sign=sign)[-1]
    if (dtds < 0 and t_target > t_init) or (dtds > 0 and t_target < t_init):
        sign *= -1
    return sign


# %% check transversality of tangent


def transversality_check(tangent, parameters):
    """Check transversality of given tangent.

    Based on angle between tangent and straight line in t-direction.
    """
    dummy_tangent = np.zeros(shape=len(tangent), dtype=np.float64)
    dummy_tangent[-1] = 1

    scalar_product = min([1, max([-1, np.dot(tangent, dummy_tangent)])])
    angle = np.arccos(scalar_product) * 180 / np.pi
    
    if angle > parameters['transvers_angle_max']:
        raise ValueError(f'\nTangent has angle {angle: 0.2f}° relative to straight line in t-direction. Starting point is not transversal to x-plane.')

    return


# %% bifurcation detection


def swap_sign_bifurcation(sign, parameters, tangent_new, tangent_old,
                          verbose=False):
    """Check for bifurcation.

    Decide whether or not orientation of path should be changed.

    Bifurcation detection based on angle between consecutive tangents.
    Works reasonably well.
    parameters['bifurc_angle_min'] is crucial:
        If too close to 180°, actual bifurcations may be undetected.
        If too far away from 180°, bifurcations may be falsely detected.


    Note:
    Bifurcation detection based on a change in sign of the determinant
    of the augmented Jacobian, as suggested by Allgower/Georg (1990, p. 79)
    only works well for simple bifurcations, but does not work well for
    general higher-dimensional bifurcations.

    """
    scalar_product = min([1, max([-1, np.dot(tangent_new, tangent_old)])])
    angle = np.arccos(scalar_product) * 180 / np.pi

    if angle > parameters['bifurc_angle_min']:
        if verbose:
            sys.stdout.write(f'\nBifurcation point encountered at angle {angle: 0.2f}°. Direction swapped.\n')
            sys.stdout.flush()
        return True
    return False


# %% step size adjustment


def deflate(ds, parameters):
    """Deflate step size by factor."""
    return ds * parameters['ds_defl']


def inflate(ds, parameters, t, t_old, t_target, corr_steps, corr_dist=0):
    """Inflate step size by factor.

    Keep step size if at least 10 corrector steps were needed.
    Inflate step size if less than 10 corrector steps were needed.

    Cap inflation at ds_max.
    Cap inflation at |(t_target-t) / t_old| to avoid exceeding
    hom_par_target by much, where |(hom_par_target-hom_par) / t_diff|
    extrapolates change in homotopy parameter at current step size.

    Optional:
    Let step size inflation also depend on
    distance of corrector step relative to step size
    """
    if corr_steps < 10:
        return min([parameters['ds_infl'] * ds,
                    parameters['ds_max'],
                    np.abs((t_target-t) / (t-t_old))])
    else:
        return ds


# %% quasi Newton corrector


def corrector(y_pred, H, J, ds, tangent, parameters):
    """Perform corrector iterations.

    Use quasi Newton method:
    Compute Jacobian inverse once at the beginning,
    not anew at each Newton iteration.

    Return corrector point
    AND a couple of key figures from the corrector process.
    """

    # keep predictor point as fallback option
    J_y_pred = J(y_pred)
    detJ_y_pred = np.linalg.det(np.vstack([J_y_pred, tangent]))

    # compute inverse of Jacobian
    y = y_pred.copy()
    H_y = H(y)
    J_y = J(y)
    detJ_y = np.linalg.det(np.vstack([J_y, tangent]))
    Q, R = qr(J_y)
    J_pinv = jac_pinv(Q=Q, R=R)

    # initialize correction
    corr_dist_old = np.inf
    corr_dist_tot = 0
    corr_step = 0

    while np.max(np.abs(H_y)) > parameters['H_tol']:

        corr_step += 1

        # corrector step
        vec = np.dot(J_pinv, H_y)
        y = y - vec
        corr_dist_step = np.linalg.norm(vec)
        corr_dist_tot += corr_dist_step
        corr_dist = corr_dist_step / max([ds, 1])
        corr_ratio = corr_dist / corr_dist_old
        corr_dist_old = corr_dist

        # if corrector step leads to invalid point,
        # decrease step size and return to predictor step
        H_y = H(y)
        if np.isnan(H_y).any():
            return y, J_y_pred, False, corr_step, corr_dist_tot, 'NaN in H(y)'

        # if corrector step violates restrictions on
            # distance,
            # ratio of consecutive distances or
            # number of steps,
        # decrease step size and return to predictor step
        if corr_dist > parameters['corr_dist_max'] \
                or corr_ratio > parameters['corr_ratio_max'] \
                or corr_step > parameters['corr_steps_max']:
            err_msg = ''
            if corr_dist > parameters['corr_dist_max']:
                err_msg += f'\ncorr_dist = {corr_dist: 0.4f} > \
                    corr_dist_max = {parameters["corr_dist_max"]: 0.4f};   '
            if corr_ratio > parameters['corr_ratio_max']:
                err_msg += f'\ncorr_ratio = {corr_ratio: 0.4f} > \
                    corr_ratio_max = {parameters["corr_ratio_max"]: 0.4f};   '
            if corr_step > parameters['corr_steps_max']:
                err_msg += f'\ncorr_step = {corr_step} > \
                    corr_steps_max = {parameters["corr_steps_max"]};   '
            cond = np.linalg.cond(J_y_pred)
            err_msg += f'cond(J) = {cond: 0.0f}'
            return y, J_y_pred, False, corr_step, corr_dist_tot, err_msg

    # if determinant of augmented Jacobian changes too much during correction,
    # then also decrease step size and return to predictor step
    J_y = J(y)
    detJ_y = np.linalg.det(np.vstack([J_y, tangent]))
    detJ_change = np.abs((detJ_y-detJ_y_pred) / detJ_y_pred)
    if detJ_change > parameters['detJ_change_max']:
        err_msg = f'\ndetJ_change = {detJ_change: 0.4f} > \
            detJ_change_max = {parameters["detJ_change_max"]}'
        return y, J_y, False, corr_step, corr_dist_tot, err_msg

    # else: corrector successful
    return y, J_y, True, corr_step, corr_dist_tot, ''


# %% after predictor-corrector step, decide how to proceed

def how_to_proceed(y_corr, y_old, s, ds, t_init, t_target,
                   trans_x, trans_x_old, corr_steps, corr_dist,
                   parameters, verbose=False, err_msg=''):
    """Whether or not to continue, and at which point.

    Case 1) finished
        Case 1a) successful
        Case 2b) failed

    Case 2) continue
        Case 2a) at current position
        Case 2b) at previous position with decreased step size
    """

    t = y_corr[-1]
    step_dist = np.linalg.norm(y_corr-y_old)
    s += step_dist

    increasing = True
    if t_target < t_init:
        increasing = False

    continue_tracing = True
    success_step = True

    # good case: path tracking fine
    if ds >= parameters['ds_min'] \
            and (
                (
                    increasing
                    and t >= t_init
                    and t <= t_target + parameters['ds_max']
                ) or (
                    not increasing
                    and t <= t_init
                    and t >= t_target - parameters['ds_max']
                )
            ):

        # (a) convergence criterion: trans_x -> trans_x_final
        if np.isinf(t_target):

            # check convergence
            if ds == parameters['ds_max']:
                y_test = np.max(np.abs(trans_x - trans_x_old)) / ds
                if y_test < parameters['y_tol']:
                    # path tracing successful
                    continue_tracing = False

        # (b) convergence criterion: t -> t_target
        else:

            # case 1: t still in bound -> update and continue loop
            if (increasing and t < t_target - parameters['t_tol']) \
                    or (not increasing and t > t_target + parameters['t_tol']):
                ds = inflate(ds=ds, parameters=parameters, t=t,
                             t_old=y_old[-1], t_target=t_target,
                             corr_steps=corr_steps, corr_dist=corr_dist)

            # case 2: t too far -> use previous step with decreased step size
            # (very rare due to ds <= |(t_target-t)/(t-t_old)|)
            elif (increasing and t > t_target) \
                    or (not increasing and t < t_target):
                t = y_old[-1]
                s -= step_dist
                ds = np.abs(t - t_target)
                y_corr = y_old.copy()

            # case 3: t_target - t_tol <= t <= t_target (for increasing t)
            else:
                # path tracing successful
                continue_tracing = False

    # bad case: path tracking stuck
    else:
        continue_tracing = False
        success_step = False
        if verbose:
            sys.stdout.write('\nHomotopy continuation got stuck.' + err_msg)
            sys.stdout.flush()

    return t, s, ds, y_corr.copy(), continue_tracing, success_step
