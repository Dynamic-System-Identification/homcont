"""Main homsolve function performing numerical path tracing.


HomCont:
Solving systems of nonlinear equations by homotopy continuation.

Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it
and/or modify it under the terms of the MIT License.
"""


import numpy as np
import time
import datetime
import sys

import homcont.helpers as func
from homcont.defaults import PARAMETERS
from homcont.path import HomPath


# %% homsolve function


def homsolve(H: callable, y0: np.ndarray, J: callable = None,
             t_target: float = np.inf, max_steps: float = np.inf,
             x_transformer: callable = lambda x: x,
             hompath: HomPath = None, verbose: bool = False, **kwargs
             ) -> dict:
    """Function to perform homotopy continuation.

    Given:  1) System of equations H(x,t) = 0 with homotopy parameter t.
            2) Known solution x0 at t0, i.e. H(x0,t0) = 0.

    Wanted: A solution x* at t*, i.e. H(x*,t*) = 0.

    Idea:   Start at y0 = (x0,t0) and trace implied path up to y* = (x*,t*).


    Parameters
    ----------
    H : callable
        Homotopy function.
    y0 : np.ndarray
        The starting point for homotopy continuation.
        Must be 1D array.
        Must (approximately) solve the system H(x0) = 0.
        The homotopy parameter is stored in the last entry.
        The variables of interest are stored in the other entries.
    J : callable, optional
        Jacobian matrix.
        If not supplied, a finite difference approximation is used.
    t_target : float, optional
        Target value of the homotopy parameter, by default np.inf.
        If np.inf, iteration continues until all variables converge.
    max_steps : float, optional
        Maximum number of predictor-corrector iterations, by default np.inf.
    x_transformer : callable, optional
        Transformation of x to check for convergence, by default lambda x : x.
    hompath : HomPath, optional
        Instance of class hompath, by default None.
        Saves every successful predictor-corrector step.
        Survives abortion of solver.
    verbose : bool, optional
        Whether or not to show progress reports, by default False.

    **kwargs : Overwrite default path tracing parameters.
        x_tol : float
            Convergence criterion for variables, defaults to 1e-7.
            Active only if hom_par_target = np.inf.
        t_tol : float
            Convergence criterion for homotopy parameter, defaults to 1e-7.
            Active only if hom_par_target < np.inf.
        H_tol : float
            Accuracy of corrector steps, defaults to 1e-7.
        ds0 : float
            Initial step size, defaults to 0.01.
        ds_infl : float
            Step size inflation factor, defaults to 1.2.
        ds_defl : float
            Step size deflation factor, defaults to 0.5.
        ds_min : float
            Minimum step size, defaults to 1e-9.
        ds_max : float
            Maximum step size, defaults to 1000.
        corr_steps_max : int
            Maximum number of corrector steps, defaults to 20.
        corr_dist_max : float
            Maximum distance of corrector steps, defaults to 0.3.
        corr_ratio_max : float
            Maximum ratio between consecutive corrector steps, defaults to 0.3.
        detJ_change_max : float
            Maximum percentage change of determinant of augmented Jacobian
            between consecutive corrector steps, defaults to 0.3.
        bifurc_angle_min : float
            Minimum angle (in degrees) between two consecutive predictor
            tangents to be considered a bifurcation, defaults to 177.5.
            Bifurcations trigger a swap of path orientation.


    Returns
    -------
    dict
        Solution dictionary with the following entries.
            success : bool
                Whether or not homotopy continuation was successful.
            num_steps : int
                Number of predictor-corrector steps until convergence.
            path_len : float
                Length of y-path from starting point to solution.
            y : np.ndarray
                Solution vector.
                The final homotopy parameter is stored in the last entry.

    """

    # if not supplied, define Jacobian
    if J is None:
        import numdifftools as nd
        J = nd.Jacobian(H)

    # check input
    func.input_check(y0=y0, H=H, J=J, x_transformer=x_transformer)

    # start stopwatch
    tic = time.perf_counter()
    if verbose:
        print('=' * 50)
        print('Start homotopy continuation')

    # default tracing parameters
    parameters = PARAMETERS.copy()

    # use **kwargs to overwrite default parameters
    for key in parameters.keys():
        parameters[key] = kwargs.get(key, parameters[key])

    # fix initial homotopy parameter
    t_init = y0[-1]

    # old position
    y_old = y0.copy()
    t = y_old[-1]
    J_y_old = J(y_old)

    # get initial orientation of homotopy path
    Q, R = func.qr(J_y=J_y_old)
    sign = func.greedy_sign(y=y_old, Q=Q, R=R, t_target=t_target)
    tangent_old = func.tangent_qr(Q=Q, R=R, sign=sign)

    # check transversality of starting point
    func.transversality_check(tangent=tangent_old, parameters=parameters)

    # initial corrector
    y_corr = y_old.copy()
    trans_x = x_transformer(y_corr[:-1])
    trans_x_old = trans_x.copy()
    J_y_corr = J_y_old.copy()
    corr_steps = 0
    corr_dist = 0
    err_msg_corr = ''
    success_step = False

    # initialize homotopy continuation
    num_steps = 0
    s = 0
    ds = parameters['ds0']
    if hompath is not None:
        cond = np.linalg.cond(J_y_old)
        hompath.update(t=t, x=y_old[:-1], cond=cond, sign=sign)

    # start homotopy continuation

    # predictor-corrector loop
    continue_tracing = True
    while continue_tracing:

        num_steps += 1

        # compute new tangent at y_old
        Q, R = func.qr(J_y_old)
        tangent_new = func.tangent_qr(Q=Q, R=R, sign=sign)
        # tangent_new = func.tangent(J_y=J_y_old, sign=sign)

        # test for bifurcation point, swap sign if necessary
        swap_sign = func.swap_sign_bifurcation(
            sign=sign, parameters=parameters, tangent_new=tangent_new,
            tangent_old=tangent_old, verbose=verbose
            )
        if swap_sign:
            sign *= -1
            tangent_new *= -1

        # predictor-corrector loop
        success_corr = False
        while not success_corr:

            # predictor step
            y_pred = y_old + ds * tangent_new

            # if predictor step invalid, retry with smaller step size
            if np.isnan(H(y_pred)).any():
                ds = func.deflate(ds=ds, parameters=parameters)
                if ds < parameters['ds_min']:
                    break

            # if predictor step valid, continue with corrector step
            else:

                # corrector step
                (
                    y_corr, J_y_corr, success_corr,
                    corr_steps, corr_dist, err_msg_corr
                ) = func.corrector(
                    y_pred=y_pred, H=H, J=J, ds=ds, tangent=tangent_new,
                    parameters=parameters
                    )

                # if corrector step failed,
                # decrease step size and return to predictor step
                if not success_corr:
                    ds = func.deflate(ds=ds, parameters=parameters)
                    if ds < parameters['ds_min']:
                        break

        # check new position and decide how to proceed
        trans_x = x_transformer(y_corr[:-1])
        trans_x_old = x_transformer(y_old[:-1])
        t, s, ds, y_old, continue_tracing, success_step = func.how_to_proceed(
            y_corr=y_corr, y_old=y_old, s=s, ds=ds, t_init=t_init,
            t_target=t_target, trans_x=trans_x, trans_x_old=trans_x_old,
            corr_steps=corr_steps, corr_dist=corr_dist, parameters=parameters,
            verbose=verbose, err_msg=err_msg_corr
        )

        # get ready for next iteration
        tangent_old = tangent_new.copy()
        J_y_old = J_y_corr.copy()

        # print successful step
        cond = np.linalg.cond(J_y_old)
        if hompath is not None:
            hompath.update(t=t, x=y_corr[:-1], sign=sign, cond=cond)
        if verbose and success_corr and success_step:
            sys.stdout.write(f'\rStep {num_steps}:   t = {t: 0.4f},   '
                             + f's = {s: 0.2f},   ds = {ds: 0.2f},   '
                             + f'cond(J) = {cond: 0.0f}            ')
            sys.stdout.flush()

        # check number of steps
        if num_steps > max_steps:
            continue_tracing = False
            success_step = False
            if verbose:
                sys.stdout.write(f'\nMaximum number {max_steps} '
                                 + 'of steps reached.')
                sys.stdout.flush()

    # end of path tracing loop

    # stop stopwatch and report final step
    if verbose:
        H_test = np.max(np.abs(H(y_corr)))
        x_test = np.max(np.abs(trans_x-trans_x_old)) / ds
        elapsed = datetime.timedelta(seconds=round(time.perf_counter()-tic))
        print(f'\nFinal Result:   max|dx|/ds = {x_test:0.1E},   '
              + f'max|H| = {H_test:0.1E}')
        print(f'Time elapsed = {elapsed}')
        print('End homotopy continuation')
        print('=' * 50)

    return {
        'success': success_step,
        'num_steps': num_steps,
        'path_len': s,
        'y': y_corr
        }
