"""hompath class to save history of steps (path) of homotopy continuation.


HomCont:
Python software for solving systems of nonlinear equations
by homotopy continuation.

Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it
and/or modify it under the terms of the MIT License.
"""


import numpy as np
import matplotlib.pyplot as plt


# %% hompath class


class HomPath:
    """A container to save homotopy path.

    Survives abortion of computation.

    Variables of interest are saved in separate arrays,
    each step corresponds to one entry or row.


    Attributes
    ----------
    t : np.ndarray
        Homotopy parameter.
        shape=(max_steps,), dtype=np.float64
    x : np.ndarray
        Variables to track.
        Euclidean distance between consecutive steps add to path length.
        shape=(max_steps, dim), dtype=np.float64
    s : np.ndarray
        Path length.
        shape=(max_steps,), dtype=np.float64
    cond : np.ndarray
        Condition number of Jacobian to indicate numerical stability.
        shape=(max_steps,), dtype=np.float64
    sign : np.ndarray
        Orientation of path as sign of predictor tangent.
        shape=(max_steps,), dtype=np.float64
    x_transformer : callable
        Function to transform x for path length and plotting,
        by default lambda x: x.
    """

    def __init__(self, dim: int, max_steps: int = 500000,
                 x_transformer: callable = lambda x: x):
        """Initialization of path.

        Parameters
        ----------
        dim : int
            Number of variables to be tracked.
        max_steps : int, optional
            Maximum number of steps to be tracked, by default 500000.
        x_transformer : callable, optional
            Function to transform x for path length and plotting,
            by default lambda x : x.
        """
        self.t = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.x = np.nan * np.empty(shape=(max_steps, dim), dtype=np.float64)
        self.s = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.cond = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.sign = np.nan * np.empty(shape=max_steps, dtype=np.float64)
        self.x_transformer = x_transformer

    def update(self, t: float, x: np.ndarray,
               cond: float = None, sign: int = 1):
        # check input
        if len(x.shape) != 1:
            raise ValueError('"x" must be 1D array.')
        if x.shape[0] != self.x.shape[1]:
            raise ValueError(f'"x" should have length \
                             {self.x.shape[1]}, but has \
                             length {x.shape[0]}.')
        # save step
        nan_indices = np.where(np.isnan(self.t))[0]
        if len(nan_indices):
            idx = nan_indices[0]
            self.t[idx] = t
            self.x[idx] = x
            if idx == 0:
                trans_x = self.x_transformer(x)
                self.s[idx] = np.linalg.norm(trans_x)
            else:
                trans_x = self.x_transformer(x)
                trans_x_old = self.x_transformer(self.x[idx-1])
                ds = np.linalg.norm(trans_x - trans_x_old)
                self.s[idx] = self.s[idx-1] + ds
            if cond is not None:
                self.cond[idx] = cond
            if sign is not None:
                self.sign[idx] = sign

    def plotprep(self, max_points: int = 100000,
                 freq: int = 10):
        # cut NaN values
        idx = np.where(~np.isnan(self.t))[0]
        self.t = self.t[idx]
        self.x = self.x[idx, :]
        self.s = self.s[idx]
        self.cond = self.cond[idx]
        self.sign = self.sign[idx]
        # downsample
        while len(self.t) > max_points:
            self.t = self.t[::freq]
            self.x = self.x[::freq, :]
            self.s = self.s[::freq]
            self.cond = self.cond[::freq]
            self.sign = self.sign[::freq]

    def plot(self, x_name: str = 'variables'):
        x_Name = x_name[0].upper() + x_name[1:]
        self.plotprep()
        trans_x = self.x_transformer(self.x)
        trans_x_min = min([np.amin(trans_x), 0])
        trans_x_max = max([np.amax(trans_x), 1])
        fig = plt.figure(figsize=(10, 7))
        # path length -> homotopy parameter
        ax1 = fig.add_subplot(221)
        ax1.set_title('Homotopy path')
        ax1.set_xlabel(r'path length $s$')
        ax1.set_ylabel(r'homotopy parameter $t$')
        ax1.set_ylim(0, np.max([1, np.amax(self.t)]))
        ax1.plot(self.s, self.t)
        ax1.grid()
        # path length -> variables
        ax2 = fig.add_subplot(222)
        ax2.set_title(fr'{x_Name}')
        ax2.set_xlabel(r'path length $s$')
        ax2.set_ylabel(fr'{x_name}')
        ax2.set_ylim(trans_x_min, trans_x_max)
        ax2.plot(self.s, trans_x)
        ax2.grid()
        # s -> cond(J)
        ax3 = fig.add_subplot(223)
        ax3.set_title('Numerical stability')
        ax3.set_xlabel(r'path length $s$')
        ax3.set_ylabel(r'condition number $cond(J)$')
        ax3.plot(self.s, self.cond)
        ax3.grid()
        # t -> y
        ax4 = fig.add_subplot(224)
        ax4.set_title(fr'{x_Name} II')
        ax4.set_xlabel(r'homotopy parameter $t$')
        ax4.set_ylabel(fr'{x_name}')
        ax4.set_ylim(trans_x_min, trans_x_max)
        ax4.plot(self.t, trans_x)
        ax4.grid()
        # ax4 = fig.add_subplot(224)
        # ax4.set_title('Orientation')
        # ax4.set_xlabel(r'path length $s$')
        # ax4.set_ylabel('sign of tangent')
        # ax4.set_ylim(-1.5,1.5)
        # ax4.plot(self.s, self.sign)
        # ax4.grid()
        plt.tight_layout()
        plt.show()
        return fig


# %% testing


if __name__ == '__main__':

    dim = 2
    hom_path = HomPath(dim=dim)
    hom_path.update(t=0, x=np.array([1.5, 2.1]))
    hom_path.update(t=2.0, x=np.array([1, 2]))
    hom_path.plot()
