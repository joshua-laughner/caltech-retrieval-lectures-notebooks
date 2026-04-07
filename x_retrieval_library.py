"""Code for the Caltech retrievals class"""

import time

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from numpy.typing import ArrayLike
from scipy import interpolate
from scipy.linalg import inv

# Disable the over-zealous pylint variable name conventions
# pylint: disable=invalid-name

# ============================================================== General setup

# Set up our data (taken from first MLS equator crossing on Jan 1, 2019).
#
# Note, the "fmt" comments below simply disable some automatic source formatting I used.
#
# fmt: off
z_data = np.array([0.35469648, 2.0135603, 3.622111, 5.1815395, 6.6900277, 8.146757,
                   9.537037, 10.85816 , 12.120895, 13.323103, 14.473699, 15.585789,
                   16.66661 , 17.72381 , 18.785986, 19.88103 , 21.016922, 22.196718,
                   23.409372, 24.635672, 25.861237, 27.080996, 28.299957, 29.52994 ,
                   30.778082, 32.0437 , 33.32278 , 34.60837 , 35.89749 , 37.192165,
                   38.49567 , 39.820965, 41.177185, 42.562073, 43.975468, 45.426296,
                   46.911858, 49.890263, 52.817924, 55.679977, 58.479008, 61.207726,
                   63.839584, 68.88816 , 73.791214, 78.59532 , 83.137245, 87.217735,
                   91.00151 , 94.83131 , 98.89325 , 103.40231 , 109.1182 , 117.07407 ,
                   126.50113])
T_data = np.array([299.9758, 290.7029, 282.0607, 273.21158, 263.9224, 254.78156,
                   240.26122, 230.15697, 219.47052, 208.60435, 201.09332, 194.89366,
                   189.95895, 186.48318, 191.73029, 198.18614, 206.27557, 213.81993,
                   217.97499, 218.67865, 217.71346, 216.61172, 217.42906, 220.53575,
                   223.89566, 226.7589, 228.68813, 229.0786, 229.943, 231.05702,
                   233.08907, 238.81361, 244.10013, 249.0242, 254.24986, 262.35178,
                   266.61884, 263.64734, 257.5859, 251.96533, 246.36646, 239.44647,
                   229.12225, 220.29475, 216.16745, 211.4879, 192.82748, 170.31734,
                   165.58655, 172.11748, 182.12318, 205.08037, 275.96072, 378.2, 378.2])
Ta_data = np.array([311.9915, 295.11774, 279.8465, 266.11893, 253.87607, 243.05908,
                    233.60902, 225.46703, 218.57416, 212.87155, 208.3003, 204.80148,
                    202.31621, 200.33243, 199.46873, 199.78888, 201.15764, 203.42142,
                    206.4122, 209.42676, 212.59938, 215.76196, 218.69034, 221.2645,
                    223.66364, 226.14873, 228.64406, 231.39745, 234.37071, 237.67535,
                    241.40215, 245.13364, 248.27058, 250.64024, 252.79182, 254.62236,
                    254.99687, 253.45027, 250.11534, 244.68779, 235.35904, 222.26154,
                    212.6158, 201.07045, 191.39256, 184.38904, 180.8668, 181.63278,
                    187.49388, 199.25702, 217.72911, 243.71709, 278.02786, 321.46832,
                    374.84543])
# fmt: on

SMALLER_FONT = 10.0


# ============================================================== Helper routines


def get_sigma_from_S(S: ArrayLike) -> ArrayLike:
    """Return the square root of the diagonal for a covariance matrix

    Negative values on the diagonal quietly set to zero

    Parameters
    ----------
    S : ArrayLike
        Supplied covariance matrix

    Returns
    -------
    ArrayLike
        Square root of diagonal (negatives set to zero)
    """
    # Avoid taking square root of negative number
    return np.sqrt(np.maximum(np.diag(S), 0.0))


def show_matrix(
    ax: plt.Axes,
    m: ArrayLike = None,
    title: str = None,
    unit: str = None,
    xlabel: str = None,
    ylabel: str = None,
    no_y: bool = False,
    vmin: float = None,
    vmax: float = None,
    aspect: float = None,
):
    """Show an image of a matrix in a given set of axes

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes to show plot on
    m : int, optional
        Matrix to show
    title : str, optional
        Title string to give
    unit : str, optional
        String for units of matrix values
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    no_y : bool, optiona
        If set, skip the y-axis
    vmin : float, optional
        Optional minimum value for range etc.
    vmax : float, optional
        Optional maximum value for range etc.
    """
    if m is None:
        return
    if xlabel is None:
        xlabel = "Altitude / km"
    if ylabel is None:
        ylabel = "Altitude / km"
    # Get the max abs value for setting colorbar
    if vmin is None:
        vmin = np.min(m)
    if vmax is None:
        vmax = np.max(m)
    maxabs = max([np.abs(vmin), np.abs(vmax)])
    color_info = ax.imshow(m, cmap="RdBu", vmin=-maxabs, vmax=maxabs, aspect=aspect)
    plt.colorbar(color_info, ax=ax, orientation="horizontal", label=unit)
    ax.set_title(title)
    if no_y:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def y_alt_label(ax: plt.Axes, no_y: bool):
    """Show or hide a y-axis for altitude

    Depending on the no_y flag, it either puts up a label and keeps the tick labels
    present or hides both.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes to put label on.
    no_y : bool
        If set, hide the axis label and tick labels
    """
    if no_y:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Altitude / km")


def li(
    a1: ArrayLike,
    a0: ArrayLike,
    z1: ArrayLike,
    z0: ArrayLike,
    a: ArrayLike,
) -> ArrayLike:
    """A very quick linear interpolator (for computing FWHM)

    Interpolates z as a function of a

    Parameters
    ----------
    a1, a0 : ArrayLike
        The two (sets of) a values
    z1, z0 : ArrayLike
        The two (sets of) z values
    a : ArrayLike
        The a value(s) we're interpolating to

    Returns
    -------
    ArrayLike
        Value of z corresponding to a using linear interpolation
    """
    return z0 + (z1 - z0) / (a1 - a0) * (a - a0)


def fwhm_vector(z: ArrayLike, a: ArrayLike) -> float:
    """Compute the Full Width at Half Maximum for a row in the Averaging Kernel

    Defined as the width (in km) of the shape of a

    Parameters
    ----------
    z : ArrayLike
        The vertical grid (in km) corresponding to the elements of x (and thus the
        row/columns of A)
    a : ArrayLike
        A row of the averaging kernel matrix A

    Returns
    -------
    float
        The FWHM for the vector a

    """
    # Locate the maximum
    max_i = np.argmax(a)
    max_a = a[max_i]
    half_max = max_a / 2
    # Check that there is actually a maximim!
    if max_a == np.min(a):
        return 0.0
    # Now search forward from the maximum to the first place where it goes below half
    # max
    try:
        i1 = max_i + np.nonzero(a[max_i:] < half_max)[0][0]
        i0 = i1 - 1
        z_upper = li(a[i1], a[i0], z[i1], z[i0], half_max)
    except IndexError:
        z_upper = z[-1]
    try:
        i0 = max_i - np.nonzero(a[max_i::-1] < half_max)[0][0]
        i1 = i0 + 1
        z_lower = li(a[i1], a[i0], z[i1], z[i0], half_max)
    except IndexError:
        z_lower = z[0]
    return z_upper - z_lower


def fwhm(z: ArrayLike, A: ArrayLike) -> ArrayLike:
    """Compute a set of Full Width at Half Maximums for an averaging kernel matrix

    Parameters
    ----------
    z : ArrayLike
        The vertical grid (in km) corresponding to the elements of x (and thus the
        row/columns of A)
    A : ArrayLike
        The Averaging Kernel matrix A

    Returns
    -------
    ArrayLike
        Vector of FWHMs for each level in x
    """
    m = A.shape[1]
    result = np.zeros(shape=[m])
    for i in range(m):
        result[i] = fwhm_vector(z, A[i, :])
    return result


# ============================================================== Classes


class Atmosphere:
    """A container for describing the state of the atmosphere"""

    def __init__(self):
        """Populate an atmosphere"""
        # Set up a vertical grid
        self.z = np.linspace(0, 100, 101)
        self.z_top = np.max(self.z)
        self.n = len(self.z)
        # Populate the truth and the a priori
        truth_interpolator = interpolate.interp1d(
            z_data, T_data, bounds_error=False, fill_value="extrapolate"
        )
        self.x_true = truth_interpolator(self.z)
        apriori_interpolator = interpolate.interp1d(
            z_data, Ta_data, bounds_error=False, fill_value="extrapolate"
        )
        self.x_a = apriori_interpolator(self.z) + 20.0


class MeasurementSystem:
    """A container for describing a measurement system"""

    def __init__(
        self,
        atmosphere: Atmosphere,
        m: int,
        m_width: float,
        m_margin: float,
        sigma_y: float,
    ):
        """Construct the weighting functions for a measurement

        Parameters
        ----------
        atmosphere : Atmosphere
            Model atmosphere
        m : int
            Number of measurements
        m_width : float
            Measurement width (in km)
        m_margin : float
            Space at the top and bottom of the profile within which the measurements are
            insensitive (in km)
        sigma_y : float
            Measurement noise (in K)
        """
        # Lay out the Gaussians that form the weighting functions
        self.atmosphere = atmosphere
        self.m = m
        self.m_width = m_width
        self.m_margin = m_margin
        self.z_peak = np.linspace(
            self.m_margin, atmosphere.z_top - self.m_margin, self.m
        )
        # Lock z_peak to the vertical grid
        self.z_peak = atmosphere.z[np.searchsorted(atmosphere.z, self.z_peak)]
        # Before we compute the weighting function matrix, build an (m,n) matrix of
        # delta_z
        delta_z = np.abs(self.z_peak[:, np.newaxis] - atmosphere.z[np.newaxis, :])
        # Now build some Gaussians as our weighting functions, avoid dividing by zero
        # for extreme cases
        self.K = np.exp(-((delta_z / (np.maximum(m_width, 1e-12))) ** 2))
        # Normalize K
        self.K /= np.max(np.sum(self.K, axis=1, keepdims=True))
        # Also construct the measurement covariance matrix, again, avoid dividing by
        # zero
        self.sigma_y = max(sigma_y, 1e-6)
        self.S_y = np.diag([self.sigma_y**2] * self.m)
        self.S_y_inv = np.diag([self.sigma_y ** (-2)] * self.m)


class Measurements:
    """The measurements of one particular state"""

    def __init__(self, measurement_system: MeasurementSystem, x: ArrayLike = None):
        """Populate a set of measurements from the atmosphere, or from supplied x

        Parameters
        ----------
        measurement_system : MeasurementSystem
            Description of the measurement system
        x : ArrayLike, optional
            The state vector to simulate measurements of (otherwise assumes the truth as
            defined in measurement_system)
        """
        self.measurement_system = measurement_system
        self.m = measurement_system.m
        self.atmosphere = self.measurement_system.atmosphere
        self.K = measurement_system.K
        self.sigma_y = measurement_system.sigma_y
        self.S_y = measurement_system.S_y
        self.S_y_inv = measurement_system.S_y_inv
        if x is None:
            x = self.atmosphere.x_true
        self.y_clean = self.K @ x
        self.y_noisy = self.y_clean + np.random.normal(0.0, self.sigma_y, size=self.m)


class Retrieval:
    """A class for performing and storing results of retrieval calcaultons"""

    def __init__(
        self,
        measurements: Measurements,
        sigma_a: float,
        a_width: float,
        y: ArrayLike = None,
    ):
        """Define a retrieval setup

        Parameters
        ----------
        measurements : MeasurementSystem
            The measurement system we're retrieving from
        sigma_a : float, optional
            The a priori uncertainty (in K)
        a_width : float, optional
            The a priori smoothing width (in km)
        y : ArrayLike, optional
            The measurement vector. If not supplied, get the "clean" measurements from
            the measurement system

        Raises
        ------
        ValueError
            If supplied y has the wrong shape
        """
        self.measurements = measurements
        self.measurement_system = self.measurements.measurement_system
        self.atmosphere = self.measurements.atmosphere
        self.n = self.atmosphere.n
        self.m = self.measurement_system.m
        self.sigma_a = max(sigma_a, 1e-6)
        self.a_width = a_width
        # Copy stuff from the atomsphere
        self.z = self.atmosphere.z
        self.x_true = self.atmosphere.x_true
        self.x_a = self.atmosphere.x_a
        # Copy a whole bunch of other stuff from the measurements
        self.y_clean = measurements.y_clean
        self.y_noisy = measurements.y_noisy
        self.K = measurements.K
        self.S_y = measurements.S_y
        self.S_y_inv = measurements.S_y_inv
        self.sigma_y = measurements.sigma_y
        # Construct S_a
        if self.a_width != 0.0 and self.a_width is not None:
            delta_z = np.abs(
                self.atmosphere.z[:, np.newaxis] - self.atmosphere.z[np.newaxis, :]
            )
            self.S_a = self.sigma_a**2 * np.exp(-delta_z / self.a_width)
            self.S_a_inv = inv(self.S_a)
        else:
            self.S_a = np.diag([self.sigma_a**2] * self.n)
            self.S_a_inv = np.diag([self.sigma_a ** (-2)] * self.n)
        # Pick a measurement vector
        if y is None or y is False:
            y = measurements.y_clean
        elif y is True:
            y = measurements.y_noisy
        else:
            if y.shape() != [self.m]:
                raise ValueError("Inappropriate y")
        self.y = y
        # Construct S_a_inv
        self.S_a_inv = inv(self.S_a)
        # Now do the retrieval calculation, capitalize on the diagonality of S_y_inv
        self.KT_SyI_K = (self.K.T * np.diag(self.S_y_inv)) @ self.K
        self.S_x_inv = self.S_a_inv + self.KT_SyI_K
        self.S_x = inv(self.S_x_inv)
        self.G = self.S_x @ self.K.T @ self.S_y_inv
        self.x_hat = self.x_a + self.G @ (y - self.K @ self.x_a)
        # Compute the averaging kernels
        self.A = self.G @ self.K
        # Compute A - I
        self.AmI = self.A - np.eye(self.n)
        # Compute various error terms
        self.S_x_noise = self.G @ self.S_y @ self.G.T
        self.S_x_smoothing = self.AmI @ self.S_a @ self.AmI.T
        self.S_x_total = self.S_x_noise + self.S_x_smoothing
        #
        self.sigma_a_full = get_sigma_from_S(self.S_a)
        self.sigma_x = get_sigma_from_S(self.S_x)
        self.sigma_x_noise = get_sigma_from_S(self.S_x_noise)
        self.sigma_x_smoothing = get_sigma_from_S(self.S_x_smoothing)
        self.sigma_x_total = get_sigma_from_S(self.S_x_total)

    def show_profiles(self, ax: plt.Axes, no_y: bool = False):
        """Show all the various profiles for the state vector and related quantities

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to show plot on
        no_y : bool, optional
            If set, hide y-axis label
        """
        # Show a priori and error range
        ax.fill_betweenx(
            self.z,
            self.x_a - self.sigma_a,
            self.x_a + self.sigma_a,
            alpha=0.05,
            color="darkgreen",
        )
        ax.plot(self.x_a, self.z, label="a priori", color="darkgreen", alpha=0.5)
        # Show the truth
        ax.plot(self.x_true, self.z, label="Truth", color="crimson", linewidth=2.0)
        # We're only going to show every other level when it comes to the error bars.
        downsample = slice(None, None, 2)
        ax.errorbar(
            self.x_hat[downsample],
            self.z[downsample],
            xerr=self.sigma_x[downsample],
            color="black",
            linestyle="",
            ecolor="grey",
        )
        ax.plot(self.x_hat, self.z, label="Retrieved", color="black")
        ax.legend()
        ax.set_xlabel("Temperature / K")
        ax.set_title("Temperature profiles")
        y_alt_label(ax, no_y)

    def show_K_profiles(self, ax: plt.Axes, no_y: bool = False):
        """Show profiles of the weighting functions

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to use
        no_y : bool, optional
            If set, hide y-axis labels etc.
        """
        ax.plot(self.K.T, self.z)
        ax.set_title(r"Weighting funcs. ($\mathbf{K}$)", fontsize=SMALLER_FONT)
        y_alt_label(ax, no_y)

    def show_G_profiles(self, ax: plt.Axes, no_y: bool = False):
        """Show profiles of the contribution functions

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to use
        no_y : bool, optional
            If set, hide y-axis labels etc.
        """
        ax.plot(self.G, self.z)
        ax.set_title(r"Contribution funcs. ($\mathbf{G}$)", fontsize=SMALLER_FONT)
        y_alt_label(ax, no_y)

    def show_A_profiles(self, ax: plt.Axes, no_y: bool = False):
        """Show profiles of the averaging kernels

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to use
        no_y : bool, optional
            If set, hide y-axis labels etc.
        """
        ax.plot(self.A.T, self.z, color="lightgrey", linewidth=0.5)
        ax.plot(self.A[::10, :].T, self.z)
        # ax.plot(np.sum(A, axis=1), self.z, color="black", linewidth=2.0)
        y_alt_label(ax, no_y)
        ax.set_title(r"Averaging kernels ($\mathbf{A}$)", fontsize=SMALLER_FONT)

    def show_A_areas(self, ax: plt.Axes, no_y: bool = False):
        """Plot averaging kernel areas

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to use
        no_y : bool, optional
            If set, hide y-axis labels etc.
        """
        A_total = np.sum(self.A, axis=1)
        ax.plot(A_total, self.z, color="black")
        ax.axvline(1.0, color="grey", linewidth=0.5)
        y_alt_label(ax, no_y)
        ax.set_xlim(left=0.0)
        dofs = np.sum(np.diag(self.A))
        ax.set_title("Kernel areas", fontsize=SMALLER_FONT)
        ax.annotate(f"{dofs:.1f} DOFs", (0.05, 5))

    def show_A_resolutions(self, ax: plt.Axes, no_y: bool = False):
        """Plot averaging kernel FWHMs

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to use
        no_y : bool, optional
            If set, hide y-axis labels etc.
        """
        ax.plot(fwhm(self.z, self.A), self.z)
        ax.set_title("Kernel resolutions", fontsize=SMALLER_FONT)
        ax.set_xlabel("FWHM / km")
        ax.set_xlim(left=0.0)
        y_alt_label(ax, no_y)

    def show_error_diagnostics(self, ax: plt.Axes, no_y: bool = False):
        """Show the various error terms

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to use
        no_y : bool, optional
            If set, hide y-axis labels etc.
        """
        ax.plot(self.sigma_x_total, self.z, label="Total", marker=".")
        ax.plot(self.sigma_x_noise, self.z, label="Noise")
        ax.plot(self.sigma_x_smoothing, self.z, label="Smoothing")
        ax.plot(self.sigma_a_full, self.z, label="A priori")
        # ax.plot(self.sigma_x, self.z, label="$\mathbf{S_x}$", color="black")
        ax.legend()
        ax.set_xlabel("Temperature error / K")
        ax.set_title("Uncertainties", fontsize=SMALLER_FONT)
        y_alt_label(ax, no_y)

    def show_dashboard(self, consistent_ranges: bool):
        """Display the control dashboard

        Parameters
        ----------
        consistent_ranges : bool
            If set, use the same color ranges for all matrices in a given space
        """
        # ------- Setup
        matplotlib.rcParams.update({"font.size": 8})
        plt.close()
        fig = plt.figure(figsize=(16, 6))

        outer = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1.5, 1.0, 2.0])
        # ------- Verious profiles
        left = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[0], hspace=0.30
        )
        self.show_K_profiles(plt.subplot(left[0, 0]))
        self.show_G_profiles(plt.subplot(left[0, 1]), True)
        self.show_error_diagnostics(plt.subplot(left[0, 2]), True)
        self.show_A_profiles(plt.subplot(left[1, 0]))
        self.show_A_areas(plt.subplot(left[1, 1]), True)
        self.show_A_resolutions(plt.subplot(left[1, 2]), True)
        #
        # ------ Main profiles
        self.show_profiles(plt.subplot(outer[1]))
        #
        # ------ Matrices
        right = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[2])
        # Probably should do these with a zip sometime in the future.
        matrices = [
            self.S_a,
            self.S_x,
            self.A,
            self.S_x_noise,
            self.S_x_smoothing,
            self.S_x_total,
        ]
        titles = [
            r"$\mathbf{S_a}$",
            r"$\mathbf{S_x}$",
            r"$\mathbf{A}$",
            r"$\mathbf{S_x}$ (noise)",
            r"$\mathbf{S_x}$ (smoothing)",
            r"$\mathbf{S_x}$ (total)",
        ]
        common_range_flags = [
            False,
            True,
            False,
            True,
            True,
            True,
        ]
        # Work out the common range if there is one to work out
        if consistent_ranges:
            consistent_matrices = [
                matrix for matrix, flag in zip(matrices, common_range_flags) if flag
            ]
            vmin = min(np.min(matrix) for matrix in consistent_matrices)
            vmax = max(np.max(matrix) for matrix in consistent_matrices)
        K2 = r"$K^2$"
        units = [K2] * 2 + [""] + [K2] * 3
        M = iter(matrices)
        t = iter(titles)
        u = iter(units)
        f = iter(common_range_flags)
        for r in range(2):
            for c in range(3):
                ax = plt.subplot(right[r, c])
                flag = next(f)
                if flag and consistent_ranges:
                    this_vmin, this_vmax = vmin, vmax
                else:
                    this_vmin, this_vmax = None, None
                show_matrix(
                    ax,
                    next(M),
                    next(t),
                    unit=next(u),
                    no_y=c > 0,
                    vmin=this_vmin,
                    vmax=this_vmax,
                )
                ax.tick_params(axis="x", pad=0)
        fig.tight_layout()
        # return fig
        plt.show()

    def show_matrix_dashboard(self, consistent_ranges: bool):
        """A special dashboard that just shows the matrices"""
        matplotlib.rcParams.update({"font.size": 8})
        plt.close()
        fig, axes = plt.subplots(2, 4, layout="constrained", figsize=[10, 6])
        axes_iterator = iter(axes.ravel())
        label = "Measurement #"
        #
        if consistent_ranges:
            consistent_matrices = [self.KT_SyI_K, self.S_a_inv, self.S_x_inv]
            vmin = min(np.min(matrix) for matrix in consistent_matrices)
            vmax = max(np.max(matrix) for matrix in consistent_matrices)
        else:
            vmin, vmax = None, None
        ax = next(axes_iterator)
        # For the weighting functions, use a sqrt to reduce the dynamic range of the
        # aspect ratio
        show_matrix(
            ax,
            self.K,
            title=r"$\mathbf{K}$",
            ylabel=label,
            aspect=np.sqrt(self.n / self.m),
        )
        #
        ax = next(axes_iterator)
        show_matrix(
            ax,
            self.S_y,
            title=r"$\mathbf{S_y}$",
            ylabel=label,
            xlabel=label,
        )
        #
        ax = next(axes_iterator)
        show_matrix(
            ax,
            self.S_y_inv,
            title=r"$\mathbf{S}_{\mathbf{y}}^{-1}$",
            xlabel=label,
            ylabel=label,
        )
        #
        ax = next(axes_iterator)
        show_matrix(
            ax,
            self.S_a,
            title=r"$\mathbf{S_a}$",
        )
        #
        #
        ax = next(axes_iterator)
        show_matrix(
            ax,
            self.S_a_inv,
            title=r"$\mathbf{S_a}^{-1}$",
            vmin=vmin,
            vmax=vmax,
        )
        ax = next(axes_iterator)
        #
        show_matrix(
            ax,
            self.KT_SyI_K,
            title=r"$\mathbf{K}^{T}\mathbf{S}_{\mathbf{y}}^{-1}\mathbf{K}$",
            vmin=vmin,
            vmax=vmax,
        )
        #
        ax = next(axes_iterator)
        show_matrix(
            ax,
            self.S_x_inv,
            title=r"$\mathbf{K}^T\mathbf{S}_{\mathbf{y}}^{-1}\mathbf{K} + \mathbf{S_a}^{-1}$",
            vmin=vmin,
            vmax=vmax,
        )
        #
        ax = next(axes_iterator)
        show_matrix(
            ax,
            self.S_x,
            title=r"$\mathbf{S_x} = \left(\mathbf{K}^{T}\mathbf{S}_{\mathbf{y}}^{-1}\mathbf{K} + \mathbf{S_a}^{-1}\right)^{-1}$",
        )

        plt.show()


def compute_all_results(
    m: int,
    m_width: float,
    m_margin: float,
    sigma_y: float,
    sigma_a: float,
    a_width: float,
    add_noise: bool,
):
    """Show one realization of the measurement system

    Parameters
    ----------
    m : int
        Number of measurments
    m_width : float
        Measurement width (in km)
    m_margin : float
        Space at the top and bottom of the profile within which the measurements are
        insensitive (in km)
    sigma_y : float
        Measurement noise (in K)
    sigma_a : float, optional
        The a priori uncertainty (in K)
    a_width : float, optional
        The a priori smoothing width (in km)
    add_noise : bool
        If set, add (Gaussian) noise to the measurements
    """
    atmosphere = Atmosphere()
    measurement_system = MeasurementSystem(atmosphere, m, m_width, m_margin, sigma_y)
    measurements = Measurements(measurement_system, x=atmosphere.x_true)
    retrieval = Retrieval(measurements, y=add_noise, sigma_a=sigma_a, a_width=a_width)
    return retrieval


def explore_retrievals():
    """Full UI for doing retrieval exploration"""
    # Set a default paramater
    style = {"description_width": "initial"}
    # Define widgets for each argument
    arguments = {
        "m": widgets.IntSlider(
            min=1,
            max=100,
            step=1,
            value=8,
            continuous_update=False,
            description="Num. measurements. (m)",
            style=style,
        ),
        "m_width": widgets.FloatSlider(
            min=0.0,
            max=50.0,
            value=10.0,
            continuous_update=False,
            description="Measurement width / km",
            style=style,
        ),
        "m_margin": widgets.FloatSlider(
            min=0.0,
            max=50.0,
            value=0.0,
            continuous_update=False,
            description="Measurement margin / km",
            style=style,
        ),
        "sigma_y": widgets.FloatSlider(
            min=0.0,
            max=100.0,
            value=1.0,
            continuous_update=False,
            description="Measurement noise / K",
            style=style,
        ),
        "sigma_a": widgets.FloatSlider(
            min=0.0,
            max=100.0,
            value=40.0,
            continuous_update=False,
            description="A priori uncertainty / K",
            style=style,
        ),
        "a_width": widgets.FloatSlider(
            min=0.0,
            max=50.0,
            value=10.0,
            continuous_update=False,
            description="Smoothing width / km",
            style=style,
        ),
        "add_noise": widgets.Checkbox(value=True, description="Add noise"),
    }
    # Get a lits of all the retrieval setting widgets
    setting_widgets = list(arguments.values())
    # Add a button to initiate the computations
    consistent_ranges = widgets.Checkbox(
        value=False, description="Consistent matrix ranges"
    )
    update_button = widgets.Button(description="Compute")
    notes_widget = widgets.Label(value="")
    # Put together the other non-retrieval control widgets
    other_control_widgets = [consistent_ranges, update_button, notes_widget]
    #
    # Put all those controls in a box
    controls = widgets.HBox(
        setting_widgets + other_control_widgets,
        layout=widgets.Layout(flex_flow="row wrap"),
    )
    # Now have a tab widget for the outputs
    output_overview = widgets.Output()
    output_matrices = widgets.Output()
    output_collection = widgets.Tab(children=[output_overview, output_matrices])
    output_collection.set_title(0, "Overview")
    output_collection.set_title(1, "Matrices")

    # Inline function to show the output
    def show_output(_):
        # Temporarily disable/rename the Compute button
        update_button.disabled = True
        update_button.description = "(Computing...)"
        # Build the arguments to the one_realization function from the values of the
        # user-suppled inputs in the widgets.
        kwargs = {key: item.value for key, item in arguments.items()}
        # While we're here, get the consistent_ranges value
        use_consistent_ranges = consistent_ranges.value
        # Clear the previous results
        for output in [output_overview, output_matrices]:
            output.clear_output(wait=True)
        # Perform the computations
        start_time = time.time()
        retrieval = compute_all_results(**kwargs)
        mid_time = time.time()
        with output_overview:
            retrieval.show_dashboard(consistent_ranges=use_consistent_ranges)
        with output_matrices:
            retrieval.show_matrix_dashboard(consistent_ranges=use_consistent_ranges)
        end_time = time.time()
        compute_time = mid_time - start_time
        render_time = end_time - mid_time
        notes_widget.value = (
            f"Computing: {compute_time:.3f} s, Rendering: {render_time:.3f} s"
        )
        # Reset and reenable the compute button
        update_button.description = "Compute"
        update_button.disabled = False

    # Attach the show_output function to the update button.
    update_button.on_click(show_output)
    # # Observe the control widgets
    # for widget in setting_widgets:
    #     widget.observe(show_output)
    # Define the user interface as the fusion of the controls and the output
    ui = widgets.VBox([controls, output_collection])
    return ui
