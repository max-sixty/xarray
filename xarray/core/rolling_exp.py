from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any, Generic, Hashable, Union, cast

import numpy as np
from packaging.version import Version
from typing_extensions import reveal_type

from xarray import DataArray, Dataset, Variable
from xarray.core.computation import apply_ufunc
from xarray.core.options import _get_keep_attrs
from xarray.core.pdcompat import count_not_none
from xarray.core.types import T_DataArrayOrSet, T_DataWithCoords
from xarray.core.utils import V

try:
    import numbagg
    from numbagg import move_exp_nanmean, move_exp_nansum

    _NUMBAGG_VERSION: Version | None = Version(numbagg.__version__)
except ImportError:
    _NUMBAGG_VERSION = None


def _get_alpha(
    obj: T_DataWithCoords,
    # One of comass, span, halflife, or alpha must be supplied
    # **kwargs: float | DataArray | Variable,
    **kwargs: float | DataArray,
) -> float | DataArray:
    """
    Convert comass, span, halflife to alpha.

    Only checks for invalid values if supplied with a number rather than an array;
    otherwise it's too expensive and we rely on the user to check.
    """

    window_type, window = next(iter(kwargs.items()))
    if next(iter(kwargs.values())) is not None:
        raise ValueError(
            f"Only one of `comass`, `span`, `halflife`, or `alpha` can be supplied, got {kwargs}"
        )

    if window_type not in ("comass", "span", "halflife", "alpha"):
        raise ValueError(
            f"window_type must be one of 'comass', 'span', 'halflife', or 'alpha', got {window_type}"
        )

    if isinstance(window, Number):
        # Check for invalid values
        if window_type == "comass":
            if window < 0:
                raise ValueError("comass must satisfy: comass >= 0")
        if window_type == "span":
            if window < 1:
                raise ValueError("span must satisfy: span >= 1")
        if window_type == "halflife":
            if window <= 0:
                raise ValueError("halflife must satisfy: halflife > 0")
        if window_type == "alpha":
            if not 0 < window <= 1:
                raise ValueError("alpha must satisfy: 0 < alpha <= 1")

    # It's a dimension name
    elif isinstance(window, Hashable):
        # if obj is None:
        #     raise ValueError(
        #         "If a dimension name is supplied, obj must be supplied too"
        #     )
        window = obj[window]

    # It's an array
    # if isinstance(window, (Variable, DataArray)):
    if isinstance(window, (DataArray)):
        # TODO: Actually we probably want to do downstream because for datasets we want
        # to broadcast separately...
        # if obj is None:
        #     raise ValueError("If an array is supplied, obj must be supplied too")
        window = window.broadcast_like(cast(Union[DataArray, Dataset], obj))

    # Now we have either a number of an array, we can calculate alpha
    if window_type == "comass":
        return 1 / (window + 1)
    elif window_type == "span":
        return 2 / (window + 1)
    elif window_type == "halflife":
        return 1 - np.exp(np.log(0.5) / window)
    elif window_type == "alpha":
        return window
    else:
        raise ValueError("unreachable")


class RollingExp(Generic[T_DataWithCoords]):
    """
    Exponentially-weighted moving window object.
    Similar to EWM in pandas

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : mapping of hashable to int (or float for alpha type)
        A mapping from the name of the dimension to create the rolling
        exponential window along (e.g. `time`) to the size of the moving window.
    window_type : {"span", "com", "halflife", "alpha"}, default: "span"
        The format of the previously supplied window. Each is a simple
        numerical transformation of the others. Described in detail:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

    Returns
    -------
    RollingExp : type of input argument
    """

    def __init__(
        self,
        obj: T_DataWithCoords,
        windows: Mapping[Any, int | float],
        window_type: str = "span",
        min_weight: float = 0.0,
    ):
        if _NUMBAGG_VERSION is None:
            raise ImportError(
                "numbagg >= 0.2.1 is required for rolling_exp but currently numbagg is not installed"
            )
        elif _NUMBAGG_VERSION < Version("0.2.1"):
            raise ImportError(
                f"numbagg >= 0.2.1 is required for rolling_exp but currently version {_NUMBAGG_VERSION} is installed"
            )
        elif _NUMBAGG_VERSION < Version("0.3.1") and min_weight > 0:
            raise ImportError(
                f"numbagg >= 0.3.1 is required for `min_weight > 0` within `.rolling_exp` but currently version {_NUMBAGG_VERSION} is installed"
            )

        self.obj: T_DataWithCoords = obj
        dim, window = next(iter(windows.items()))
        if _NUMBAGG_VERSION < Version("0.6.2") and not isinstance(window, Number):
            raise ImportError(
                f"numbagg >= 0.6.2 is required for non-constant window values (as an array) within `.rolling_exp` but currently version {_NUMBAGG_VERSION} is installed"
            )
        self.dim = dim
        self.alpha = _get_alpha(
            obj=cast(T_DataWithCoords, self), **{window_type: window}
        )
        self.min_weight = min_weight
        # Don't pass min_weight=0 so we can support older versions of numbagg
        kwargs = dict(alpha=self.alpha, axis=-1)
        if min_weight > 0:
            kwargs["min_weight"] = min_weight
        self.kwargs = kwargs

    def mean(self, keep_attrs: bool | None = None) -> T_DataWithCoords:
        """
        Exponentially weighted moving average.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").mean()
        <xarray.DataArray (x: 5)>
        array([1.        , 1.        , 1.69230769, 1.9       , 1.96694215])
        Dimensions without coordinates: x
        """

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dim_order = self.obj.dims

        return apply_ufunc(
            move_exp_nanmean,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=keep_attrs,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def sum(self, keep_attrs: bool | None = None) -> T_DataWithCoords:
        """
        Exponentially weighted moving sum.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").sum()
        <xarray.DataArray (x: 5)>
        array([1.        , 1.33333333, 2.44444444, 2.81481481, 2.9382716 ])
        Dimensions without coordinates: x
        """

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dim_order = self.obj.dims

        return apply_ufunc(
            move_exp_nansum,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=keep_attrs,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def std(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving standard deviation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").std()
        <xarray.DataArray (x: 5)>
        array([       nan, 0.        , 0.67936622, 0.42966892, 0.25389527])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().std(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nanstd,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def var(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving variance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").var()
        <xarray.DataArray (x: 5)>
        array([       nan, 0.        , 0.46153846, 0.18461538, 0.06446281])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().var(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nanvar,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def cov(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving covariance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").cov(da**2)
        <xarray.DataArray (x: 5)>
        array([       nan, 0.        , 1.38461538, 0.55384615, 0.19338843])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().cov(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nancov,
            self.obj,
            other,
            input_core_dims=[[self.dim], [self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def corr(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving correlation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").corr(da.shift(x=1))
        <xarray.DataArray (x: 5)>
        array([       nan,        nan,        nan, 0.4330127 , 0.48038446])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().cov(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nancorr,
            self.obj,
            other,
            input_core_dims=[[self.dim], [self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)
