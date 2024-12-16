from typing import List

import dace

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float, FloatField
from pyFV3.stencils.fillz import FillNegativeTracerValues
from pyFV3.stencils.map_single import MapSingle
from pyFV3.tracers import Tracers


class MapNTracer:
    """
    Fortran code is mapn_tracer, test class is MapN_Tracer_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        kord: int,
        fill: bool,
        tracers: Tracers,
        exclude_tracers: List[str],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        self._exclude_tracers = exclude_tracers
        self._qs = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )

        self._map_single = {}
        for name in tracers.names():
            if name == "cloud":
                this_kord = 9
            else:
                this_kord = kord
            self._map_single[name] = MapSingle(
                stencil_factory,
                quantity_factory,
                this_kord,
                0,
                dims=[X_DIM, Y_DIM, Z_DIM],
            )

        if fill:
            self._fill_negative_tracers = True
            self._fillz = FillNegativeTracerValues(
                stencil_factory,
                quantity_factory,
                exclude_tracers=self._exclude_tracers,
            )
        else:
            self._fill_negative_tracers = False

    def __call__(
        self,
        pe1: FloatField,
        pe2: FloatField,
        dp2: FloatField,
        tracers: Tracers,
    ):
        """
        Remaps the tracer species onto the Eulerian grid
        and optionally fills negative values in the tracer fields
        Assumes the minimum value is 0 for each tracer

        Args:
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            dp2 (in): Difference in pressure between Eulerian levels
            tracers (inout): tracers to be remapped
        """
        for name in tracers.names():
            if name in self._exclude_tracers:
                continue
            self._map_single[name](tracers[name], pe1, pe2, self._qs)

        if self._fill_negative_tracers is True:
            self._fillz(dp2, tracers)
