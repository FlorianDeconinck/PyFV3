import pytest

from ndsl import Namelist, QuantityFactory, StencilFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import ParallelTranslate
from pyFV3.stencils import FiniteVolumeTransport, TracerAdvection
from pyFV3.tracers import Tracers
from pyFV3.utils.functional_validation import get_subset_func


class TranslateTracer2D1L(ParallelTranslate):
    inputs = {
        "tracers": {
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/m^2",
        }
    }

    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self._base.in_vars["data_vars"] = {
            "tracers": {},
            "dp1": {},
            "mfxd_R4": grid.x3d_compute_dict(),
            "mfyd_R4": grid.y3d_compute_dict(),
            "cxd_R4": grid.x3d_compute_domain_y_dict(),
            "cyd_R4": grid.y3d_compute_domain_x_dict(),
        }
        self._base.in_vars["parameters"] = ["nq"]
        self._base.out_vars = self._base.in_vars["data_vars"]
        self.stencil_factory = stencil_factory
        self._quantity_factory = QuantityFactory.from_backend(
            sizer=stencil_factory.grid_indexing._sizer,
            backend=stencil_factory.backend,
        )
        self.namelist = namelist
        self._subset = get_subset_func(
            self.grid.grid_indexing,
            dims=[X_DIM, Y_DIM, Z_DIM],
            n_halo=((0, 0), (0, 0)),
        )

    def collect_input_data(self, serializer, savepoint):
        input_data = self._base.collect_input_data(serializer, savepoint)
        return input_data

    def compute_parallel(self, inputs, communicator):
        tracers = Tracers.make_from_4D_array(
            quantity_factory=self._quantity_factory,
            tracer_mapping=Tracers.blind_mapping_from_data(inputs["tracers"]),
            tracer_data=inputs["tracers"],
        )
        self._base.make_storage_data_input_vars(inputs, dict_4d=False)
        inputs.pop("tracers")
        inputs.pop("nq")  # Fortran NQ is intrinsic to Tracers (e.g Tracers.count)
        transport = FiniteVolumeTransport(
            stencil_factory=self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            grid_data=self.grid.grid_data,
            damping_coefficients=self.grid.damping_coefficients,
            grid_type=self.grid.grid_type,
            hord=self.namelist.hord_tr,
        )

        self.tracer_advection = TracerAdvection(
            self.stencil_factory,
            self.grid.quantity_factory,
            transport,
            self.grid.grid_data,
            communicator,
            tracers,
            exclude_tracers=["cloud"],
            update_mass_courant=False,
        )
        inputs["x_mass_flux"] = inputs.pop("mfxd_R4")
        inputs["y_mass_flux"] = inputs.pop("mfyd_R4")
        inputs["x_courant"] = inputs.pop("cxd_R4")
        inputs["y_courant"] = inputs.pop("cyd_R4")
        self.tracer_advection(tracers=tracers, **inputs)
        inputs["mfxd_R4"] = inputs.pop("x_mass_flux")
        inputs["mfyd_R4"] = inputs.pop("y_mass_flux")
        inputs["cxd_R4"] = inputs.pop("x_courant")
        inputs["cyd_R4"] = inputs.pop("y_courant")
        inputs["tracers"] = tracers.as_4D_array()
        outputs = self._base.slice_output(inputs)
        outputs["tracers"] = self.subset_output("tracers", outputs["tracers"])
        return outputs

    def compute_sequential(self, a, b):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )

    def subset_output(self, varname: str, output):
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if varname in ["tracers"]:
            return self._subset(output)
        else:
            return output
