import pytest

import ndsl.dsl.gt4py_utils as utils
from ndsl import Namelist, StencilFactory, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import ParallelTranslate
from pyFV3.stencils import FiniteVolumeTransport, TracerAdvection
from pyFV3.utils.functional_validation import get_subset_func
from pyFV3.tracers import Tracers


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
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
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
        tracers = Tracers.make_from_fortran(
            quantity_factory=self._quantity_factory,
            tracer_mapping=[
                "vapor",
                "liquid",
                "rain",
                "ice",
                "snow",
                "graupel",
                "qo3mr",
                "qsgs_tke",
            ],
            tracer_data=inputs["tracers"],
        )
        self._base.make_storage_data_input_vars(inputs, dict_4d=False)
        inputs.pop("nq")  # Fortran NQ is intrinsic to Tracers (e.g Tracers.count)
        all_tracers = inputs.pop("tracers")
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
        )
        inputs["x_mass_flux"] = inputs.pop("mfxd")
        inputs["y_mass_flux"] = inputs.pop("mfyd")
        inputs["x_courant"] = inputs.pop("cxd")
        inputs["y_courant"] = inputs.pop("cyd")
        self.tracer_advection(tracers=tracers, **inputs)
        inputs["mfxd"] = inputs.pop("x_mass_flux")
        inputs["mfyd"] = inputs.pop("y_mass_flux")
        inputs["cxd"] = inputs.pop("x_courant")
        inputs["cyd"] = inputs.pop("y_courant")
        # Put back un-advected tracers
        # Tracers have -1 on all cartesian because of NDSL padding
        # Dev note: qcld is not advected in Pace dataset for some reason
        tracers_as_4d = tracers.as_fortran_4D()
        for idx in range(0, tracers_as_4d.shape[3]):
            all_tracers[:-1, :-1, :-1, idx] = tracers_as_4d[:, :, :, idx]
        inputs["tracers"] = all_tracers
        # need to convert tracers dict to [x, y, z, n_tracer] array before subsetting
        outputs = self._base.slice_output(inputs)
        outputs["tracers"] = self.subset_output("tracers", outputs["tracers"])
        return outputs

    def get_advected_tracer_dict(self, all_tracers, nq):
        all_tracers = {**all_tracers}  # make a new dict so we don't modify the input
        properties = self.inputs["tracers"]
        for name in utils.tracer_variables:
            self.grid.quantity_dict_update(
                all_tracers,
                name,
                dims=properties["dims"],
                units=properties["units"],
            )
        tracer_names = utils.tracer_variables[:nq]
        return {name: all_tracers[name + "_quantity"] for name in tracer_names}

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
