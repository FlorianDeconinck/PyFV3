import numpy as np

from ndsl import Namelist, StencilFactory, QuantityFactory
from ndsl.stencils.testing import pad_field_in_j
from ndsl.utils import safe_assign_array
from pyFV3.stencils.fillz import FillNegativeTracerValues
from pyFV3.testing import TranslateDycoreFortranData2Py
from pyFV3.tracers import Tracers
from typing import List


class TranslateFillz(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "dp2": {"istart": grid.is_, "iend": grid.ie, "axis": 1},
            "q2tracers": {"istart": grid.is_, "iend": grid.ie, "axis": 1},
        }
        self.in_vars["parameters"] = ["nq"]
        self.out_vars = {
            "q2tracers": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "axis": 1,
            }
        }
        self.max_error = 1e-13
        self.ignore_near_zero_errors = {"q2tracers": True}
        self.stencil_factory = stencil_factory
        self._quantity_factory = QuantityFactory.from_backend(
            sizer=stencil_factory.grid_indexing._sizer,
            backend=stencil_factory.backend,
        )

    def make_storage_data_input_vars(self, inputs, tracer_mapping: List[str]):
        storage_vars = self.storage_vars()
        info = storage_vars["dp2"]
        inputs["dp2"] = self.make_storage_data(
            np.squeeze(inputs["dp2"]), istart=info["istart"], axis=info["axis"]
        )
        info = storage_vars["q2tracers"]
        for i in range(int(inputs["nq"])):
            inputs["tracers"][tracer_mapping[i]] = self.make_storage_data(
                np.squeeze(inputs["q2tracers"][:, :, i]),
                istart=info["istart"],
                axis=info["axis"],
            )
        del inputs["q2tracers"]

    def compute(self, inputs):
        tracer_mapping = [
            "vapor",
            "liquid",
            "rain",
            "ice",
            "snow",
            "graupel",
            "qo3mr",
            "qsgs_tke",
        ]
        tracers = Tracers.make(
            quantity_factory=self._quantity_factory,
            tracer_mapping=tracer_mapping,
        )
        inputs["tracers"] = tracers

        self.make_storage_data_input_vars(inputs, tracer_mapping=tracer_mapping)
        for name, value in tuple(inputs.items()):
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        for name, value in tuple(inputs["tracers"].items()):
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs["tracers"][name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        inputs.pop("nq")
        fillz = FillNegativeTracerValues(
            self.stencil_factory,
            self.grid.quantity_factory,
        )
        fillz(**inputs)
        ds = self.grid.default_domain_dict()
        ds.update(self.out_vars["q2tracers"])
        tracers = np.zeros((self.grid.nic, self.grid.npz, inputs["tracers"].count))
        for varname, data in inputs["tracers"].items():
            index = tracer_mapping.index(varname)
            data[self.grid.slice_dict(ds)]
            safe_assign_array(
                tracers[:, :, index], np.squeeze(data[self.grid.slice_dict(ds)])
            )
        out = {"q2tracers": tracers}
        return out
