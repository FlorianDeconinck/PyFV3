#!/bin/bash
rm -rf ./.gt_cache_*

export PACE_FLOAT_PRECISION=32
export PACE_CONSTANTS=GEOS
export FV3_DACEMODE=Python

python -m pytest -v -s -x \
    --data_path=../../../test_data/geos/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug \
    --backend=numpy \
    --multimodal_metric \
    ../../savepoint
