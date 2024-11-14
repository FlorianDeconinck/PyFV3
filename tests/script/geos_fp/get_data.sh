#!/bin/bash

set -e -x

TEST_DATA_PATH="../../../test_data/geos/"
mkdir -p $TEST_DATA_PATH
cd $TEST_DATA_PATH
wget -r -nH --cut-dir=5 -np -R "index.html*" https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/
