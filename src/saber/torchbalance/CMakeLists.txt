# (C) Copyright 2024 NOAA/NWS/NCEP/EMC
# (C) Copyright 2025- UCAR.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

if( Torch_FOUND )
    list( APPEND torchbalance_src_files_list
    TorchBalance.h
    TorchBalance.cc
    TorchBalanceSurfaceEmulator.h
    TorchBalanceSurfaceEmulator.cc
    TorchBalanceVerticalEmulator.h
    TorchBalanceVerticalEmulator.cc
    TorchBalanceNonLocalEmulator.h
    TorchBalanceNonLocalEmulator.cc
    )
endif()

set( torchbalance_src_files ${torchbalance_src_files_list} PARENT_SCOPE )
