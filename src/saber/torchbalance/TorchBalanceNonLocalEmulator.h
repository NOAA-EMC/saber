/*
* (C) Copyright 2024 NOAA/NWS/NCEP/EMC
* (C) Copyright 2025- UCAR
*
* This software is licensed under the terms of the Apache Licence Version 2.0
* which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
*/

#pragma once

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "oops/base/FieldSet3D.h"
#include "oops/base/GeometryData.h"

namespace saber {

/// Stateful TorchScript non-local balance emulator.
///
/// The TorchScript module owns the rank-local linear operator. C++ owns data packing,
/// geometry tensors, and Atlas halo/adjoint-halo exchanges.
class TorchBalanceNonLocalEmulator {
 public:
  TorchBalanceNonLocalEmulator(const oops::GeometryData &,
                               const oops::FieldSet3D &,
                               const std::string &,
                               const std::vector<std::string> &);

  void multiply(oops::FieldSet3D &) const;
  void multiplyAD(oops::FieldSet3D &) const;
  void multiply(const oops::FieldSet3D &, oops::FieldSet3D &) const;
  void multiplyAD(const oops::FieldSet3D &, oops::FieldSet3D &) const;

 private:
  std::vector<std::string> readStringListAttribute(const std::string &) const;
  std::vector<torch::Tensor> packFields(const oops::FieldSet3D &,
                                        const std::vector<std::string> &) const;
  void addReturnedFields(const c10::List<torch::Tensor> &,
                         const std::vector<std::string> &,
                         const std::vector<int> &,
                         const bool,
                         oops::FieldSet3D &) const;
  torch::Tensor makeGeometryTensor(const std::string &) const;

  const oops::GeometryData & geometryData_;
  torch::jit::script::Module module_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
  std::vector<std::string> geometryTensorNames_;
  std::vector<torch::Tensor> geometryTensors_;
  torch::Tensor mask_;
};

}  // namespace saber
