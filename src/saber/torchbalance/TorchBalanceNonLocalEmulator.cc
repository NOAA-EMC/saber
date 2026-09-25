/*
* (C) Copyright 2024 NOAA/NWS/NCEP/EMC
* (C) Copyright 2025- UCAR
*
* This software is licensed under the terms of the Apache Licence Version 2.0
* which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
*/

#include "saber/torchbalance/TorchBalanceNonLocalEmulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "atlas/array.h"
#include "atlas/functionspace/NodeColumns.h"

#include "eckit/exception/Exceptions.h"

#include "oops/util/Logger.h"

namespace {

struct ValueStats {
  int64_t nfinite = 0;
  int64_t nnan = 0;
  int64_t ninf = 0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
};

void updateStats(const double value, ValueStats & stats) {
  if (std::isnan(value)) {
    ++stats.nnan;
    return;
  }
  if (!std::isfinite(value)) {
    ++stats.ninf;
    return;
  }
  ++stats.nfinite;
  stats.min = std::min(stats.min, value);
  stats.max = std::max(stats.max, value);
}

std::string statsSummary(const ValueStats & stats) {
  std::ostringstream oss;
  oss << "finite=" << stats.nfinite
      << " nan=" << stats.nnan
      << " inf=" << stats.ninf;
  if (stats.nfinite > 0) {
    oss << " min=" << stats.min
        << " max=" << stats.max;
  }
  return oss.str();
}

ValueStats tensorStats(torch::Tensor tensor) {
  ValueStats stats;
  torch::Tensor cpu = tensor.to(torch::kCPU).to(torch::kFloat).contiguous();
  torch::Tensor flat_tensor = cpu.view({-1});
  auto flat = flat_tensor.accessor<float, 1>();
  for (int64_t idx = 0; idx < cpu.numel(); ++idx) {
    updateStats(static_cast<double>(flat[idx]), stats);
  }
  return stats;
}

}  // namespace

namespace saber {

// --------------------------------------------------------------------------------------

TorchBalanceNonLocalEmulator::TorchBalanceNonLocalEmulator(
    const oops::GeometryData & geometryData,
    const oops::FieldSet3D & xb,
    const std::string & torchscriptPath,
    const std::vector<std::string> & geometryTensorNames,
    const std::vector<std::string> & backgroundTensorNames)
  : geometryData_(geometryData),
    module_(torch::jit::load(torchscriptPath)),
    geometryTensorNames_(geometryTensorNames),
    backgroundTensorNames_(backgroundTensorNames) {
  module_.eval();

  inputNames_ = readStringListAttribute("input_names");
  outputNames_ = readStringListAttribute("output_names");

  for (const auto & inputName : inputNames_) {
    if (!xb.has(inputName)) {
      throw eckit::Exception("TorchBalance non-local emulator missing input field '"
                             + inputName + "'", Here());
    }
  }
  for (const auto & outputName : outputNames_) {
    if (!xb.has(outputName)) {
      throw eckit::Exception("TorchBalance non-local emulator missing output field '"
                             + outputName + "'", Here());
    }
  }

  geometryTensors_.reserve(geometryTensorNames_.size() + backgroundTensorNames_.size());
  for (const auto & tensorName : geometryTensorNames_) {
    geometryTensors_.push_back(makeGeometryTensor(tensorName));
  }
  for (const auto & tensorName : backgroundTensorNames_) {
    geometryTensors_.push_back(makeBackgroundTensor(tensorName, xb));
  }

  const int nnodes = geometryData_.functionSpace().lonlat().shape(0);
  mask_ = torch::ones({nnodes, 1}, torch::kFloat);

  oops::Log::info() << "TorchBalance non-local emulator loaded " << torchscriptPath
                    << " with " << inputNames_.size() << " inputs, "
                    << outputNames_.size() << " outputs, "
                    << geometryTensorNames_.size() << " geometry tensors, and "
                    << backgroundTensorNames_.size() << " background tensors" << std::endl;
}

// --------------------------------------------------------------------------------------

std::vector<std::string> TorchBalanceNonLocalEmulator::readStringListAttribute(
    const std::string & attrName) const {
  std::vector<std::string> values;
  try {
    for (const auto & value : module_.attr(attrName).toListRef()) {
      values.push_back(value.toString()->string());
    }
  } catch (const c10::Error & err) {
    throw eckit::Exception("TorchBalance non-local emulator requires TorchScript attribute '"
                           + attrName + "'", Here());
  }
  return values;
}

// --------------------------------------------------------------------------------------

std::vector<torch::Tensor> TorchBalanceNonLocalEmulator::packFields(
    const oops::FieldSet3D & fset,
    const std::vector<std::string> & fieldNames) const {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(fieldNames.size());

  for (const auto & fieldName : fieldNames) {
    if (!fset.has(fieldName)) {
      throw eckit::Exception("TorchBalance non-local emulator missing field '"
                             + fieldName + "' during tensor packing", Here());
    }

    const atlas::Field & field = fset[fieldName];
    auto view = atlas::array::make_view<double, 2>(field);
    const int nnodes = view.shape(0);
    const int nlevels = view.shape(1);

    ValueStats fieldStats;
    std::vector<float> data;
    data.reserve(nnodes * nlevels);
    for (atlas::idx_t jnode = 0; jnode < nnodes; ++jnode) {
      for (int level = 0; level < nlevels; ++level) {
        const double value = view(jnode, level);
        updateStats(value, fieldStats);
        data.push_back(static_cast<float>(value));
      }
    }

    tensors.push_back(torch::from_blob(data.data(), {nnodes, nlevels}, torch::kFloat).clone());
    oops::Log::debug() << "TorchBalance non-local packFields: " << fieldName
                       << " shape=[" << nnodes << ", " << nlevels << "] "
                       << statsSummary(fieldStats) << std::endl;
  }

  return tensors;
}

// --------------------------------------------------------------------------------------

void TorchBalanceNonLocalEmulator::addReturnedFields(
    const c10::List<torch::Tensor> & returnedTensors,
    const std::vector<std::string> & fieldNames,
    const std::vector<int> & addOnGhost,
    const bool adjointHaloExchange,
    oops::FieldSet3D & fset) const {
  if (static_cast<size_t>(returnedTensors.size()) != fieldNames.size()) {
    throw eckit::Exception(
        "TorchBalance non-local emulator returned " + std::to_string(returnedTensors.size())
        + " tensors for " + std::to_string(fieldNames.size()) + " configured fields", Here());
  }

  auto ghostView = atlas::array::make_view<int, 1>(geometryData_.functionSpace().ghost());

  for (size_t jfield = 0; jfield < fieldNames.size(); ++jfield) {
    atlas::Field & field = fset[fieldNames[jfield]];
    auto view = atlas::array::make_view<double, 2>(field);
    torch::Tensor tensor = returnedTensors.get(jfield).to(torch::kCPU).to(torch::kFloat)
      .contiguous();

    if (tensor.dim() != 2 || tensor.size(0) != view.shape(0) || tensor.size(1) != view.shape(1)) {
      std::ostringstream err;
      err << "TorchBalance non-local emulator returned tensor " << jfield
          << " for field '" << fieldNames[jfield] << "' with shape " << tensor.sizes()
          << ", expected [" << view.shape(0) << ", " << view.shape(1) << "]";
      throw eckit::Exception(err.str(), Here());
    }

    oops::Log::debug() << "TorchBalance non-local addReturnedFields: tensor "
                       << fieldNames[jfield] << " shape=[" << tensor.size(0)
                       << ", " << tensor.size(1) << "] "
                       << statsSummary(tensorStats(tensor)) << std::endl;

    auto accessor = tensor.accessor<float, 2>();
    ValueStats outStats;
    for (atlas::idx_t jnode = 0; jnode < view.shape(0); ++jnode) {
      if (!addOnGhost[jfield] && ghostView(jnode) > 0) continue;
      for (int level = 0; level < view.shape(1); ++level) {
        view(jnode, level) += static_cast<double>(accessor[jnode][level]);
        updateStats(view(jnode, level), outStats);
      }
    }

//     if (adjointHaloExchange) {
//      field.adjointHaloExchange();
//      field.set_dirty(true);
//    } else {
//      field.haloExchange();
//      field.set_dirty(false);
//    }

    oops::Log::debug() << "TorchBalance non-local addReturnedFields: field "
                       << fieldNames[jfield] << " after add+halo "
                       << statsSummary(outStats) << std::endl;
  }
}

// --------------------------------------------------------------------------------------

torch::Tensor TorchBalanceNonLocalEmulator::makeGeometryTensor(const std::string & tensorName)
    const {
  const auto & fs = geometryData_.functionSpace();

  if (tensorName == "lonlat") {
    auto view = atlas::array::make_view<double, 2>(fs.lonlat());
    std::vector<float> data;
    data.reserve(view.shape(0) * view.shape(1));
    for (atlas::idx_t jnode = 0; jnode < view.shape(0); ++jnode) {
      for (int jdim = 0; jdim < view.shape(1); ++jdim) {
        data.push_back(static_cast<float>(view(jnode, jdim)));
      }
    }
    return torch::from_blob(data.data(), {view.shape(0), view.shape(1)}, torch::kFloat).clone();
  }

  if (tensorName == "global_index") {
    auto view = atlas::array::make_view<atlas::gidx_t, 1>(fs.global_index());
    std::vector<int64_t> data;
    data.reserve(view.shape(0));
    for (atlas::idx_t jnode = 0; jnode < view.shape(0); ++jnode) {
      data.push_back(static_cast<int64_t>(view(jnode)));
    }
    return torch::from_blob(data.data(), {view.shape(0)}, torch::kLong).clone();
  }

  if (tensorName == "ghost") {
    auto view = atlas::array::make_view<int, 1>(fs.ghost());
    std::vector<int64_t> data;
    data.reserve(view.shape(0));
    for (atlas::idx_t jnode = 0; jnode < view.shape(0); ++jnode) {
      data.push_back(static_cast<int64_t>(view(jnode)));
    }
    return torch::from_blob(data.data(), {view.shape(0)}, torch::kLong).clone();
  }

  if (tensorName == "cell_node_connectivity" || tensorName == "cell_node_counts" ||
      tensorName == "edge_node_connectivity") {
    if (fs.type() != "NodeColumns") {
      throw eckit::Exception("TorchBalance non-local geometry tensor '" + tensorName
                             + "' is currently supported only for NodeColumns", Here());
    }
    atlas::functionspace::NodeColumns nodeColumns(fs);
    atlas::Mesh mesh(nodeColumns.mesh());

    if (tensorName == "cell_node_connectivity" || tensorName == "cell_node_counts") {
      const auto & connectivity = mesh.cells().node_connectivity();
      const atlas::idx_t ncells = mesh.cells().size();
      atlas::idx_t maxNodesPerCell = 0;
      for (atlas::idx_t cell = 0; cell < ncells; ++cell) {
        maxNodesPerCell = std::max(maxNodesPerCell, connectivity.cols(cell));
      }

      if (tensorName == "cell_node_counts") {
        std::vector<int64_t> data;
        data.reserve(ncells);
        for (atlas::idx_t cell = 0; cell < ncells; ++cell) {
          data.push_back(static_cast<int64_t>(connectivity.cols(cell)));
        }
        return torch::from_blob(data.data(), {ncells}, torch::kLong).clone();
      }

      std::vector<int64_t> data(ncells * maxNodesPerCell, -1);
      for (atlas::idx_t cell = 0; cell < ncells; ++cell) {
        for (atlas::idx_t inode = 0; inode < connectivity.cols(cell); ++inode) {
          data[cell * maxNodesPerCell + inode] =
            static_cast<int64_t>(connectivity(cell, inode));
        }
      }
      return torch::from_blob(data.data(), {ncells, maxNodesPerCell}, torch::kLong).clone();
    }

    const auto & connectivity = mesh.cells().node_connectivity();
    const atlas::idx_t ncells = mesh.cells().size();
    std::set<std::pair<int64_t, int64_t>> edgeSet;

    for (atlas::idx_t cell = 0; cell < ncells; ++cell) {
      const atlas::idx_t nnodesInCell = connectivity.cols(cell);
      if (nnodesInCell < 2) continue;
      for (atlas::idx_t inode = 0; inode < nnodesInCell; ++inode) {
        const int64_t nodeA = static_cast<int64_t>(connectivity(cell, inode));
        const int64_t nodeB = static_cast<int64_t>(
            connectivity(cell, (inode + 1) % nnodesInCell));
        if (nodeA < 0 || nodeB < 0 || nodeA == nodeB) continue;
        edgeSet.emplace(std::min(nodeA, nodeB), std::max(nodeA, nodeB));
      }
    }

    const atlas::idx_t nedges = edgeSet.size();
    std::vector<int64_t> data;
    data.reserve(nedges * 2);
    for (const auto & edge : edgeSet) {
      data.push_back(edge.first);
      data.push_back(edge.second);
    }
    return torch::from_blob(data.data(), {nedges, 2}, torch::kLong).clone();
  }

  throw eckit::Exception("TorchBalance non-local emulator unsupported geometry tensor '"
                         + tensorName + "'", Here());
}

// --------------------------------------------------------------------------------------

torch::Tensor TorchBalanceNonLocalEmulator::makeBackgroundTensor(
    const std::string & tensorName,
    const oops::FieldSet3D & xb) const {
  if (!xb.has(tensorName)) {
    throw eckit::Exception("TorchBalance non-local emulator missing background field '"
                           + tensorName + "'", Here());
  }

  const atlas::Field & field = xb[tensorName];
  atlas::Field exchanged = field.clone();
  exchanged.set_dirty(true);
  exchanged.haloExchange();
  auto view = atlas::array::make_view<double, 2>(exchanged);
  const int nnodes = view.shape(0);
  const int nlevels = view.shape(1);

  std::vector<float> data;
  data.reserve(nnodes * nlevels);
  for (atlas::idx_t jnode = 0; jnode < nnodes; ++jnode) {
    for (int level = 0; level < nlevels; ++level) {
      data.push_back(static_cast<float>(view(jnode, level)));
    }
  }

  return torch::from_blob(data.data(), {nnodes, nlevels}, torch::kFloat).clone();
}

// --------------------------------------------------------------------------------------

void TorchBalanceNonLocalEmulator::multiply(oops::FieldSet3D & fset) const {
  multiply(static_cast<const oops::FieldSet3D &>(fset), fset);
}

// --------------------------------------------------------------------------------------

void TorchBalanceNonLocalEmulator::multiply(const oops::FieldSet3D & inputFset,
                                            oops::FieldSet3D & outputFset) const {
  atlas::FieldSet fsetCopy;
  for (const auto & field : inputFset.fieldSet()) {
    fsetCopy.add(field.clone());
  }

//  for (const auto & inputName : inputNames_) {
//    fsetCopy[inputName].haloExchange();
//  }
  oops::FieldSet3D packedFset(inputFset.validTime(), inputFset.commGeom());
  packedFset.shallowCopy(fsetCopy);
  std::vector<torch::Tensor> inputTensors = packFields(packedFset, inputNames_);

  c10::List<torch::Tensor> inputList;
  for (const auto & tensor : inputTensors) inputList.push_back(tensor);

  c10::List<torch::Tensor> geometryList;
  for (const auto & tensor : geometryTensors_) geometryList.push_back(tensor);

  std::vector<torch::jit::IValue> args{inputList, geometryList, mask_};
  c10::List<torch::Tensor> outputTensors;
  try {
    outputTensors = module_.get_method("multiply")(args).toTensorList();
  } catch (const c10::Error & err) {
    throw eckit::Exception(
        "TorchBalance non-local TorchScript method multiply(inputs, geometry, mask) failed",
        Here());
  }

  oops::Log::debug() << "TorchBalance non-local multiply: returned "
                     << outputTensors.size() << " tensors" << std::endl;
  for (size_t jfield = 0; jfield < outputNames_.size() &&
                          jfield < static_cast<size_t>(outputTensors.size()); ++jfield) {
    oops::Log::debug() << "TorchBalance non-local multiply: output " << outputNames_[jfield]
                       << " " << statsSummary(tensorStats(outputTensors.get(jfield)))
                       << std::endl;
  }

  addReturnedFields(outputTensors, outputNames_, std::vector<int>(outputNames_.size(), 0),
                    false, outputFset);
}

// --------------------------------------------------------------------------------------

void TorchBalanceNonLocalEmulator::multiplyAD(oops::FieldSet3D & fset) const {
  multiplyAD(static_cast<const oops::FieldSet3D &>(fset), fset);
}

// --------------------------------------------------------------------------------------

void TorchBalanceNonLocalEmulator::multiplyAD(const oops::FieldSet3D & inputFset,
                                              oops::FieldSet3D & outputFset) const {
  atlas::FieldSet fsetCopy;
  for (const auto & field : inputFset.fieldSet()) {
    fsetCopy.add(field.clone());
  }

  // Log adjoint input fields before processing
  for (const auto & outputName : outputNames_) {
    if (fsetCopy.has(outputName)) {
      auto view = atlas::array::make_view<double, 2>(fsetCopy[outputName]);
      ValueStats adjointStats;
      for (atlas::idx_t jnode = 0; jnode < view.shape(0); ++jnode) {
        for (int level = 0; level < view.shape(1); ++level) {
          updateStats(view(jnode, level), adjointStats);
        }
      }
      oops::Log::debug() << "TorchBalance non-local multiplyAD: adjoint input " << outputName
                         << " " << statsSummary(adjointStats) << std::endl;
    }
  }

  for (const auto & outputName : outputNames_) {
    //fsetCopy[outputName].adjointHaloExchange();
    auto ghostView = atlas::array::make_view<int, 1>(geometryData_.functionSpace().ghost());
    auto outputView = atlas::array::make_view<double, 2>(fsetCopy[outputName]);
    for (atlas::idx_t jnode = 0; jnode < outputView.shape(0); ++jnode) {
      if (ghostView(jnode) == 0) continue;
      for (int level = 0; level < outputView.shape(1); ++level) {
        outputView(jnode, level) = 0.0;
      }
    }
  }
  oops::FieldSet3D packedFset(inputFset.validTime(), inputFset.commGeom());
  packedFset.shallowCopy(fsetCopy);
  std::vector<torch::Tensor> outputTensors = packFields(packedFset, outputNames_);

  c10::List<torch::Tensor> outputList;
  for (const auto & tensor : outputTensors) outputList.push_back(tensor);

  c10::List<torch::Tensor> geometryList;
  for (const auto & tensor : geometryTensors_) geometryList.push_back(tensor);

  std::vector<torch::jit::IValue> args{outputList, geometryList, mask_};
  c10::List<torch::Tensor> inputTensors;
  try {
    inputTensors = module_.get_method("multiply_ad")(args).toTensorList();
  } catch (const c10::Error & err) {
    throw eckit::Exception(
        "TorchBalance non-local TorchScript method multiply_ad(outputs_ad, geometry, mask) failed",
        Here());
  }

  // Log auxiliary tensor stats (geometry tensors followed by background tensors)
  oops::Log::debug() << "TorchBalance non-local multiplyAD: " << geometryTensors_.size()
                     << " auxiliary tensors being passed to multiply_ad"
                     << " (" << geometryTensorNames_.size() << " geometry + "
                     << backgroundTensorNames_.size() << " background)" << std::endl;
  for (size_t jgeo = 0; jgeo < geometryTensors_.size(); ++jgeo) {
    const bool isGeometry = (jgeo < geometryTensorNames_.size());
    const size_t localIdx = isGeometry ? jgeo : (jgeo - geometryTensorNames_.size());
    const std::string & tensorName = isGeometry
      ? geometryTensorNames_[localIdx]
      : backgroundTensorNames_[localIdx];
    oops::Log::debug() << "TorchBalance non-local multiplyAD: "
                       << (isGeometry ? "geometry" : "background")
                       << " tensor " << jgeo << " (" << tensorName << ")"
                       << " shape=" << geometryTensors_[jgeo].sizes()
                       << " " << statsSummary(tensorStats(geometryTensors_[jgeo]))
                       << std::endl;
  }

  oops::Log::debug() << "TorchBalance non-local multiplyAD: returned "
                     << inputTensors.size() << " tensors" << std::endl;
  for (size_t jfield = 0; jfield < inputNames_.size() &&
                          jfield < static_cast<size_t>(inputTensors.size()); ++jfield) {
    oops::Log::debug() << "TorchBalance non-local multiplyAD: output " << inputNames_[jfield]
                       << " " << statsSummary(tensorStats(inputTensors.get(jfield)))
                       << std::endl;
  }

  addReturnedFields(inputTensors, inputNames_, std::vector<int>(inputNames_.size(), 1),
                    true, outputFset);
}

// --------------------------------------------------------------------------------------

}  // namespace saber
