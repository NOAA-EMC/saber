/*
 * (C) Copyright 2022 United States Government as represented by the Administrator of the National
 *     Aeronautics and Space Administration
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"
#include "atlas/functionspace.h"
#include "atlas/grid.h"
#include "atlas/library.h"
#include "atlas/runtime/Log.h"

#include "oops/base/Geometry.h"
#include "oops/base/State.h"
#include "oops/base/Variables.h"
#include "oops/util/abor1_cpp.h"

#include "saber/gsi/grid/GSI_Grid.h"
#include "saber/gsi/interpolation/unstructured_interp/UnstructuredInterpolation.h"

using atlas::option::levels;
using atlas::option::name;

namespace oops {
  class Variables;
}

namespace saber {
namespace gsi {

// -------------------------------------------------------------------------------------------------

class InterpolationImplParameters : public GridParameters {
  OOPS_CONCRETE_PARAMETERS(InterpolationImplParameters, GridParameters)

 public:
  // Interpolation method
  oops::Parameter<std::string> interpMethod{"interpolation method", "barycent", this};
};

// -------------------------------------------------------------------------------------------------

class InterpolationImpl {
 public:
  static const std::string classname() {return "saber::gsi::InterpolationImpl";}

  typedef atlas::functionspace::PointCloud PointCloud_;
  typedef eckit::mpi::Comm Comm_;
  typedef InterpolationImplParameters Parameters_;

  InterpolationImpl(const Comm_ &, const PointCloud_ &, const Parameters_ &,
                    const std::vector<std::string>);
  ~InterpolationImpl();
  void multiply(atlas::FieldSet &) const;
  void multiplyAD(atlas::FieldSet &) const;

 private:
  void print(std::ostream &) const;
  // Interpolation object
  std::unique_ptr<UnstructuredInterpolation> interpolator_;
  // Function spaces
  atlas::FunctionSpace gsiGridFuncSpace_;
  atlas::FunctionSpace modGridFuncSpace_;
  // Variables
  std::vector<std::string> variables_;
  // Expected number of levels in GSI grid
  int gsiLevels_;
  // Grid
  Grid grid_;
};

// -------------------------------------------------------------------------------------------------

}  // namespace gsi
}  // namespace saber
