/*
 * (C) Copyright 2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/oops/ID.h"

#include <vector>

#include "oops/util/Logger.h"

namespace saber {
namespace generic {

// -----------------------------------------------------------------------------

static SaberCentralBlockMaker<ID> makerID_("ID");

// -----------------------------------------------------------------------------

ID::ID(const oops::GeometryData & geometryData,
       const std::vector<size_t> & activeVariableSizes,
       const oops::Variables & activeVars,
       const Parameters_ & params,
       const atlas::FieldSet & xb,
       const atlas::FieldSet & fg,
       const std::vector<atlas::FieldSet> & fsetVec)
{
  oops::Log::trace() << classname() << "::ID starting" << std::endl;
  oops::Log::trace() << classname() << "::ID done" << std::endl;
}

// -----------------------------------------------------------------------------

void ID::randomize(atlas::FieldSet & fset) const {
  oops::Log::trace() << classname() << "::randomize starting" << std::endl;
  oops::Log::trace() << classname() << "::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

void ID::multiply(atlas::FieldSet & fset) const {
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;
  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void ID::print(std::ostream & os) const {
  os << classname();
}

// -----------------------------------------------------------------------------

}  // namespace generic
}  // namespace saber
