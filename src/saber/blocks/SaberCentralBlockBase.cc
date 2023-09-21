/*
 * (C) Copyright 2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/blocks/SaberCentralBlockBase.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"

#include <boost/noncopyable.hpp>

#include "eckit/exception/Exceptions.h"

#include "oops/base/GeometryData.h"
#include "oops/base/Variables.h"
#include "oops/util/AssociativeContainers.h"
#include "oops/util/FieldSetHelpers.h"
#include "oops/util/FieldSetOperations.h"
#include "oops/util/Logger.h"
#include "oops/util/parameters/OptionalParameter.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"
#include "oops/util/parameters/RequiredPolymorphicParameter.h"
#include "oops/util/Printable.h"

#include "saber/blocks/SaberBlockParametersBase.h"

namespace saber {

// -----------------------------------------------------------------------------

SaberCentralBlockFactory::SaberCentralBlockFactory(const std::string & name) {
  if (getMakers().find(name) != getMakers().end()) {
    oops::Log::error() << name << " already registered in saber::SaberCentralBlockFactory."
                       << std::endl;
    throw eckit::Exception("Element already registered in saber::SaberCentralBlockFactory.",
      Here());
  }
  getMakers()[name] = this;
}

// -----------------------------------------------------------------------------

std::unique_ptr<SaberCentralBlockBase> SaberCentralBlockFactory::create(
  const oops::GeometryData & geometryData,
  const oops::Variables & vars,
  const eckit::Configuration & covarConf,
  const SaberBlockParametersBase & params,
  const oops::FieldSet3D & xb,
  const oops::FieldSet3D & fg) {
  oops::Log::trace() << "SaberCentralBlockBase::create starting" << std::endl;
  const std::string id = params.saberBlockName;
  typename std::map<std::string, SaberCentralBlockFactory*>::iterator jsb = getMakers().find(id);
  if (jsb == getMakers().end()) {
    oops::Log::error() << id << " does not exist in saber::SaberCentralBlockFactory." << std::endl;
    throw eckit::UserError("Element does not exist in saber::SaberCentralBlockFactory.", Here());
  }
  std::unique_ptr<SaberCentralBlockBase> ptr =
    jsb->second->make(geometryData, vars, covarConf,
                      params, xb, fg);
  oops::Log::trace() << "SaberCentralBlockBase::create done" << std::endl;
  return ptr;
}

// -----------------------------------------------------------------------------

std::unique_ptr<SaberBlockParametersBase>
SaberCentralBlockFactory::createParameters(const std::string &name) {
  typename std::map<std::string, SaberCentralBlockFactory*>::iterator it =
      getMakers().find(name);
  if (it == getMakers().end()) {
    throw std::runtime_error(name + " does not exist in saber::SaberCentralBlockFactory");
  }
  return it->second->makeParameters();
}

// -----------------------------------------------------------------------------

void SaberCentralBlockBase::adjointTest(const oops::GeometryData & geometryData,
                                        const oops::Variables & vars,
                                        const double & adjointTolerance) const {
  oops::Log::trace() << "SaberCentralBlockBase::adjointTest starting" << std::endl;

  // Create random FieldSets
  atlas::FieldSet fset1 =  util::createRandomFieldSet(geometryData.comm(),
                                                      geometryData.functionSpace(),
                                                      vars);
  atlas::FieldSet fset2 =  util::createRandomFieldSet(geometryData.comm(),
                                                      geometryData.functionSpace(),
                                                      vars);

  // Copy FieldSets
  atlas::FieldSet fset1Save = util::copyFieldSet(fset1);
  atlas::FieldSet fset2Save = util::copyFieldSet(fset2);

  // Apply forward multiplication only (self-adjointness test)
  this->multiply(fset1);
  this->multiply(fset2);

  // Compute adjoint test
  const double dp1 = util::dotProductFieldSets(fset1,
                                               fset2Save,
                                               vars.variables(),
                                               geometryData.comm());
  const double dp2 = util::dotProductFieldSets(fset2,
                                               fset1Save,
                                               vars.variables(),
                                               geometryData.comm());
  oops::Log::info() << std::setprecision(16) << "Info     : Adjoint test: y^t (Ax) = " << dp1
                    << ": x^t (A^t y) = " << dp2 << " : adjoint tolerance = "
                    << adjointTolerance << std::endl;
  oops::Log::test() << "Adjoint test for block " << this->blockName();
  if (std::abs(dp1-dp2)/std::abs(0.5*(dp1+dp2)) < adjointTolerance) {
    oops::Log::test() << " passed" << std::endl;
  } else {
    oops::Log::test() << " failed" << std::endl;
    throw eckit::Exception("Adjoint test failure for block " + this->blockName(), Here());
  }

  oops::Log::trace() << "SaberCentralBlockBase::adjointTest done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
