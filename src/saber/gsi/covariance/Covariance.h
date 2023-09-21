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

#include "oops/base/GeometryData.h"
#include "oops/base/Variables.h"
#include "oops/util/parameters/OptionalParameter.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"

#include "saber/blocks/SaberBlockParametersBase.h"
#include "saber/blocks/SaberCentralBlockBase.h"
#include "saber/gsi/covariance/Covariance.interface.h"
#include "saber/gsi/utils/GSIParameters.h"

namespace oops {
  class Variables;
}

namespace saber {
namespace gsi {

// -------------------------------------------------------------------------------------------------

class CovarianceParameters : public SaberBlockParametersBase {
  OOPS_CONCRETE_PARAMETERS(CovarianceParameters, SaberBlockParametersBase)

 public:
  // File containing grid and coefficients
  oops::OptionalParameter<GSIParameters> readParams{"read", this};

  // Mandatory active variables
  oops::Variables mandatoryActiveVars() const override {return oops::Variables();}
};

// -------------------------------------------------------------------------------------------------

class Covariance : public SaberCentralBlockBase {
 public:
  static const std::string classname() {return "saber::gsi::Covariance";}

  typedef CovarianceParameters Parameters_;

  Covariance(const oops::GeometryData &,
             const oops::Variables &,
             const eckit::Configuration &,
             const Parameters_ &,
             const oops::FieldSet3D &,
             const oops::FieldSet3D &);
  virtual ~Covariance();

  void randomize(atlas::FieldSet &) const override;
  void multiply(atlas::FieldSet &) const override;

  void read() override;

 private:
  void print(std::ostream &) const override;

  // Fortran LinkedList key
  CovarianceKey keySelf_;
  // Parameters
  Parameters_ params_;
  // Variables
  std::vector<std::string> variables_;
  // GSI grid FunctionSpace
  atlas::FunctionSpace gsiGridFuncSpace_;
  // Communicator
  const eckit::mpi::Comm * comm_;
  // Background
  atlas::FieldSet xb_;
  // First guess
  atlas::FieldSet fg_;
  // Valid time
  util::DateTime validTimeOfXbFg_;
};

// -------------------------------------------------------------------------------------------------

}  // namespace gsi
}  // namespace saber
