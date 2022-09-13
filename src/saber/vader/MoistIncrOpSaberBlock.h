/*
 * (C) Crown Copyright 2022 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef SABER_VADER_MOISTINCROPSABERBLOCK_H_
#define SABER_VADER_MOISTINCROPSABERBLOCK_H_

#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"
#include "atlas/functionspace.h"

#include "eckit/exception/Exceptions.h"

#include "oops/base/Variables.h"

#include "saber/oops/SaberBlockBase.h"
#include "saber/oops/SaberBlockParametersBase.h"

namespace oops {
  class Variables;
}

namespace saber {

// -----------------------------------------------------------------------------

class MoistIncrOpSaberBlockParameters : public SaberBlockParametersBase {
  OOPS_CONCRETE_PARAMETERS(MoistIncrOpSaberBlockParameters, SaberBlockParametersBase)
 public:
};

// -----------------------------------------------------------------------------

class MoistIncrOpSaberBlock : public SaberBlockBase {
 public:
  static const std::string classname() {return "saber::AirTemperatureSaberBlock";}

  typedef MoistIncrOpSaberBlockParameters Parameters_;

  MoistIncrOpSaberBlock(const eckit::mpi::Comm &,
                        const atlas::FunctionSpace &,
                        const atlas::FieldSet &,
                        const std::vector<size_t> &,
                        const Parameters_ &,
                        const atlas::FieldSet &,
                        const atlas::FieldSet &,
                        const std::vector<atlas::FieldSet> &);
  virtual ~MoistIncrOpSaberBlock();

  void randomize(atlas::FieldSet &) const override;
  void multiply(atlas::FieldSet &) const override;
  void inverseMultiply(atlas::FieldSet &) const override;
  void multiplyAD(atlas::FieldSet &) const override;
  void inverseMultiplyAD(atlas::FieldSet &) const override;

 private:
  void print(std::ostream &) const override;
  atlas::FieldSet augmentedStateFieldSet_;
};

// -----------------------------------------------------------------------------

}  // namespace saber

#endif  // SABER_VADER_MOISTINCROPSABERBLOCK_H_
