/*
 * (C) Crown Copyright 2022 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/spectralb/SPNOINTERP_Cov.h"

#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"

#include "oops/base/Variables.h"
#include "oops/util/abor1_cpp.h"

#include "saber/oops/SaberCentralBlockBase.h"
#include "saber/oops/SaberCentralBlockParametersBase.h"
#include "saber/spectralb/spectralbnointerp.h"

namespace oops {
  class Variables;
}

namespace saber {
namespace spectralb {

// -----------------------------------------------------------------------------

static SaberCentralBlockMaker<SPNOINTERP_COV>  makerSPNOINTERP_COV_("SPNOINTERP_COV");

// -----------------------------------------------------------------------------

SPNOINTERP_COV::SPNOINTERP_COV(const oops::GeometryData & geometryData,
                               const std::vector<size_t> & activeVariableSizes,
                               const oops::Variables & inoutVars,
                               const Parameters_ & params,
                               const atlas::FieldSet & xb,
                               const atlas::FieldSet & fg,
                               const std::vector<atlas::FieldSet> & fsetVec)
  : spectralb_()
{
  oops::Log::trace() << classname() << "::SPNOINTERP_COV starting" << std::endl;

  // Get active variables
  oops::Variables activeVars = params.activeVars.value().get_value_or(inoutVars);

  // Initialize SpectralBNoInterp
  spectralb_.reset(new SpectralBNoInterp(activeVariableSizes,
                                         activeVars,
                                         params.spectralbParams.value()));

  oops::Log::trace() << classname() << "::SPNOINTERP_COV done" << std::endl;
}

// -----------------------------------------------------------------------------

SPNOINTERP_COV::~SPNOINTERP_COV() {
  oops::Log::trace() << classname() << "::~SPNOINTERP_COV starting" << std::endl;
  util::Timer timer(classname(), "~SPNOINTERP_COV");
  oops::Log::trace() << classname() << "::~SPNOINTERP_COV done" << std::endl;
}

// -----------------------------------------------------------------------------

void SPNOINTERP_COV::randomize(atlas::FieldSet & fset) const {
  oops::Log::trace() << classname() << "::randomize starting" << std::endl;
  ABORT("SPNOINTERP_COV::randomize: not implemented");
  oops::Log::trace() << classname() << "::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

void SPNOINTERP_COV::multiply(atlas::FieldSet & fset) const {
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;
  spectralb_->multiply(fset);
  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void SPNOINTERP_COV::print(std::ostream & os) const {
  os << classname();
}

// -----------------------------------------------------------------------------

}  // namespace spectralb
}  // namespace saber
