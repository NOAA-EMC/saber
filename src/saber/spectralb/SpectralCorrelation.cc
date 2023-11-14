/*
 * (C) Crown Copyright 2023 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <cmath>

#include "saber/spectralb/SpectralCorrelation.h"

#include "atlas/array/MakeView.h"
#include "atlas/field/Field.h"
#include "atlas/field/FieldSet.h"

#include "eckit/exception/Exceptions.h"

#include "oops/mpi/mpi.h"
#include "oops/util/AtlasArrayUtil.h"
#include "oops/util/FieldSetHelpers.h"
#include "oops/util/FieldSetOperations.h"
#include "oops/util/Logger.h"

#include "saber/oops/Utilities.h"
#include "saber/spectralb/CovarianceStatisticsUtils.h"

// -----------------------------------------------------------------------------

namespace saber {
namespace spectralb {

// -----------------------------------------------------------------------------

static SaberCentralBlockMaker<SpectralCorrelation> makerSpectralCorrelation_(
  "spectral correlation");

// -----------------------------------------------------------------------------
SpectralCorrelation::SpectralCorrelation(const oops::GeometryData & geometryData,
                                         const oops::Variables & centralVars,
                                         const eckit::Configuration & covarConf,
                                         const Parameters_ & params,
                                         const oops::FieldSet3D & xb,
                                         const oops::FieldSet3D & fg)
  : SaberCentralBlockBase(params), params_(params),
    activeVars_(getActiveVars(params, centralVars)),
    spectralVerticalCorrelations_(),
    geometryData_(geometryData),
    specFunctionSpace_(geometryData_.functionSpace())
{
  oops::Log::trace() << classname() << "::SpectralCorrelation starting " << std::endl;

  oops::Log::trace() << classname() << "::SpectralCorrelation done" << std::endl;
}

// -----------------------------------------------------------------------------

void SpectralCorrelation::randomize(atlas::FieldSet & fieldSet) const {
  oops::Log::trace() << classname() << "::randomize starting" << std::endl;

  oops::Log::error() << "randomization with spectral correlation saber block"
                     << " is not supported. Instead please use 'ID' central block"
                     << " and 'square root of spectral correlation' outer block."
                     << std::endl;
  throw(eckit::FunctionalityNotSupported(
        "use ID and square root of spectral correlation instead.", Here()));
  oops::Log::trace() << classname() << "::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

void SpectralCorrelation::multiply(atlas::FieldSet & fieldSet) const {
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;
  // TO DO(MW): Put common code from this method and that of Spectral Covariance
  //            in common function call.

  using atlas::array::make_view;
  using atlas::idx_t;

  const idx_t N = specFunctionSpace_.truncation();

  // This is the same as Spectral Covariance (so could be put into a function)

  std::vector<std::string> vertCovNames = spectralVerticalCorrelations_.field_names();

  const auto zonal_wavenumbers = specFunctionSpace_.zonal_wavenumbers();
  const idx_t nb_zonal_wavenumbers = zonal_wavenumbers.size();

  // Only update the fields that were specified in the active variables
  for (const auto & var : activeVars_.variables()) {
    idx_t i = 0;
    idx_t levels(fieldSet[var].levels());
    auto vertCovView = make_view<const double, 3>(spectralVerticalCorrelations_[var]);
    auto spfView = make_view<double, 2>(fieldSet[var]);

    std::vector<double> col(levels), col2(levels);
    // For each total wavenumber n1, perform a 1D convolution with vertical covariances.
    for (idx_t jm = 0; jm < nb_zonal_wavenumbers; ++jm) {
      const idx_t m1 = zonal_wavenumbers(jm);
      for (std::size_t n1 = m1; n1 <= static_cast<std::size_t>(N); ++n1) {
        // Note that img stands for imaginary component and are the
        // odd indices in the first index of the spectral fields.
        for (std::size_t img = 0; img < 2; ++img) {
          // Pre-fill vertical column to be convolved.
          for (idx_t jl = 0; jl < levels; ++jl) {
            col[static_cast<std::size_t>(jl)] = spfView(i, jl);
          }
          // The 2*n1+1 factor is there to equally distribute the covariance across
          // the spectral coefficients associated to this total wavenumber.
          const double norm = static_cast<double>((2 * n1 + 1) * vertCovView.shape(0));
          for (idx_t r = 0; r < levels; ++r) {
            col2[static_cast<std::size_t>(r)] = 0;
            for (idx_t c = 0; c < levels; ++c) {
              col2[static_cast<std::size_t>(r)] += vertCovView(n1, r, c) * col[c] / norm;
            }
          }
          for  (idx_t jl = 0; jl < levels; ++jl) {
            spfView(i, jl) = col2[static_cast<std::size_t>(jl)];
          }
          ++i;
        }
      }
    }
  }

  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void SpectralCorrelation::read() {
  oops::Log::trace() << classname() << "::read starting" << std::endl;
  // TO DO(MW): Put common code from this method and that of Spectral Covariance
  //            in common function call.

  // Note that the read can occur in either calibration mode
  // using calibrationReadParams or in standard covariance mode using
  // readParams
  spectralbReadVertCovParameters sparams;
  const auto & calibparams = params_.calibrationParams.value();
  if (calibparams != boost::none) {
    const auto & calibrationReadParams = calibparams->calibrationReadParams.value();
    if (calibrationReadParams != boost::none) {
       sparams = calibrationReadParams.value();
    }
  } else {
    sparams = *params_.readParams.value();
  }

  const auto & umatrixNetCDFParams = sparams.umatrixNetCDFNames.value();

  const int nSpectralBins = specFunctionSpace_.truncation() + 1;  // 2N
  atlas::FieldSet spectralVerticalCovariances;

  for (std::size_t i = 0; i < activeVars_.variables().size(); ++i) {
    //  allocate vert cov field based on activeVars and spectralfunctionspace_
    auto spectralVertCov =
      atlas::Field(activeVars_[i],
                   atlas::array::make_datatype<double>(),
                   atlas::array::make_shape(nSpectralBins,
                                            activeVars_.getLevels(activeVars_[i]),
                                            activeVars_.getLevels(activeVars_[i])));
    if (umatrixNetCDFParams != boost::none) {
      const oops::Variables netCDFVars(umatrixNetCDFParams.value());
      createSpectralCovarianceFromUMatrixFile(activeVars_[i],
                                              netCDFVars[i],
                                              sparams,
                                              spectralVertCov);

    } else {
      readSpectralCovarianceFromFile(activeVars_[i],
                                     sparams,
                                     spectralVertCov);
    }

    spectralVerticalCovariances.add(spectralVertCov);
  }

  spectralVerticalCorrelations_.clear();
  atlas::FieldSet sds = createVerticalSD(activeVars_, spectralVerticalCovariances);
  spectralVerticalCorrelations_ =
    createSpectralCorrelations(activeVars_, spectralVerticalCovariances, sds);

  oops::Log::trace() << classname() << "::read done" << std::endl;
}


void SpectralCorrelation::directCalibration(const std::vector<atlas::FieldSet> &
                                            MOSpectralCovariancesEns) {
  oops::Log::trace() << classname() << "::directCalibration starting" << std::endl;

  oops::Log::error() << "directCalibration with spectral correlation saber block"
                     << " is not supported. Instead please use 'spectral covariance"
                     << " central block."
                     << std::endl;
  throw(eckit::FunctionalityNotSupported(
        "use spectral covariance central block instead.", Here()));

  oops::Log::trace() << classname() << "::directCalibration done" << std::endl;
}

void SpectralCorrelation::write() const {
  oops::Log::trace() << classname() << "::write starting" << std::endl;
  // TO DO(MW): Put common code from this method and that of Spectral Covariance
  //            in common function call.

  spectralbCalibrationWriteParameters writeParams =
    (params_.calibrationParams.value())->writeParams.value();
  eckit::LocalConfiguration writeConfig;
  writeParams.serialize(writeConfig);

  const std::string filepath{writeParams.filePath};
  const std::string mpi_pattern = writeParams.mpiPattern;
  const std::string mpi_size = std::to_string(oops::mpi::world().size());
  ::util::seekAndReplace(writeConfig, mpi_pattern, mpi_size);
  const std::string ncfilepath = "./" + writeConfig.getString("file path");

  using atlas::array::make_view;
  using atlas::idx_t;

  // The spectralVerticalCovariances that we write should be a gathered version of
  // the one in memory  ... it should not affect the internal version.
  atlas::FieldSet spectralVertCovToWrite;
  copySpectralFieldSet(spectralVerticalCorrelations_, spectralVertCovToWrite);

  // gather and sum on pe 0
  const std::size_t root(0);
  gatherSumSpectralFieldSet(geometryData_.comm(),
                            root,
                            spectralVertCovToWrite);

  const std::vector<std::string> dim_names{"total wavenumber",
                                           "model levels 1",
                                           "model levels 2"};
  const std::vector<atlas::idx_t> dim_sizes{spectralVertCovToWrite[0].shape()[0],
                                            spectralVertCovToWrite[0].shape()[1],
                                            spectralVertCovToWrite[0].shape()[2]};
  std::vector<std::string> field_names(activeVars_.variables());
  std::vector<std::vector<std::string>> dim_names_for_every_var;
  for (auto & field : field_names) {
    field.append(" spectral vertical correlation");
    dim_names_for_every_var.push_back(dim_names);
  }

  std::vector<int> netcdf_general_ids;
  std::vector<int> netcdf_dim_ids;
  std::vector<int> netcdf_var_ids;
  std::vector<std::vector<int>> netcdf_dim_var_ids;

  if (oops::mpi::world().rank() == root) {
    ::util::atlasArrayWriteHeader(ncfilepath,
                                  dim_names,
                                  dim_sizes,
                                  field_names,
                                  dim_names_for_every_var,
                                  netcdf_general_ids,
                                  netcdf_dim_ids,
                                  netcdf_var_ids,
                                  netcdf_dim_var_ids);

    std::size_t t(0);
    for (const atlas::Field & fld : spectralVertCovToWrite) {
      auto fview = atlas::array::make_view<const double, 3>(fld);
      ::util::atlasArrayWriteData(netcdf_general_ids,
                                  netcdf_var_ids[t],
                                  fview);
      ++t;
    }
  }

  oops::Log::trace() << classname() << "::write done" << std::endl;
}

// -----------------------------------------------------------------------------

void SpectralCorrelation::print(std::ostream & os) const {
  os << classname();
}

}  // namespace spectralb
}  // namespace saber
