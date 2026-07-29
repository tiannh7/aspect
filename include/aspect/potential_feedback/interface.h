/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_potential_feedback_interface_h
#define _aspect_potential_feedback_interface_h

#include <aspect/global.h>
#include <aspect/potential_feedback/surface_history.h>

#include <deal.II/base/parameter_handler.h>

#include <set>
#include <string>
#include <vector>

namespace aspect
{
  namespace PotentialFeedback
  {
    /**
     * Marker base class for boundary traction plugins that represent
     * potential-feedback forcing. Legacy feedback implementations use this to
     * avoid treating the unified feedback adapter as an externally applied
     * load when reconstructing surface mass.
     */
    struct BoundaryTractionMarker
    {
      virtual ~BoundaryTractionMarker() = default;
    };



    /**
     * Degree-1 reference-frame convention for potential-feedback sources.
     */
    enum class DegreeOneReferenceFrame
    {
      none,
      geoid_cancellation,
      center_of_mass,
      citcomsve_center_of_mass
    };



    /** Reference state used for the prescribed ice load. */
    enum class IceLoadReference
    {
      first_history_file,
      zero_thickness,
      signed_anomaly
    };



    /** Parsed settings for the coupled GIA surface load. */
    struct GlacialIsostaticAdjustmentSettings
    {
      IceLoadReference ice_load_reference =
        IceLoadReference::first_history_file;

      SurfaceHistoryConfiguration ice_history;
      SurfaceHistoryConfiguration ocean_history;

      double density_ice = 917.4;
      double density_water = 1000.0;
      unsigned int maximum_degree = 32;
      std::vector<unsigned int> diagnostic_degrees = {2};
    };



    /**
     * Densities immediately above and below a density interface.
     */
    struct InterfaceDensity
    {
      double density_above = 0.0;
      double density_below = 0.0;

      double density_contrast() const;
    };



    /**
     * Shared density-interface properties used by all potential-feedback
     * mechanisms.
     */
    struct InterfaceProperties
    {
      InterfaceDensity surface;
      InterfaceDensity cmb;
    };



    /**
     * Parsed settings for the top-level Potential feedback hierarchy.
     */
    struct Settings
    {
      std::vector<std::string> feedback_mechanisms;

      InterfaceProperties interface_properties;

      std::set<types::boundary_id> active_feedback_boundary_indicators;
      bool include_surface_feedback = false;
      bool include_cmb_feedback = false;
      unsigned int self_gravity_max_degree = 32;

      double self_gravity_density_above_surface = 0.0;
      double self_gravity_density_below_surface = 0.0;
      double self_gravity_density_above_cmb = 0.0;
      double self_gravity_density_below_cmb = 0.0;
      std::string include_internal_density_anomalies = "auto";
      double reference_density_for_internal_anomalies = 0.0;
      double internal_density_anomaly_tolerance = 0.0;
      std::string full_domain_volume_source_discretization = "quadrature point";
      unsigned int full_domain_potential_radial_subdivisions = 32;
      std::string radial_transfer_scheme = "symmetric support projection";
      unsigned int maximum_target_enriched_radial_cache_supports = 100000;
      std::string surface_angular_analysis_scheme = "direct quadrature";
      bool use_adjoint_consistent_surface_potential_traction = false;

      std::string tidal_model_name = "none";
      unsigned int tidal_harmonic_degree = 2;
      unsigned int tidal_harmonic_order = 0;
      std::string tidal_coefficient_type = "cosine";
      std::string tidal_potential_quantity = "potential height";
      double tidal_potential_height_amplitude = 0.0;
      double tidal_potential_amplitude = 0.0;
      double tidal_reference_gravity = 1.0;
      std::string tidal_normalization = "geodesy 4pi";
      std::string tidal_time_dependence = "none";
      double tidal_angular_frequency = 0.0;
      double tidal_phase = 0.0;

      double rotational_fluid_love_number = 1.0;

      GlacialIsostaticAdjustmentSettings gia;

      double relative_tolerance = 0.0;
      unsigned int maximum_iterations = 0;
      bool freeze_feedback_after_timestep_zero = false;
      bool iterate_with_stokes = true;
      double potential_iteration_relaxation_factor = 1.0;
      // Stored in seconds after parsing, matching ASPECT's internal time units.
      double initial_displacement_timestep = 0.0;

      DegreeOneReferenceFrame degree_one_reference_frame =
        DegreeOneReferenceFrame::none;
      double center_of_mass_absolute_tolerance = 0.0;
      bool center_of_mass_correction = false;
      bool remove_pure_rotation_from_displacement = false;
      bool citcomsve_degree_one_load_compensation = false;

      bool has_active_mechanisms() const;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };
  }
}

#endif
