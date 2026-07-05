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

#include <deal.II/base/parameter_handler.h>

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
     * Shared planetary constants used by potential-feedback mechanisms.
     */
    struct PlanetaryConstants
    {
      std::string model_name;
      double planet_mass = 0.0;
      double planet_mean_density = 0.0;
      double polar_moment_of_inertia = 0.0;
      double equatorial_moment_of_inertia = 0.0;
      double rotation_rate = 0.0;
    };



    /**
     * Parsed settings for the top-level Potential feedback hierarchy.
     *
     * This structure is intentionally limited to shared configuration. The
     * first migration step keeps the existing feedback physics in the legacy
     * boundary traction plugins while establishing the new parameter surface.
     */
    struct Settings
    {
      std::vector<std::string> feedback_mechanisms;
      std::string interface_source;

      InterfaceProperties interface_properties;
      PlanetaryConstants planet;

      std::vector<std::string> self_gravity_source_interfaces;
      std::vector<std::string> self_gravity_apply_boundaries;
      unsigned int self_gravity_min_degree = 0;
      unsigned int self_gravity_max_degree = 0;

      double self_gravity_density_above_surface = 0.0;
      double self_gravity_density_below_surface = 0.0;
      double self_gravity_density_above_cmb = 0.0;
      double self_gravity_density_below_cmb = 0.0;
      std::string include_internal_density_anomalies = "auto";
      double reference_density_for_internal_anomalies = 0.0;
      double internal_density_anomaly_tolerance = 0.0;

      std::vector<std::string> rotational_inertia_source_interfaces;
      std::vector<std::string> rotational_apply_boundaries;
      unsigned int rotational_min_degree = 0;
      unsigned int rotational_max_degree = 0;

      double relative_tolerance = 0.0;
      unsigned int maximum_iterations = 0;
      bool freeze_feedback_after_timestep_zero = false;
      bool iterate_with_stokes = true;
      // Stored in seconds after parsing, matching ASPECT's internal time units.
      double initial_displacement_timestep = 0.0;

      bool center_of_mass_correction = false;
      bool remove_pure_rotation_from_displacement = false;
      bool citcomsve_degree_one_load_compensation = false;
      double citcomsve_degree_one_load_compensation_scale = 1.0;
      bool citcomsve_degree_one_cmb_final_rhs_override = false;
      double citcomsve_degree_one_cmb_final_rhs_value = 0.0;

      bool has_active_mechanisms() const;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };
  }
}

#endif
