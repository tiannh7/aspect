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

#ifndef _aspect_potential_feedback_glacial_isostatic_adjustment_h
#define _aspect_potential_feedback_glacial_isostatic_adjustment_h

#include <aspect/potential_feedback/interface.h>
#include <aspect/potential_feedback/surface_history.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace GIA
    {
      /** Compute the spatially uniform sea-level constant. */
      double barystatic_sea_level(
        const double ocean_area,
        const double ice_mass_change,
        const double relative_sea_level_volume,
        const double density_water);

      /** Evaluate the local canonical CitcomSVE relative sea-level change. */
      double sea_level_change(
        const double ocean_function,
        const double relative_geoid,
        const double barystatic_sea_level);

      /**
       * Convert an ice-history value to surface mass density relative to the
       * selected reference. Physical thickness modes clamp negative history
       * values to zero; signed-anomaly mode preserves both signs for
       * single-harmonic verification problems.
       */
      double ice_load_mass_density(
        const double current_ice_thickness,
        const double initial_ice_thickness,
        const double density_ice,
        const IceLoadReference reference);

      /**
       * Return the degree-l sea-level coefficient for a global ocean and a
       * single nonzero-degree spherical-harmonic ice-load coefficient.
       *
       * The relative sea-level response is the coefficient of N-U generated
       * per unit surface-mass coefficient. Consequently it has units of
       * meters divided by kg/m^2.
       */
      double uniform_ocean_sea_level_coefficient(
        const double relative_sea_level_response,
        const double ice_load_coefficient,
        const double density_water);
    }



    /**
     * Coupled ice, ocean, and sea-level-equation surface load for GIA.
     */
    template <int dim>
    class GlacialIsostaticAdjustment : public ::aspect::SimulatorAccess<dim>
    {
      public:
        /** Configure the component from the unified potential-feedback block. */
        void configure_from_potential_feedback_settings(
          const PotentialFeedback::Settings &settings);

        /** Set the provider for current surface potential height Phi/g. */
        void set_surface_potential_height_function(
          const std::function<double(const Point<dim> &)> &function);

        /** Initialize input histories, transforms, and update callbacks. */
        void initialize();

        /** Update histories and the GIA surface load at a time-step boundary. */
        void update();

        /** Recompute the load after other feedback mechanisms were updated. */
        void update_load_from_current_potential();

        /** Return the applied ice-plus-ocean traction. */
        Tensor<1,dim>
        boundary_traction(const types::boundary_id boundary_indicator,
                          const Point<dim> &position,
                          const Tensor<1,dim> &normal_vector) const;

        /** Return current total GIA surface mass per unit area. */
        double surface_mass_density(const Point<dim> &position) const;

        /** Return current prescribed ice-load mass per unit area. */
        double ice_load_mass_density(const Point<dim> &position) const;

        /** Return current ocean-load mass per unit area. */
        double ocean_load_mass_density(const Point<dim> &position) const;

        /** Return current relative sea-level/ocean-height change. */
        double sea_level_change(const Point<dim> &position) const;

        /** Return the current (spectrally represented) ocean function. */
        double ocean_function(const Point<dim> &position) const;

        /** Return a total surface-load spherical-harmonic coefficient pair. */
        std::pair<double,double>
        total_load_coefficient(const unsigned int degree,
                               const unsigned int order) const;

        /** Return an ice-load spherical-harmonic coefficient pair. */
        std::pair<double,double>
        ice_load_coefficient(const unsigned int degree,
                             const unsigned int order) const;

        /** Return an ocean-load spherical-harmonic coefficient pair. */
        std::pair<double,double>
        ocean_load_coefficient(const unsigned int degree,
                               const unsigned int order) const;

        /** Return a relative-sea-level spherical-harmonic coefficient pair. */
        std::pair<double,double>
        sea_level_coefficient(const unsigned int degree,
                              const unsigned int order) const;

        /** Return the spatially uniform barystatic constant in meters. */
        double barystatic_sea_level() const;

        /** Return the eustatic ice-volume contribution in meters. */
        double eustatic_sea_level() const;

        /** Return the prescribed ice-mass change integrated over the surface. */
        double ice_mass_change() const;

        /** Return the applied ocean-water mass change integrated over the surface. */
        double ocean_water_mass_change() const;

        /** Return the applied ice-plus-ocean mass residual. */
        double water_mass_residual() const;

        /** Return the applied ice-plus-ocean mass residual relative to load mass. */
        double relative_water_mass_residual() const;

        /** Return whether the GIA load fixed-point iteration has converged. */
        bool potential_is_converged() const;

        /** Return the latest relative surface-load coefficient change. */
        double potential_relative_change_value() const;

        /** Return whether the component is enabled. */
        bool is_enabled() const;

        /** Serialize checkpointed GIA state. */
        template <class Archive>
        void serialize(Archive &ar, const unsigned int version);

      private:
        void compute_surface_load(const bool include_current_velocity_increment);

        void update_after_stokes_solve();

        double synthesize(const std::vector<double> &cos_coefficients,
                          const std::vector<double> &sin_coefficients,
                          const Point<dim> &position) const;

        std::pair<double,double>
        coefficient(const std::vector<double> &cos_coefficients,
                    const std::vector<double> &sin_coefficients,
                    const unsigned int degree,
                    const unsigned int order) const;

        void relax_coefficients(std::vector<double> &stored_cos,
                                std::vector<double> &stored_sin,
                                const std::vector<double> &new_cos,
                                const std::vector<double> &new_sin);

        double coefficient_relative_change(
          const std::vector<double> &old_cos,
          const std::vector<double> &old_sin,
          const std::vector<double> &new_cos,
          const std::vector<double> &new_sin) const;

        double coefficient_absolute_change(
          const std::vector<double> &old_cos,
          const std::vector<double> &old_sin,
          const std::vector<double> &new_cos,
          const std::vector<double> &new_sin) const;

        bool enabled = false;
        bool iterate_with_stokes = true;
        bool freeze_feedback_after_timestep_zero = false;
        IceLoadReference ice_load_reference =
          IceLoadReference::first_history_file;

        double density_ice = 917.4;
        double density_water = 1000.0;
        double initial_displacement_timestep = 0.0;
        double potential_convergence_tolerance = 1e-3;
        double absolute_coefficient_tolerance = 1e-4;
        double potential_relaxation_factor = 1.0;
        unsigned int maximum_degree = 32;
        std::vector<unsigned int> diagnostic_degrees = {2};
        unsigned int projection_longitude_samples = 0;
        unsigned int projection_latitude_samples = 0;
        unsigned int maximum_potential_iterations = 20;

        SurfaceHistoryConfiguration ice_history_configuration;
        SurfaceHistoryConfiguration ocean_history_configuration;

        types::boundary_id top_boundary_id = numbers::invalid_boundary_id;
        SurfaceHistory<dim> ice_history;
        SurfaceHistory<dim> ocean_history;
        std::unique_ptr<Utilities::SphericalHarmonicTransform> sh_transform;

        std::function<double(const Point<dim> &)>
        surface_potential_height_function;

        std::vector<double> total_load_cos_coefficients;
        std::vector<double> total_load_sin_coefficients;
        std::vector<double> ice_load_cos_coefficients;
        std::vector<double> ice_load_sin_coefficients;
        std::vector<double> ocean_load_cos_coefficients;
        std::vector<double> ocean_load_sin_coefficients;
        std::vector<double> sea_level_cos_coefficients;
        std::vector<double> sea_level_sin_coefficients;
        std::vector<double> ocean_function_cos_coefficients;
        std::vector<double> ocean_function_sin_coefficients;

        double current_barystatic_sea_level = 0.0;
        double current_eustatic_sea_level = 0.0;
        double current_ice_mass_change = 0.0;
        double current_ocean_area = 0.0;
        double current_ocean_water_mass_change = 0.0;
        double current_water_mass_residual = 0.0;
        double current_relative_water_mass_residual = 0.0;
        double potential_relative_change =
          std::numeric_limits<double>::infinity();
        double potential_absolute_change =
          std::numeric_limits<double>::infinity();
        unsigned int current_potential_iteration_step =
          numbers::invalid_unsigned_int;
        unsigned int potential_iteration_number = 0;
    };



    template <int dim>
    template <class Archive>
    void
    GlacialIsostaticAdjustment<dim>::serialize(Archive &ar,
                                               const unsigned int)
    {
      ar &total_load_cos_coefficients
      & total_load_sin_coefficients
      & ice_load_cos_coefficients
      & ice_load_sin_coefficients
      & ocean_load_cos_coefficients
      & ocean_load_sin_coefficients
      & sea_level_cos_coefficients
      & sea_level_sin_coefficients
      & ocean_function_cos_coefficients
      & ocean_function_sin_coefficients
      & current_barystatic_sea_level
      & current_eustatic_sea_level
      & current_ice_mass_change
      & current_ocean_area
      & current_ocean_water_mass_change
      & current_water_mass_residual
      & current_relative_water_mass_residual
      & potential_relative_change
      & potential_absolute_change
      & current_potential_iteration_step
      & potential_iteration_number;
    }
  }
}

#endif
