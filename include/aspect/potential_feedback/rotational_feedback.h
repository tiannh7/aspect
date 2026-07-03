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

#ifndef _aspect_potential_feedback_rotational_feedback_h
#define _aspect_potential_feedback_rotational_feedback_h

#include <aspect/boundary_traction/interface.h>
#include <aspect/potential_feedback/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <memory>
#include <vector>

namespace aspect
{
  namespace PotentialFeedback
  {
    /**
     * Boundary traction plugin for the rotational-potential feedback associated
     * with polar wander in degree-2 order-1 surface-loading benchmarks.
     *
     * The plugin computes the products-of-inertia perturbations from the
     * effective boundary mass, converts them to a small perturbation of the
     * rotation vector, and applies the induced centrifugal-potential
     * perturbation as an additional normal traction. It is intended as a
     * benchmark-scale spherical-shell capability, not as a full GIA sea-level
     * or rotational-Liouville implementation.
     *
     * @ingroup BoundaryTractions
     */
    template <int dim>
    class RotationalFeedback : public BoundaryTraction::Interface<dim>,
      public ::aspect::SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        void update() override;

        Tensor<1,dim>
        boundary_traction(const types::boundary_id boundary_indicator,
                          const Point<dim> &position,
                          const Tensor<1,dim> &normal_vector) const override;

        /**
         * Return whether the stored rotational potential changed by less than
         * the configured relative tolerance in the last feedback update.
         */
        bool potential_is_converged() const;

        double potential_relative_change_value() const;

        void configure_from_potential_feedback_settings(
          const PotentialFeedback::Settings &settings);

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

      private:
        void compute_rotational_feedback(const bool include_current_velocity_increment);

        void update_after_stokes_solve();

        bool enabled;
        bool include_cmb_contribution;
        bool iterate_with_stokes;
        bool enable_surface_potential_traction;
        bool enable_cmb_potential_traction;
        bool apply_center_of_mass_correction;

        unsigned int max_degree;
        unsigned int min_degree;

        double density_above_surface;
        double density_below_surface;
        double density_above_cmb;
        double density_below_cmb;
        double planet_mass;
        double polar_moment_of_inertia;
        double equatorial_moment_of_inertia;
        double rotation_rate;
        double initial_displacement_timestep;
        double potential_convergence_tolerance;
        double potential_relative_change;
        unsigned int maximum_potential_iterations;
        unsigned int current_potential_iteration_step;
        unsigned int potential_iteration_number;

        mutable double delta_ixz;
        mutable double delta_iyz;
        mutable Tensor<1,3> center_of_mass_shift;
        mutable Tensor<1,3> rotation_vector_perturbation;

        types::boundary_id top_boundary_id;
        types::boundary_id bottom_boundary_id;

        std::unique_ptr<Utilities::SphericalHarmonicTransform> sh_transform;

        std::vector<double> surface_potential_cos_coeffs;
        std::vector<double> surface_potential_sin_coeffs;
        std::vector<double> cmb_potential_cos_coeffs;
        std::vector<double> cmb_potential_sin_coeffs;
    };
  }
}

#endif
