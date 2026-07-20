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

#include <functional>
#include <memory>
#include <string>
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
     * effective boundary mass and, when selected, the same internal mechanical
     * density sources used by self-gravity. It converts them to a small
     * perturbation of the rotation vector and applies the induced
     * centrifugal-potential perturbation as an additional normal traction. It
     * is intended as a quasi-static spherical-shell capability. When combined
     * with the GIA mechanism it supplies the physical polar-wander feedback
     * used by the Zhong and Yuan benchmarks, but it is not a general
     * time-dependent rotational-Liouville solver.
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

        /** Return the current rotational potential height Phi/g. */
        double potential_height(const types::boundary_id boundary_indicator,
                                const Point<dim> &position) const;

        /** Return the current rotational scalar potential Phi at an
         * arbitrary volume point. */
        double full_domain_potential(const Point<dim> &position) const;

        /** Add a load owned by the unified potential-feedback adapter. */
        void set_additional_load_traction_function(
          const std::function<Tensor<1,dim>(const types::boundary_id,
                                            const Point<dim> &,
                                            const Tensor<1,dim> &)> &function);

        /** Supply the final self-gravity surface Phi/g coefficients. When
         * available, rotational feedback uses the shared degree-2/order-1
         * mass potential as its forcing source and retains its independent
         * inertia integration as a diagnostic. */
        void set_self_gravity_surface_potential_coefficient_function(
          const std::function<std::pair<double,double>(const unsigned int,
                                                       const unsigned int)> &function);

        /**
         * Return whether the stored rotational potential changed by less than
         * the configured relative tolerance in the last feedback update.
         */
        bool potential_is_converged() const;

        double potential_relative_change_value() const;

        std::pair<double,double>
        surface_potential_coefficient(const unsigned int degree,
                                      const unsigned int order) const;

        void configure_from_potential_feedback_settings(
          const PotentialFeedback::Settings &settings);

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

      private:
        void compute_rotational_feedback(const bool include_current_velocity_increment);

        void update_after_stokes_solve();

        bool enabled;
        bool include_surface_contribution;
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
        std::string include_internal_density_anomalies = "false";
        double reference_density_for_internal_anomalies = 0.0;
        double fluid_love_number;
        double initial_displacement_timestep;
        double potential_convergence_tolerance;
        double potential_relative_change;
        double potential_source_relative_difference;
        unsigned int maximum_potential_iterations;
        unsigned int current_potential_iteration_step;
        unsigned int potential_iteration_number;

        mutable double delta_ixz;
        mutable double delta_iyz;
        mutable double rotational_potential_prefactor;

        types::boundary_id top_boundary_id;
        types::boundary_id bottom_boundary_id;

        std::unique_ptr<Utilities::SphericalHarmonicTransform> sh_transform;

        std::vector<double> surface_potential_cos_coeffs;
        std::vector<double> surface_potential_sin_coeffs;
        std::vector<double> cmb_potential_cos_coeffs;
        std::vector<double> cmb_potential_sin_coeffs;

        std::function<Tensor<1,dim>(const types::boundary_id,
                                    const Point<dim> &,
                                    const Tensor<1,dim> &)>
        additional_load_traction_function;

        std::function<std::pair<double,double>(const unsigned int,
                                               const unsigned int)>
        self_gravity_surface_potential_coefficient_function;
    };
  }
}

#endif
