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

#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/simulator.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace PotentialFeedback
  {
    template <>
    void
    SelfGravitation<3>::initialize()
    {
      top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      update_derived_planetary_constants();

      if (degree_one_reference_frame ==
          DegreeOneReferenceFrame::center_of_mass)
        {
          const auto &parameters = this->get_parameters();
          AssertThrow(parameters.density_source_law
                      == Parameters<3>::Formulation::DensitySourceLaw::
                      mechanical_mass_conservation,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "the mechanical mass-conservation density source "
                        "law, so that its constraint and the Stokes forcing "
                        "use the same discrete mass source."));
          AssertThrow(self_gravity_mass_feedback_enabled,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "self gravity in Potential feedback/List of feedback "
                        "mechanisms."));
          AssertThrow(include_surface_contribution,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "surface mass feedback."));

          const auto &boundary_velocity_manager =
            this->get_boundary_velocity_manager();
          const bool bottom_normal_velocity_is_zero =
            boundary_velocity_manager
            .get_zero_boundary_velocity_indicators().count(bottom_boundary_id)
            > 0
            || boundary_velocity_manager
            .get_tangential_boundary_velocity_indicators()
            .count(bottom_boundary_id) > 0;
          AssertThrow(include_cmb_contribution
                      || bottom_normal_velocity_is_zero,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "CMB mass feedback unless the bottom boundary has zero "
                        "normal velocity."));
          AssertThrow(include_internal_density_anomalies != "false",
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "internal density anomalies so that volume mass and "
                        "displaced internal interfaces are included in the "
                        "COM constraint."));
          AssertThrow(iterate_with_stokes,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "Potential feedback/Potential iteration/Iterate with "
                        "Stokes = true."));
          AssertThrow(!freeze_potential_after_timestep_zero,
                      ExcMessage(
                        "The coupled center-of-mass reference frame cannot "
                        "freeze its potential after timestep zero because the "
                        "mass-dipole constraint must follow the evolving "
                        "deformation."));
          AssertThrow(potential_iteration_relaxation_factor > 0.0
                      && potential_iteration_relaxation_factor <= 1.0,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "a potential-iteration relaxation factor in (0,1]."));
          AssertThrow(!(parameters.nullspace_removal
                        & Parameters<3>::NullspaceRemoval::postsolve_translation),
                      ExcMessage(
                        "Do not combine the coupled center-of-mass reference "
                        "frame with net-translation or linear-momentum "
                        "nullspace removal because those options project the "
                        "velocity after the solve."));

          // The full-domain potential and the COM dipole must include the same
          // internal mechanical source on the first pre-Stokes update. At each
          // post-Stokes update the iteration counter is reset and incremented
          // before source evaluation, so it is again positive.
          potential_iteration_number = 1;
        }

      const double mm_initial_elastic_dt =
        this->get_material_model().initial_elastic_time_step();
      if (mm_initial_elastic_dt > 0.0 && initial_displacement_timestep == 0.0)
        initial_displacement_timestep = mm_initial_elastic_dt;

      if (configured_from_potential_feedback)
        {
          enable_surface_potential_traction = include_surface_contribution;
          enable_cmb_potential_traction = include_cmb_contribution;
        }

      last_text_output_time = -1.0;
      last_text_output_step = 0;
      current_tracked_step = (unsigned int)-1;
      printing_this_step = true;

      sh_transform = std::make_unique<Utilities::SphericalHarmonicTransform>(
                       max_degree, min_degree);

      if (iterate_with_stokes)
        this->get_signals().post_stokes_solver.connect(
          [this](const SimulatorAccess<3> &,
                 const unsigned int,
                 const unsigned int,
                 const SolverControl &,
                 const SolverControl &)
        {
          this->update_after_stokes_solve();
        });

      this->get_signals().post_mesh_deformation.connect(
        [this](const SimulatorAccess<3> &)
      {
        this->compute_self_gravity_correction(false);
      });
    }



    template <>
    double
    SelfGravitation<3>::compute_reference_planet_mass(
      const double inner_radius,
      const double outer_radius) const
    {
      AssertThrow(outer_radius > inner_radius && inner_radius >= 0.0,
                  ExcMessage("The reference planet mass requires valid inner "
                             "and outer radii."));

      const GeometryModel::SphericalShell<3> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<3>>(
          this->get_geometry_model());
      AssertThrow(std::abs(geometry.outer_radius() - outer_radius)
                  <= 100.0 * std::numeric_limits<double>::epsilon()
                  * outer_radius,
                  ExcInternalError());

      const double surface_gravity =
        this->get_gravity_model()
        .gravity_vector(geometry.representative_point(0.0)).norm();
      AssertThrow(surface_gravity > 0.0,
                  ExcMessage("The center-of-mass reference mass requires a "
                             "positive surface gravity magnitude."));

      // Surface gravity is the only configured quantity that determines the
      // total mass of the complete planet, including the unresolved core. A
      // CMB-side interface density is not a volume-mean core density.
      return surface_gravity * outer_radius * outer_radius / constants::big_g;
    }



    template <>
    void
    SelfGravitation<3>::update_derived_planetary_constants()
    {
      AssertThrow(Plugins::plugin_type_matches<
                  const GeometryModel::SphericalShell<3>>(
                    this->get_geometry_model()),
                  ExcMessage("Self-gravitation requires a spherical shell geometry."));

      const GeometryModel::SphericalShell<3> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<3>>(
          this->get_geometry_model());

      const double outer_radius = geometry.outer_radius();
      AssertThrow(outer_radius > 0.0,
                  ExcMessage("Self-gravitation requires a positive outer "
                             "radius to derive planetary constants."));

      // The same total planetary mass normalizes both the Green kernels and
      // D-Mc=0. Consequently selecting the COM frame cannot alter l>=2.
      planet_mass =
        compute_reference_planet_mass(geometry.inner_radius(), outer_radius);
      planet_mean_density =
        3.0 * planet_mass
        / (4.0 * numbers::PI
           * outer_radius * outer_radius * outer_radius);

      AssertThrow(planet_mass > 0.0,
                  ExcMessage("Derived planet mass must be positive."));
      AssertThrow(planet_mean_density > 0.0,
                  ExcMessage("Derived planet mean density must be positive."));
    }



    template <>
    bool
    SelfGravitation<3>::potential_is_converged() const
    {
      if (degree_one_reference_frame ==
          DegreeOneReferenceFrame::center_of_mass)
        {
          double constraint_relative_residual =
            std::numeric_limits<double>::infinity();

          if (native_center_of_mass_diagnostic.valid)
            {
              const GeometryModel::SphericalShell<3> &geometry =
                Plugins::get_plugin_as_type<
                const GeometryModel::SphericalShell<3>>(
                  this->get_geometry_model());
              const double residual_scale =
                std::max(
                  native_center_of_mass_diagnostic.mass_dipole_pre.norm(),
                  100.0 * std::numeric_limits<double>::epsilon()
                  * native_center_of_mass_diagnostic.total_mass
                  * geometry.outer_radius());

              constraint_relative_residual =
                native_center_of_mass_diagnostic.mass_dipole_post.norm()
                / residual_scale;
            }

          const bool center_of_mass_change_is_converged =
            center_of_mass_relative_change <= potential_convergence_tolerance
            || (center_of_mass_absolute_tolerance > 0.0
                && center_of_mass_absolute_change
                <= center_of_mass_absolute_tolerance);
          const bool converged =
            potential_relative_change <= potential_convergence_tolerance
            && center_of_mass_change_is_converged
            && constraint_relative_residual <= potential_convergence_tolerance;

          AssertThrow(converged
                      || potential_iteration_number
                      < maximum_potential_iterations,
                      ExcMessage(
                        "The coupled center-of-mass reference-frame solve "
                        "reached the maximum number of potential iterations "
                        "without satisfying the potential-change, COM-"
                        "coefficient-change, and D-Mc residual tolerances."));
          return converged;
        }

      return potential_relative_change <= potential_convergence_tolerance
             || potential_iteration_number >= maximum_potential_iterations;
    }
  }
}
