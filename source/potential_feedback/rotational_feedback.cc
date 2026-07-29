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

#include <aspect/potential_feedback/rotational_feedback.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/density_source_manager.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <tuple>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace
    {
      bool
      rotational_feedback_list_contains(const std::vector<std::string> &values,
                                        const std::string &name)
      {
        return std::find(values.begin(), values.end(), name) != values.end();
      }

      bool
      print_rotational_feedback_diagnostic_once(
        const std::string &name,
        const unsigned int timestep_number,
        const unsigned int iteration_number)
      {
        static std::set<std::tuple<std::string, unsigned int, unsigned int>>
        printed_diagnostics;

        return printed_diagnostics
               .insert(std::make_tuple(name,
                                       timestep_number,
                                       iteration_number))
               .second;
      }
    }



    template <int dim>
    void
    RotationalFeedback<dim>::initialize()
    {
      if (!enabled)
        return;

      AssertThrow(dim == 3,
                  ExcMessage("Rotational feedback is currently implemented "
                             "only for 3D spherical shells."));
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Rotational feedback requires a spherical shell "
                             "geometry."));

      top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      sh_transform = std::make_unique<Utilities::SphericalHarmonicTransform>(
                       max_degree, 0);

      if (iterate_with_stokes)
        this->get_signals().post_stokes_solver.connect(
          [this](const SimulatorAccess<dim> &,
                 const unsigned int,
                 const unsigned int,
                 const SolverControl &,
                 const SolverControl &)
        {
          this->update_after_stokes_solve();
        });

      this->get_signals().post_mesh_deformation.connect(
        [this](const SimulatorAccess<dim> &)
      {
        this->compute_rotational_feedback(false);
      });
    }



    template <int dim>
    void
    RotationalFeedback<dim>::update()
    {
      if (enabled)
        compute_rotational_feedback(false);
    }



    template <int dim>
    void
    RotationalFeedback<dim>::update_after_stokes_solve()
    {
      compute_rotational_feedback(true);
    }



    template <int dim>
    void
    RotationalFeedback<dim>::compute_rotational_feedback(
      const bool include_current_velocity_increment)
    {
      TimerOutput::Scope update_timer(this->get_computing_timer(),
                                      "Potential feedback: rotational update");

      const std::vector<double> old_surface_cos =
        surface_potential_cos_coeffs;
      const std::vector<double> old_surface_sin =
        surface_potential_sin_coeffs;
      const std::vector<double> old_cmb_cos =
        cmb_potential_cos_coeffs;
      const std::vector<double> old_cmb_sin =
        cmb_potential_sin_coeffs;

      const GeometryModel::SphericalShell<dim> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          this->get_geometry_model());

      const double outer_radius = geometry.outer_radius();
      double displacement_timestep = this->get_timestep();
      if (displacement_timestep == 0.0)
        displacement_timestep = initial_displacement_timestep;

      if (include_current_velocity_increment)
        {
          const unsigned int step = this->get_timestep_number();
          if (current_potential_iteration_step != step)
            {
              current_potential_iteration_step = step;
              potential_iteration_number = 0;
            }
          ++potential_iteration_number;
        }

      const unsigned int quadrature_degree =
        std::max(2u,
                 this->introspection().polynomial_degree.velocities + 1u);
      const QGauss<dim - 1> quadrature_formula_face(quadrature_degree);

      FEFaceValues<dim> fe_face_values(this->get_mapping(),
                                       this->get_fe(),
                                       quadrature_formula_face,
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

      const auto &mesh_deformation_handler =
        this->get_mesh_deformation_handler();
      const DoFHandler<dim> &mesh_deformation_dof_handler =
        mesh_deformation_handler.get_mesh_deformation_dof_handler();
      FEFaceValues<dim> mesh_face_values(
        this->get_mapping(),
        mesh_deformation_dof_handler.get_fe(),
        quadrature_formula_face,
        update_values);
      const FEValuesExtractors::Vector mesh_velocity_extractor(0);

      std::vector<Tensor<1,dim>> mesh_displacement_values(
        mesh_face_values.n_quadrature_points);
      const LinearAlgebra::Vector *projected_mesh_velocity = nullptr;
      if (include_current_velocity_increment)
        projected_mesh_velocity =
          &mesh_deformation_handler.get_projected_free_surface_velocity(true);

      std::vector<Tensor<1,dim>> projected_mesh_velocity_values(
        mesh_face_values.n_quadrature_points);

      const double delta_rho_surface =
        density_below_surface - density_above_surface;
      const double delta_rho_cmb =
        density_below_cmb - density_above_cmb;

      std::vector<Point<dim>> surface_positions;
      std::vector<double> surface_theta;
      std::vector<double> surface_phi;
      std::vector<double> surface_weights;
      std::vector<double> surface_mass_per_area;
      std::vector<Tensor<1,dim>> surface_normals;

      std::vector<Point<dim>> cmb_positions;
      std::vector<double> cmb_theta;
      std::vector<double> cmb_phi;
      std::vector<double> cmb_weights;
      std::vector<double> cmb_mass_per_area;
      std::vector<Tensor<1,dim>> cmb_normals;

      const auto &traction_manager = this->get_boundary_traction_manager();
      const auto &plugins = traction_manager.get_active_plugins();
      const auto &plugin_boundaries =
        traction_manager.get_active_plugin_boundary_indicators();

      const auto load_traction_on_boundary =
        [&plugins,
         &plugin_boundaries,
         this]
        (const types::boundary_id boundary_id,
         const Point<dim> &position,
         const Tensor<1,dim> &face_normal)
      {
        Tensor<1,dim> load_traction;

        unsigned int plugin_index = 0;
        for (const auto &plugin : plugins)
          {
            const bool is_feedback_plugin =
              (dynamic_cast<const SelfGravitation<dim> *>(plugin.get()) != nullptr
               || dynamic_cast<const RotationalFeedback<dim> *>(plugin.get()) != nullptr
               || dynamic_cast<const PotentialFeedback::BoundaryTractionMarker *>(plugin.get()) != nullptr);

            if (plugin_boundaries[plugin_index] == boundary_id
                && !is_feedback_plugin)
              load_traction += plugin->boundary_traction(boundary_id,
                                                         position,
                                                         face_normal);

            ++plugin_index;
          }

        if (additional_load_traction_function)
          load_traction += additional_load_traction_function(boundary_id,
                                                             position,
                                                             face_normal);

        return load_traction;
      };

      auto mesh_cell = mesh_deformation_dof_handler.begin_active();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        {
          const auto current_mesh_cell = mesh_cell;
          ++mesh_cell;

          if (cell->is_locally_owned() && cell->at_boundary())
            for (const unsigned int f : cell->face_indices())
              {
                if (!cell->at_boundary(f))
                  continue;

                const types::boundary_id bid = cell->face(f)->boundary_id();
                const bool is_surface = (bid == top_boundary_id) &&
                                        include_surface_contribution;
                const bool is_cmb = (bid == bottom_boundary_id) &&
                                    include_cmb_contribution;
                if (!is_surface && !is_cmb)
                  continue;

                fe_face_values.reinit(cell, f);
                mesh_face_values.reinit(current_mesh_cell, f);
                mesh_face_values[mesh_velocity_extractor]
                .get_function_values(
                  mesh_deformation_handler.get_mesh_displacements(),
                  mesh_displacement_values);

                if (include_current_velocity_increment)
                  {
                    mesh_face_values[mesh_velocity_extractor]
                    .get_function_values(*projected_mesh_velocity,
                                         projected_mesh_velocity_values);
                  }

                for (unsigned int q = 0;
                     q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    const Point<dim> position =
                      fe_face_values.quadrature_point(q);
                    const std::array<double, dim> scoord =
                      Utilities::Coordinates::cartesian_to_spherical_coordinates(
                        position);
                    const double radius = scoord[0];
                    const double phi = scoord[1];
                    double theta = numbers::PI / 2.0;
                    if constexpr (dim == 3)
                      theta = scoord[2];
                    const Tensor<1,dim> radial_unit = position / radius;
                    const double committed_radial_displacement =
                      mesh_displacement_values[q] * radial_unit;

                    const double predicted_radial_displacement =
                      (include_current_velocity_increment
                       ? displacement_timestep
                       * (projected_mesh_velocity_values[q] * radial_unit)
                       : 0.0);

                    const double reference_radius =
                      is_surface ? outer_radius : geometry.inner_radius();
                    const double weight =
                      fe_face_values.JxW(q) /
                      (reference_radius * reference_radius);

                    double sigma = 0.0;
                    if (is_surface)
                      {
                        const double h_rock =
                          committed_radial_displacement
                          + predicted_radial_displacement;

                        const Tensor<1,dim> load_traction =
                          load_traction_on_boundary(
                            top_boundary_id,
                            position,
                            fe_face_values.normal_vector(q));

                        const double g_norm =
                          this->get_gravity_model()
                          .gravity_vector(position).norm();
                        const double load_mass_per_area =
                          (g_norm > 0.0
                           ? -(load_traction * fe_face_values.normal_vector(q))
                           / g_norm
                           : 0.0);
                        sigma = load_mass_per_area;
                        sigma += delta_rho_surface * h_rock;

                        surface_positions.push_back(position);
                        surface_theta.push_back(theta);
                        surface_phi.push_back(phi);
                        surface_weights.push_back(weight);
                        surface_mass_per_area.push_back(sigma);
                        surface_normals.push_back(fe_face_values.normal_vector(q));
                      }
                    else
                      {
                        const double h_cmb =
                          committed_radial_displacement
                          + predicted_radial_displacement;
                        sigma = delta_rho_cmb * h_cmb;

                        const Tensor<1,dim> load_traction =
                          load_traction_on_boundary(
                            bottom_boundary_id,
                            position,
                            fe_face_values.normal_vector(q));
                        const double g_norm =
                          this->get_gravity_model()
                          .gravity_vector(position).norm();
                        if (g_norm > 0.0)
                          sigma +=
                            (load_traction * fe_face_values.normal_vector(q))
                            / g_norm;

                        cmb_positions.push_back(position);
                        cmb_theta.push_back(theta);
                        cmb_phi.push_back(phi);
                        cmb_weights.push_back(weight);
                        cmb_mass_per_area.push_back(sigma);
                        cmb_normals.push_back(fe_face_values.normal_vector(q));
                      }
                  }
              }
        }

      Assert(mesh_cell == mesh_deformation_dof_handler.end(),
             ExcInternalError());

      double local_delta_ixz = 0.0;
      double local_delta_iyz = 0.0;
      double internal_delta_ixz = 0.0;
      double internal_delta_iyz = 0.0;

      if constexpr (dim == 3)
        if (this->get_density_source_manager()
            .internal_density_anomalies_are_enabled(
              include_internal_density_anomalies))
          {
            const typename DensitySourceManager<dim>::InternalMassMoments
            internal_moments =
              this->get_density_source_manager()
              .compute_internal_mass_moments(
                reference_density_for_internal_anomalies,
                full_domain_volume_source_discretization,
                include_current_velocity_increment);
            internal_delta_ixz = internal_moments.inertia_tensor[0][2];
            internal_delta_iyz = internal_moments.inertia_tensor[1][2];
          }

      {
        TimerOutput::Scope moments_timer(this->get_computing_timer(),
                                         "Potential feedback: rotational moments");
        auto accumulate_moments =
          [&local_delta_ixz, &local_delta_iyz]
          (const std::vector<Point<dim>> &positions,
           const std::vector<double> &weights,
           const std::vector<double> &sigma_values,
           const double reference_radius)
        {
          for (unsigned int i = 0; i < positions.size(); ++i)
            {
              const double dA =
                weights[i] * reference_radius * reference_radius;
              const double dm = sigma_values[i] * dA;
              const double x = positions[i][0];
              const double y = positions[i][1];
              const double z = positions[i][2];

              local_delta_ixz -= dm * x * z;
              local_delta_iyz -= dm * y * z;
            }
        };

        if (apply_center_of_mass_correction && delta_rho_surface != 0.0)
          {
            std::vector<double> effective_height(surface_mass_per_area.size());
            for (unsigned int i = 0; i < surface_mass_per_area.size(); ++i)
              effective_height[i] = surface_mass_per_area[i] / delta_rho_surface;

            const auto height_coefficients =
              sh_transform->analyze(surface_theta,
                                    surface_phi,
                                    surface_weights,
                                    effective_height,
                                    this->get_mpi_communicator());

            std::vector<double> degree_one_cos(sh_transform->n_coefficients(),
                                               0.0);
            std::vector<double> degree_one_sin(sh_transform->n_coefficients(),
                                               0.0);
            for (unsigned int m = 0; m <= 1; ++m)
              {
                const unsigned int index = sh_transform->index(1, m);
                degree_one_cos[index] = height_coefficients.first[index];
                degree_one_sin[index] = height_coefficients.second[index];
              }

            const std::vector<double> degree_one_height =
              sh_transform->synthesize(degree_one_cos,
                                       degree_one_sin,
                                       surface_theta,
                                       surface_phi);

            for (unsigned int i = 0; i < surface_mass_per_area.size(); ++i)
              surface_mass_per_area[i] -=
                delta_rho_surface * degree_one_height[i];
          }

        accumulate_moments(surface_positions,
                           surface_weights,
                           surface_mass_per_area,
                           outer_radius);
        if (include_cmb_contribution)
          accumulate_moments(cmb_positions,
                             cmb_weights,
                             cmb_mass_per_area,
                             geometry.inner_radius());

        delta_ixz = Utilities::MPI::sum(local_delta_ixz,
                                        this->get_mpi_communicator())
                    + internal_delta_ixz;
        delta_iyz = Utilities::MPI::sum(local_delta_iyz,
                                        this->get_mpi_communicator())
                    + internal_delta_iyz;
      }

      const double outer_radius_to_the_fifth =
        outer_radius * outer_radius * outer_radius * outer_radius * outer_radius;
      rotational_potential_prefactor =
        3.0 * constants::big_g
        / (fluid_love_number * outer_radius_to_the_fifth);

      const unsigned int n_coefficients = sh_transform->n_coefficients();
      surface_potential_cos_coeffs.assign(n_coefficients, 0.0);
      surface_potential_sin_coeffs.assign(n_coefficients, 0.0);
      cmb_potential_cos_coeffs.assign(n_coefficients, 0.0);
      cmb_potential_sin_coeffs.assign(n_coefficients, 0.0);

      auto analyze_rotational_potential =
        [this](const std::vector<Point<dim>> &positions,
               const std::vector<double> &theta,
               const std::vector<double> &phi,
               const std::vector<double> &weights)
      {
        TimerOutput::Scope timer(this->get_computing_timer(),
                                 "Potential feedback: rotational SH analysis");
        std::vector<double> potential_height(positions.size(), 0.0);
        for (unsigned int i = 0; i < positions.size(); ++i)
          {
            const Tensor<1,dim> gravity =
              this->get_gravity_model().gravity_vector(positions[i]);
            const double g_norm = gravity.norm();
            if (g_norm > 0.0)
              // The centrifugal-potential perturbation is
              // -3G/(kf R^5) * (dIxz x + dIyz y) z.
              potential_height[i] =
                -rotational_potential_prefactor
                * (delta_ixz * positions[i][0]
                   + delta_iyz * positions[i][1])
                * positions[i][2]
                / g_norm;
          }

        return sh_transform->analyze(theta,
                                     phi,
                                     weights,
                                     potential_height,
                                     this->get_mpi_communicator());
      };

      std::tie(surface_potential_cos_coeffs,
               surface_potential_sin_coeffs) =
                 analyze_rotational_potential(surface_positions,
                                              surface_theta,
                                              surface_phi,
                                              surface_weights);

      if (include_cmb_contribution)
        {
          std::tie(cmb_potential_cos_coeffs,
                   cmb_potential_sin_coeffs) =
                     analyze_rotational_potential(cmb_positions,
                                                  cmb_theta,
                                                  cmb_phi,
                                                  cmb_weights);
        }
      else
        {
          cmb_potential_cos_coeffs.assign(n_coefficients, 0.0);
          cmb_potential_sin_coeffs.assign(n_coefficients, 0.0);
        }

      double direct_rotational_phi21_cosine = 0.0;
      double direct_rotational_phi21_sine = 0.0;
      double self_gravity_phi21_cosine = 0.0;
      double self_gravity_phi21_sine = 0.0;
      double shared_rotational_phi21_cosine = 0.0;
      double shared_rotational_phi21_sine = 0.0;
      potential_source_relative_difference = 0.0;
      if (self_gravity_surface_potential_coefficient_function)
        {
          const unsigned int degree_two_order_one_index =
            sh_transform->index(2, 1);
          direct_rotational_phi21_cosine =
            surface_potential_cos_coeffs[degree_two_order_one_index];
          direct_rotational_phi21_sine =
            surface_potential_sin_coeffs[degree_two_order_one_index];
          const std::pair<double,double> self_gravity_coefficient =
            self_gravity_surface_potential_coefficient_function(2, 1);
          self_gravity_phi21_cosine = self_gravity_coefficient.first;
          self_gravity_phi21_sine = self_gravity_coefficient.second;
          // CitcomSVE's scalar-potential minus sign and the opposite m=1
          // associated-Legendre phase cancel in ASPECT's coefficient basis.
          shared_rotational_phi21_cosine =
            self_gravity_coefficient.first / fluid_love_number;
          shared_rotational_phi21_sine =
            self_gravity_coefficient.second / fluid_love_number;
          const double shared_norm =
            std::hypot(shared_rotational_phi21_cosine,
                       shared_rotational_phi21_sine);
          const double source_difference =
            std::hypot(direct_rotational_phi21_cosine
                       - shared_rotational_phi21_cosine,
                       direct_rotational_phi21_sine
                       - shared_rotational_phi21_sine);
          potential_source_relative_difference =
            (shared_norm > 0.0
             ? source_difference / shared_norm
             : (source_difference == 0.0
                ? 0.0
                : std::numeric_limits<double>::infinity()));

          surface_potential_cos_coeffs.assign(n_coefficients, 0.0);
          surface_potential_sin_coeffs.assign(n_coefficients, 0.0);
          surface_potential_cos_coeffs[degree_two_order_one_index] =
            shared_rotational_phi21_cosine;
          surface_potential_sin_coeffs[degree_two_order_one_index] =
            shared_rotational_phi21_sine;

          cmb_potential_cos_coeffs.assign(n_coefficients, 0.0);
          cmb_potential_sin_coeffs.assign(n_coefficients, 0.0);
          if (include_cmb_contribution)
            {
              Point<dim> surface_reference;
              Point<dim> cmb_reference;
              surface_reference[0] = outer_radius;
              cmb_reference[0] = geometry.inner_radius();
              const double surface_gravity =
                this->get_gravity_model().gravity_vector(surface_reference).norm();
              const double cmb_gravity =
                this->get_gravity_model().gravity_vector(cmb_reference).norm();
              AssertThrow(surface_gravity > 0.0 && cmb_gravity > 0.0,
                          ExcMessage("Shared rotational-potential coefficients "
                                     "require positive surface and CMB gravity."));
              const double radius_ratio =
                geometry.inner_radius() / outer_radius;
              const double cmb_height_scale =
                radius_ratio * radius_ratio * surface_gravity / cmb_gravity;
              cmb_potential_cos_coeffs[degree_two_order_one_index] =
                cmb_height_scale * shared_rotational_phi21_cosine;
              cmb_potential_sin_coeffs[degree_two_order_one_index] =
                cmb_height_scale * shared_rotational_phi21_sine;
            }
        }

      std::vector<double> old_surface_cos_padded = old_surface_cos;
      std::vector<double> old_surface_sin_padded = old_surface_sin;
      std::vector<double> old_cmb_cos_padded = old_cmb_cos;
      std::vector<double> old_cmb_sin_padded = old_cmb_sin;
      old_surface_cos_padded.resize(n_coefficients, 0.0);
      old_surface_sin_padded.resize(n_coefficients, 0.0);
      old_cmb_cos_padded.resize(n_coefficients, 0.0);
      old_cmb_sin_padded.resize(n_coefficients, 0.0);

      double difference_squared = 0.0;
      double new_norm_squared = 0.0;
      auto accumulate_change =
        [&difference_squared, &new_norm_squared]
        (const std::vector<double> &old_values,
         const std::vector<double> &new_values)
      {
        for (unsigned int i = 0; i < new_values.size(); ++i)
          {
            difference_squared +=
              (new_values[i] - old_values[i])
              * (new_values[i] - old_values[i]);
            new_norm_squared += new_values[i] * new_values[i];
          }
      };

      accumulate_change(old_surface_cos_padded, surface_potential_cos_coeffs);
      accumulate_change(old_surface_sin_padded, surface_potential_sin_coeffs);
      accumulate_change(old_cmb_cos_padded, cmb_potential_cos_coeffs);
      accumulate_change(old_cmb_sin_padded, cmb_potential_sin_coeffs);

      if (new_norm_squared == 0.0)
        potential_relative_change = 0.0;
      else
        potential_relative_change =
          std::sqrt(difference_squared) / std::sqrt(new_norm_squared);

      if (print_rotational_feedback_diagnostic_once(
            "relative change",
            this->get_timestep_number(),
            potential_iteration_number))
        {
          this->get_pcout()
              << "      Rotational feedback potential update: "
              << "relative SH coefficient change="
              << std::scientific << std::setprecision(6)
              << potential_relative_change
              << ", dIxz=" << delta_ixz
              << ", dIyz=" << delta_iyz
              << ", internal dIxz=" << internal_delta_ixz
              << ", internal dIyz=" << internal_delta_iyz
              << ", kf=" << fluid_love_number
              << ", direct/shared Phi_21 relative difference="
              << potential_source_relative_difference
              << ", rotational potential prefactor="
              << rotational_potential_prefactor
              << std::defaultfloat << std::endl;

          if (self_gravity_surface_potential_coefficient_function)
            this->get_pcout()
                << "        Phi_21/g dual-path diagnostic: self-gravity=("
                << std::scientific << std::setprecision(6)
                << self_gravity_phi21_cosine << ", "
                << self_gravity_phi21_sine << "), direct rotational=("
                << direct_rotational_phi21_cosine << ", "
                << direct_rotational_phi21_sine << "), shared rotational=("
                << shared_rotational_phi21_cosine << ", "
                << shared_rotational_phi21_sine << ")"
                << std::defaultfloat << std::endl;

          if (potential_relative_change > potential_convergence_tolerance
              && potential_iteration_number >= maximum_potential_iterations)
            this->get_pcout()
                << "        status=maximum iterations reached" << std::endl;
        }
    }



    template <int dim>
    double
    RotationalFeedback<dim>::potential_height(
      const types::boundary_id boundary_indicator,
      const Point<dim> &position) const
    {
      if (!enabled || surface_potential_cos_coeffs.empty())
        return 0.0;

      const bool is_surface = boundary_indicator == top_boundary_id;
      const bool is_cmb = boundary_indicator == bottom_boundary_id;
      if (!is_surface && !is_cmb)
        return 0.0;

      const std::array<double,dim> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      const std::vector<double> &cos_coefficients =
        (is_surface ? surface_potential_cos_coeffs : cmb_potential_cos_coeffs);
      const std::vector<double> &sin_coefficients =
        (is_surface ? surface_potential_sin_coeffs : cmb_potential_sin_coeffs);

      if constexpr (dim == 3)
        return sh_transform->synthesize(cos_coefficients,
                                        sin_coefficients,
        {spherical_coordinates[2]},
      {spherical_coordinates[1]})[0];

      return 0.0;
    }



    template <int dim>
    double
    RotationalFeedback<dim>::full_domain_potential(
      const Point<dim> &position) const
    {
      if constexpr (dim != 3)
        return 0.0;
      else
        {
          if (!enabled || surface_potential_cos_coeffs.empty())
            return 0.0;

          if (!self_gravity_surface_potential_coefficient_function)
            return -rotational_potential_prefactor
                   * (delta_ixz * position[0] + delta_iyz * position[1])
                   * position[2];

          const Point<dim> surface_reference =
            this->get_geometry_model().representative_point(0.0);
          const double outer_radius = surface_reference.norm();
          const double surface_gravity =
            this->get_gravity_model().gravity_vector(surface_reference).norm();
          AssertThrow(outer_radius > 0.0 && surface_gravity > 0.0,
                      ExcMessage("Rotational full-domain potential requires "
                                 "positive surface radius and gravity."));
          const std::array<double,dim> spherical_coordinates =
            Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
          const double surface_potential_height =
            sh_transform->synthesize(surface_potential_cos_coeffs,
                                     surface_potential_sin_coeffs,
          {spherical_coordinates[2]},
          {spherical_coordinates[1]})[0];

          return surface_gravity * surface_potential_height
                 * position.norm_square() / (outer_radius * outer_radius);
        }
    }



    template <int dim>
    void
    RotationalFeedback<dim>::set_additional_load_traction_function(
      const std::function<Tensor<1,dim>(const types::boundary_id,
                                        const Point<dim> &,
                                        const Tensor<1,dim> &)> &function)
    {
      additional_load_traction_function = function;
    }



    template <int dim>
    void
    RotationalFeedback<dim>::
    set_self_gravity_surface_potential_coefficient_function(
      const std::function<std::pair<double,double>(const unsigned int,
                                                   const unsigned int)> &function)
    {
      self_gravity_surface_potential_coefficient_function = function;
    }



    template <int dim>
    Tensor<1, dim>
    RotationalFeedback<dim>::boundary_traction(
      const types::boundary_id boundary_indicator,
      const Point<dim> &position,
      const Tensor<1, dim> &normal_vector) const
    {
      if (!enabled || surface_potential_cos_coeffs.empty())
        return Tensor<1, dim>();

      const bool is_surface = boundary_indicator == top_boundary_id;
      const bool is_cmb = boundary_indicator == bottom_boundary_id;
      if (!is_surface && !is_cmb)
        return Tensor<1, dim>();

      const std::array<double, dim> scoord =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      double polar_angle = numbers::PI / 2.0;
      if constexpr (dim == 3)
        polar_angle = scoord[2];

      const std::vector<double> theta = {polar_angle};
      const std::vector<double> phi = {scoord[1]};

      const std::vector<double> &cos_coeffs =
        (is_surface ? surface_potential_cos_coeffs : cmb_potential_cos_coeffs);
      const std::vector<double> &sin_coeffs =
        (is_surface ? surface_potential_sin_coeffs : cmb_potential_sin_coeffs);

      const double potential_height =
        sh_transform->synthesize(cos_coeffs, sin_coeffs, theta, phi)[0];
      const double g_norm =
        this->get_gravity_model().gravity_vector(position).norm();

      if (is_surface)
        return (enable_surface_potential_traction
                ? density_below_surface * g_norm * potential_height
                * normal_vector
                : Tensor<1, dim>());

      const double delta_rho_cmb = density_below_cmb - density_above_cmb;
      return (enable_cmb_potential_traction
              ? -delta_rho_cmb * g_norm * potential_height * normal_vector
              : Tensor<1, dim>());
    }



    template <int dim>
    bool
    RotationalFeedback<dim>::potential_is_converged() const
    {
      return !enabled
             || potential_relative_change <= potential_convergence_tolerance
             || potential_iteration_number >= maximum_potential_iterations;
    }



    template <int dim>
    double
    RotationalFeedback<dim>::potential_relative_change_value() const
    {
      return potential_relative_change;
    }



    template <int dim>
    std::pair<double,double>
    RotationalFeedback<dim>::surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (!enabled || surface_potential_cos_coeffs.empty()
          || degree < min_degree || degree > max_degree)
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {surface_potential_cos_coeffs.at(index),
              surface_potential_sin_coeffs.at(index)
             };
    }



    template <int dim>
    std::pair<double,double>
    RotationalFeedback<dim>::cmb_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (!enabled || cmb_potential_cos_coeffs.empty()
          || degree < min_degree || degree > max_degree)
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {cmb_potential_cos_coeffs.at(index),
              cmb_potential_sin_coeffs.at(index)
             };
    }



    template <int dim>
    void
    RotationalFeedback<dim>::configure_from_potential_feedback_settings(
      const PotentialFeedback::Settings &settings)
    {
      enabled = rotational_feedback_list_contains(settings.feedback_mechanisms,
                                                  "rotational feedback");
      max_degree = 2;
      min_degree = 0;
      density_above_surface =
        settings.interface_properties.surface.density_above;
      density_below_surface =
        settings.interface_properties.surface.density_below;
      density_above_cmb =
        settings.interface_properties.cmb.density_above;
      density_below_cmb =
        settings.interface_properties.cmb.density_below;
      include_internal_density_anomalies =
        settings.include_internal_density_anomalies;
      full_domain_volume_source_discretization =
        settings.full_domain_volume_source_discretization;
      reference_density_for_internal_anomalies =
        settings.reference_density_for_internal_anomalies;
      fluid_love_number = settings.rotational_fluid_love_number;
      include_surface_contribution =
        settings.include_surface_feedback;
      include_cmb_contribution =
        settings.include_cmb_feedback;
      apply_center_of_mass_correction =
        settings.center_of_mass_correction;
      iterate_with_stokes = settings.iterate_with_stokes;
      initial_displacement_timestep =
        settings.initial_displacement_timestep;
      potential_convergence_tolerance = settings.relative_tolerance;
      maximum_potential_iterations = settings.maximum_iterations;
      enable_surface_potential_traction =
        settings.include_surface_feedback;
      enable_cmb_potential_traction =
        settings.include_cmb_feedback;

      potential_relative_change = std::numeric_limits<double>::infinity();
      potential_source_relative_difference =
        std::numeric_limits<double>::infinity();
      current_potential_iteration_step = (unsigned int)-1;
      potential_iteration_number = 0;
      delta_ixz = 0.0;
      delta_iyz = 0.0;
      rotational_potential_prefactor = 0.0;

      if (enabled)
        {
          AssertThrow(dim == 3,
                      ExcMessage("Rotational feedback is currently "
                                 "implemented only for 3D."));
          AssertThrow(fluid_love_number > 0.0,
                      ExcMessage("Fluid Love number must be positive."));
          AssertThrow(include_surface_contribution || include_cmb_contribution,
                      ExcMessage("The `potential feedback' boundary traction "
                                 "model must be prescribed on at least the "
                                 "top/surface or bottom/CMB boundary when "
                                 "rotational feedback is enabled."));
        }
    }



    template <int dim>
    void
    RotationalFeedback<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Rotational feedback");
        {
          prm.declare_entry("Enable", "false",
                            Patterns::Bool(),
                            "Whether to enable polar-wander rotational "
                            "feedback. If false, this plugin returns zero "
                            "traction even when it is listed as a boundary "
                            "traction model.");
          prm.declare_entry("Maximum degree", "2",
                            Patterns::Integer(2),
                            "Maximum spherical harmonic degree used for "
                            "diagnostic transforms. Rotational feedback is a "
                            "degree-2 forcing in this implementation.");
          prm.declare_entry("Minimum degree", "0",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree retained for "
                            "diagnostic transforms. Set this to 0 when center "
                            "of mass correction is enabled.");
          prm.declare_entry("Density above surface", "0",
                            Patterns::Double(0),
                            "Density immediately above the surface boundary "
                            "in kg/m^3.");
          prm.declare_entry("Density below surface", "3500",
                            Patterns::Double(0),
                            "Density immediately below the surface boundary "
                            "in kg/m^3.");
          prm.declare_entry("Density above CMB", "5500",
                            Patterns::Double(0),
                            "Density immediately above the CMB in kg/m^3.");
          prm.declare_entry("Density below CMB", "9900",
                            Patterns::Double(0),
                            "Density immediately below the CMB in kg/m^3.");
          prm.declare_entry("Fluid Love number", "1.0",
                            Patterns::Double(0),
                            "Fluid degree-2 Love number k_f used in the "
                            "linearized polar-wander relation. This is the "
                            "same quantity as CitcomSVE's polar_wander_kf.");
          prm.declare_entry("Include CMB contribution", "false",
                            Patterns::Bool(),
                            "Whether CMB topography contributes to the "
                            "products of inertia and receives the induced "
                            "rotational potential traction.");
          prm.declare_entry("Apply center of mass correction", "true",
                            Patterns::Bool(),
                            "Whether to remove the degree-1 part of the "
                            "effective surface mass before computing the "
                            "products of inertia. This is a benchmark-scale "
                            "center-of-mass frame correction, not a full GIA "
                            "sea-level center-of-mass treatment.");
          prm.declare_entry("Iterate with Stokes", "true",
                            Patterns::Bool(),
                            "Recompute the rotational potential from the "
                            "current Stokes velocity after every Stokes solve.");
          prm.declare_entry("Initial displacement time step", "0",
                            Patterns::Double(0),
                            "Displacement interval used to convert the "
                            "timestep-0 Stokes velocity into an incremental "
                            "boundary displacement.");
          prm.declare_entry("Potential convergence tolerance", "1e-3",
                            Patterns::Double(0),
                            "Relative L2 change tolerance for the rotational "
                            "potential coefficient vectors.");
          prm.declare_entry("Maximum potential iterations", "10",
                            Patterns::Integer(1),
                            "Maximum number of self-consistent rotational "
                            "potential updates per timestep. The iteration "
                            "stops when the potential coefficient change "
                            "reaches the tolerance or this limit is reached.");
          prm.declare_entry("Enable surface potential traction", "true",
                            Patterns::Bool(),
                            "Whether to apply the induced rotational "
                            "potential traction at the outer surface.");
          prm.declare_entry("Enable CMB potential traction", "false",
                            Patterns::Bool(),
                            "Whether to apply the induced rotational "
                            "potential traction at the CMB.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    RotationalFeedback<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Rotational feedback");
        {
          enabled = prm.get_bool("Enable");
          max_degree = prm.get_integer("Maximum degree");
          min_degree = prm.get_integer("Minimum degree");
          density_above_surface = prm.get_double("Density above surface");
          density_below_surface = prm.get_double("Density below surface");
          density_above_cmb = prm.get_double("Density above CMB");
          density_below_cmb = prm.get_double("Density below CMB");
          fluid_love_number = prm.get_double("Fluid Love number");
          include_cmb_contribution =
            prm.get_bool("Include CMB contribution");
          include_surface_contribution = true;
          apply_center_of_mass_correction =
            prm.get_bool("Apply center of mass correction");
          iterate_with_stokes = prm.get_bool("Iterate with Stokes");
          initial_displacement_timestep =
            prm.get_double("Initial displacement time step");
          potential_convergence_tolerance =
            prm.get_double("Potential convergence tolerance");
          maximum_potential_iterations =
            prm.get_integer("Maximum potential iterations");
          enable_surface_potential_traction =
            prm.get_bool("Enable surface potential traction");
          enable_cmb_potential_traction =
            prm.get_bool("Enable CMB potential traction");

          if (this->convert_output_to_years())
            initial_displacement_timestep *= year_in_seconds;
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      potential_relative_change = std::numeric_limits<double>::infinity();
      current_potential_iteration_step = (unsigned int)-1;
      potential_iteration_number = 0;
      delta_ixz = 0.0;
      delta_iyz = 0.0;
      rotational_potential_prefactor = 0.0;

      if (enabled)
        {
          AssertThrow(dim == 3,
                      ExcMessage("Rotational feedback is currently "
                                 "implemented only for 3D."));
          AssertThrow(max_degree >= 2,
                      ExcMessage("Rotational feedback requires Maximum degree "
                                 "to be at least 2."));
          AssertThrow(min_degree <= 2,
                      ExcMessage("Rotational feedback must retain degree 2."));
          AssertThrow(!apply_center_of_mass_correction || min_degree == 0,
                      ExcMessage("Center of mass correction requires Minimum "
                                 "degree = 0 so degree-1 mass can be "
                                 "identified and removed."));
          AssertThrow(fluid_love_number > 0.0,
                      ExcMessage("Fluid Love number must be positive."));
        }
    }
  }
}


// Explicit instantiations
namespace aspect
{
  namespace PotentialFeedback
  {
    template class RotationalFeedback<2>;
    template class RotationalFeedback<3>;
  }
}
