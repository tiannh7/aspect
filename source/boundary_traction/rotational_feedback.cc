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

#include <aspect/boundary_traction/rotational_feedback.h>
#include <aspect/boundary_traction/self_gravitation.h>
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

namespace aspect
{
  namespace BoundaryTraction
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
      rotational_feedback_boundary_list_contains_surface(
        const std::vector<std::string> &values)
      {
        return rotational_feedback_list_contains(values, "outer")
               || rotational_feedback_list_contains(values, "top");
      }

      bool
      rotational_feedback_boundary_list_contains_cmb(
        const std::vector<std::string> &values)
      {
        return rotational_feedback_list_contains(values, "inner")
               || rotational_feedback_list_contains(values, "bottom");
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
                const bool is_surface = (bid == top_boundary_id);
                const bool is_cmb = (bid == bottom_boundary_id) &&
                                    include_cmb_contribution;
                if (!is_surface && !is_cmb)
                  continue;

                fe_face_values.reinit(cell, f);

                if (include_current_velocity_increment)
                  {
                    mesh_face_values.reinit(current_mesh_cell, f);
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
                          this->get_geometry_model()
                          .height_above_reference_surface(position)
                          + predicted_radial_displacement;

                        Tensor<1,dim> load_traction;
                        const auto &traction_manager =
                          this->get_boundary_traction_manager();
                        const auto &plugins =
                          traction_manager.get_active_plugins();
                        const auto &plugin_boundaries =
                          traction_manager.get_active_plugin_boundary_indicators();
                        unsigned int plugin_index = 0;
                        for (const auto &plugin : plugins)
                          {
                            const bool is_feedback_plugin =
                              (dynamic_cast<const SelfGravitation<dim> *>(plugin.get()) != nullptr
                               || dynamic_cast<const RotationalFeedback<dim> *>(plugin.get()) != nullptr
                               || dynamic_cast<const PotentialFeedback::BoundaryTractionMarker *>(plugin.get()) != nullptr);

                            if (plugin_boundaries[plugin_index] == top_boundary_id
                                && !is_feedback_plugin)
                              load_traction += plugin->boundary_traction(
                                                 top_boundary_id,
                                                 position,
                                                 fe_face_values.normal_vector(q));
                            ++plugin_index;
                          }

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
                          (radius - geometry.inner_radius())
                          + predicted_radial_displacement;
                        sigma = delta_rho_cmb * h_cmb;

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

      Tensor<1,3> local_first_moment;
      double local_delta_ixz = 0.0;
      double local_delta_iyz = 0.0;

      auto accumulate_moments =
        [&local_first_moment, &local_delta_ixz, &local_delta_iyz]
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

            local_first_moment[0] += dm * x;
            local_first_moment[1] += dm * y;
            local_first_moment[2] += dm * z;
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
                                      this->get_mpi_communicator());
      delta_iyz = Utilities::MPI::sum(local_delta_iyz,
                                      this->get_mpi_communicator());
      for (unsigned int d = 0; d < 3; ++d)
        center_of_mass_shift[d] =
          Utilities::MPI::sum(local_first_moment[d],
                              this->get_mpi_communicator())
          / planet_mass;

      const double inertia_contrast =
        polar_moment_of_inertia - equatorial_moment_of_inertia;
      rotation_vector_perturbation = Tensor<1,3>();
      rotation_vector_perturbation[0] =
        rotation_rate * delta_ixz / inertia_contrast;
      rotation_vector_perturbation[1] =
        rotation_rate * delta_iyz / inertia_contrast;

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
        std::vector<double> potential_height(positions.size(), 0.0);
        for (unsigned int i = 0; i < positions.size(); ++i)
          {
            const Tensor<1,dim> gravity =
              this->get_gravity_model().gravity_vector(positions[i]);
            const double g_norm = gravity.norm();
            if (g_norm > 0.0)
              potential_height[i] =
                rotation_rate
                * (rotation_vector_perturbation[0] * positions[i][0]
                   + rotation_vector_perturbation[1] * positions[i][1])
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

      this->get_pcout()
          << "      Rotational feedback potential update: "
          << "relative SH coefficient change="
          << std::scientific << std::setprecision(6)
          << potential_relative_change
          << ", dIxz=" << delta_ixz
          << ", dIyz=" << delta_iyz
          << ", domega=("
          << rotation_vector_perturbation[0] << ", "
          << rotation_vector_perturbation[1] << ", 0)"
          << std::defaultfloat << std::endl;

      if (potential_relative_change > potential_convergence_tolerance
          && potential_iteration_number >= maximum_potential_iterations)
        this->get_pcout()
            << "        status=maximum iterations reached" << std::endl;
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
    void
    RotationalFeedback<dim>::configure_from_potential_feedback_settings(
      const PotentialFeedback::Settings &settings)
    {
      enabled = rotational_feedback_list_contains(settings.feedback_mechanisms,
                                                  "rotational feedback");
      max_degree = settings.rotational_max_degree;
      min_degree = settings.rotational_min_degree;
      density_above_surface =
        settings.interface_properties.surface.density_above;
      density_below_surface =
        settings.interface_properties.surface.density_below;
      density_above_cmb =
        settings.interface_properties.cmb.density_above;
      density_below_cmb =
        settings.interface_properties.cmb.density_below;
      planet_mass = settings.planet.planet_mass;
      polar_moment_of_inertia =
        settings.planet.polar_moment_of_inertia;
      equatorial_moment_of_inertia =
        settings.planet.equatorial_moment_of_inertia;
      rotation_rate = settings.planet.rotation_rate;
      include_cmb_contribution =
        rotational_feedback_list_contains(
          settings.rotational_inertia_source_interfaces, "CMB");
      apply_center_of_mass_correction =
        settings.center_of_mass_correction;
      iterate_with_stokes = settings.iterate_with_stokes;
      initial_displacement_timestep =
        settings.initial_displacement_timestep;
      potential_convergence_tolerance = settings.relative_tolerance;
      maximum_potential_iterations = settings.maximum_iterations;
      enable_surface_potential_traction =
        rotational_feedback_boundary_list_contains_surface(
          settings.rotational_apply_boundaries);
      enable_cmb_potential_traction =
        rotational_feedback_boundary_list_contains_cmb(
          settings.rotational_apply_boundaries);

      if (this->convert_output_to_years())
        initial_displacement_timestep *= year_in_seconds;

      potential_relative_change = std::numeric_limits<double>::infinity();
      current_potential_iteration_step = (unsigned int)-1;
      potential_iteration_number = 0;
      delta_ixz = 0.0;
      delta_iyz = 0.0;
      center_of_mass_shift = Tensor<1,3>();
      rotation_vector_perturbation = Tensor<1,3>();

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
          AssertThrow(planet_mass > 0.0,
                      ExcMessage("Planet mass must be positive."));
          AssertThrow(polar_moment_of_inertia >
                      equatorial_moment_of_inertia,
                      ExcMessage("Polar moment of inertia must be larger than "
                                 "the equatorial moment of inertia for the "
                                 "linearized polar-wander relation."));
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
          prm.declare_entry("Planet mass", "5.9722e24",
                            Patterns::Double(0),
                            "Planet mass in kg, used only to report the "
                            "apparent center-of-mass shift.");
          prm.declare_entry("Polar moment of inertia", "8.034e37",
                            Patterns::Double(0),
                            "Polar moment of inertia C in kg m^2.");
          prm.declare_entry("Equatorial moment of inertia", "8.010e37",
                            Patterns::Double(0),
                            "Equatorial moment of inertia A in kg m^2.");
          prm.declare_entry("Rotation rate", "7.292115e-5",
                            Patterns::Double(0),
                            "Unperturbed angular rotation rate in rad/s.");
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
          planet_mass = prm.get_double("Planet mass");
          polar_moment_of_inertia =
            prm.get_double("Polar moment of inertia");
          equatorial_moment_of_inertia =
            prm.get_double("Equatorial moment of inertia");
          rotation_rate = prm.get_double("Rotation rate");
          include_cmb_contribution =
            prm.get_bool("Include CMB contribution");
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
      center_of_mass_shift = Tensor<1,3>();
      rotation_vector_perturbation = Tensor<1,3>();

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
          AssertThrow(planet_mass > 0.0,
                      ExcMessage("Planet mass must be positive."));
          AssertThrow(polar_moment_of_inertia >
                      equatorial_moment_of_inertia,
                      ExcMessage("Polar moment of inertia must be larger than "
                                 "the equatorial moment of inertia for the "
                                 "linearized polar-wander relation."));
        }
    }
  }
}


// Explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    template class RotationalFeedback<2>;
    template class RotationalFeedback<3>;
  }
}
