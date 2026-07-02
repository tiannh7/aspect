/*
  Copyright (C) 2024 by the authors of the ASPECT code.

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

#include <aspect/boundary_traction/self_gravitation.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <tuple>
#include <numeric>
#include <set>

namespace aspect
{
  namespace BoundaryTraction
  {
    namespace
    {
      bool
      self_gravity_list_contains(const std::vector<std::string> &values,
                                 const std::string &name)
      {
        return std::find(values.begin(), values.end(), name) != values.end();
      }

      bool
      print_self_gravity_diagnostic_once(
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
    SelfGravitation<dim>::initialize()
    {
      top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      if (configured_from_potential_feedback)
        {
          enable_surface_potential_traction = true;
          enable_cmb_potential_traction = true;

          if (has_legacy_apply_boundaries)
            {
              dealii::ConditionalOStream pcout(
                std::cout,
                dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
              pcout << "WARNING: Parameter <Potential feedback/Self gravity/"
                    << "Apply to boundary indicators> is deprecated. "
                    << "Self-gravity traction boundaries are now inferred from "
                    << "the active `potential feedback' boundary traction "
                    << "assignments." << std::endl;
            }
        }
      else
        {
          const double mm_initial_elastic_dt = this->get_material_model().initial_elastic_time_step();
          if (mm_initial_elastic_dt > 0.0 && initial_displacement_timestep == 0.0)
            {
              initial_displacement_timestep = mm_initial_elastic_dt;
            }
        }

      last_text_output_time = -1.0;
      last_text_output_step = 0;
      current_tracked_step = (unsigned int)-1;
      printing_this_step = true;

      if (dim == 3)
        sh_transform = std::make_unique<Utilities::SphericalHarmonicTransform>(
                         max_degree, min_degree);
      else
        fourier_transform = std::make_unique<Utilities::FourierTransform>(
                              max_degree, min_degree);

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
        this->compute_self_gravity_correction(false);
      });
    }


    template <int dim>
    void
    SelfGravitation<dim>::update()
    {
      compute_self_gravity_correction(false);
    }


    template <int dim>
    void
    SelfGravitation<dim>::update_after_stokes_solve()
    {
      compute_self_gravity_correction(true);
    }


    template <int dim>
    void
    SelfGravitation<dim>::compute_self_gravity_correction(
      const bool include_current_velocity_increment)
    {
      if (freeze_potential_after_timestep_zero &&
          this->get_timestep_number() > 0)
        return;

      const std::vector<double> old_surface_potential_cos =
        surface_potential_cos_coeffs;
      const std::vector<double> old_surface_potential_sin =
        surface_potential_sin_coeffs;
      const std::vector<double> old_cmb_potential_cos =
        cmb_potential_cos_coeffs;
      const std::vector<double> old_cmb_potential_sin =
        cmb_potential_sin_coeffs;
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Self-gravitation requires a spherical shell geometry."));

      const GeometryModel::SphericalShell<dim> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          this->get_geometry_model());

      const double outer_radius = geometry.outer_radius();
      const double inner_radius = geometry.inner_radius();
      const double radius_ratio = inner_radius / outer_radius;
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

      // Step 1: Collect surface and CMB topography at quadrature points
      // The projected field contains the Q2 Stokes velocity predictor. Using
      // the temperature degree (Q1 in this benchmark) supplies only one
      // quadrature point per face direction and biases the l=2 surface/CMB
      // coefficients by several percent. Integrate at least one order above
      // the velocity polynomial degree.
      const unsigned int quadrature_degree =
        std::max(2u,
                 this->introspection().polynomial_degree.velocities + 1u);
      const QGauss<dim - 1> quadrature_formula_face(quadrature_degree);

      FEFaceValues<dim> fe_face_values(this->get_mapping(),
                                       this->get_fe(),
                                       quadrature_formula_face,
                                       update_values |
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

      const double delta_rho_surf = density_below_surface - density_above_surface;

      // Surface topography data
      std::vector<double> phi_pts;
      std::vector<double> theta_pts; // only used in 3D
      std::vector<double> weight_pts;
      std::vector<double> topo_pts;

      // CMB topography data
      std::vector<double> cmb_phi_pts;
      std::vector<double> cmb_theta_pts; // only used in 3D
      std::vector<double> cmb_weight_pts;
      std::vector<double> cmb_topo_pts;
      std::vector<double> cmb_committed_topo_pts;

      auto mesh_cell = mesh_deformation_dof_handler.begin_active();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        {
          const auto current_mesh_cell = mesh_cell;
          ++mesh_cell;
          if (cell->is_locally_owned() && cell->at_boundary())
            {
              for (const unsigned int f : cell->face_indices())
                {
                  if (!cell->at_boundary(f))
                    continue;

                  const types::boundary_id bid = cell->face(f)->boundary_id();
                  const bool is_top    = (bid == top_boundary_id);
                  const bool is_bottom = (bid == bottom_boundary_id) && include_cmb_contribution;

                  if (!is_top && !is_bottom)
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
                        aspect::Utilities::Coordinates::
                        cartesian_to_spherical_coordinates(position);

                      // scoord: 2D = {r, phi}, 3D = {r, phi, theta}
                      const double ph = scoord[1]; // longitude / azimuthal angle
                      const Tensor<1,dim> radial_unit = position / scoord[0];
                      const double predicted_radial_displacement =
                        (include_current_velocity_increment
                         ? displacement_timestep
                         * (projected_mesh_velocity_values[q] * radial_unit)
                         : 0.0);

                      if (is_top)
                        {
                          const double h_rock =
                            this->get_geometry_model()
                            .height_above_reference_surface(position)
                            + predicted_radial_displacement;

                          // Compute the external load's equivalent height. A
                          // stateful traction plugin is instantiated once for
                          // every boundary entry in ASPECT's manager. Therefore
                          // subtracting only `this` instance from the manager's
                          // total traction is incorrect when self gravitation is
                          // active on both surface and CMB. Sum only non-self-
                          // gravity plugins assigned to the surface instead.
                          const Tensor<1,dim> face_normal =
                            fe_face_values.normal_vector(q);
                          Tensor<1,dim> load_traction;
                          const auto &traction_manager =
                            this->get_boundary_traction_manager();
                          const auto &plugins = traction_manager.get_active_plugins();
                          const auto &plugin_boundaries =
                            traction_manager.get_active_plugin_boundary_indicators();
                          unsigned int plugin_index = 0;
                          for (const auto &plugin : plugins)
                            {
                              if (plugin_boundaries[plugin_index] == top_boundary_id
                                  && dynamic_cast<const SelfGravitation<dim> *>(plugin.get()) == nullptr
                                  && dynamic_cast<const PotentialFeedback::BoundaryTractionMarker *>(plugin.get()) == nullptr)
                                load_traction += plugin->boundary_traction(
                                                   top_boundary_id, position, face_normal);
                              ++plugin_index;
                            }

                          // Inward load traction (T·n < 0) → positive surface mass
                          // σ_load = -T_load·n / g,  h_load = σ_load / Δρ
                          const double g_magnitude =
                            this->get_gravity_model().gravity_vector(position).norm();

                          double h_load = 0.0;
                          if (g_magnitude > 0 && delta_rho_surf > 0)
                            h_load = -(load_traction * face_normal) /
                                     (delta_rho_surf * g_magnitude);

                          const double h_effective = h_rock + h_load;

                          const double ref_radius = outer_radius;
                          const double w =
                            fe_face_values.JxW(q) /
                            (dim == 3 ? ref_radius *ref_radius : ref_radius);

                          phi_pts.push_back(ph);
                          if (dim == 3)
                            theta_pts.push_back(scoord[2]);
                          weight_pts.push_back(w);
                          topo_pts.push_back(h_effective);
                        }
                      else // is_bottom
                        {
                          const double r = scoord[0];
                          const double committed_cmb_topography = r - inner_radius;
                          const double cmb_topography =
                            committed_cmb_topography + predicted_radial_displacement;
                          const double ref_radius = inner_radius;
                          const double w =
                            fe_face_values.JxW(q) /
                            (dim == 3 ? ref_radius *ref_radius : ref_radius);

                          cmb_phi_pts.push_back(ph);
                          if (dim == 3)
                            cmb_theta_pts.push_back(scoord[2]);
                          cmb_weight_pts.push_back(w);
                          cmb_topo_pts.push_back(cmb_topography);
                          cmb_committed_topo_pts.push_back(
                            committed_cmb_topography);
                        }
                    }
                }
            }
        }

      Assert(mesh_cell == mesh_deformation_dof_handler.end(),
             ExcInternalError());

      // Step 2 & 3: SH/Fourier analysis + self-gravity kernel
      //
      // 3D self-gravity ratio: Rsg(l) = 3*delta_rho / ((2l+1)*rho_mean)
      // 2D self-gravity ratio: Rsg(n) = 2*delta_rho / (n * rho_mean)  [n>=1]
      //   (For n=0, Rsg=0 since uniform mass shift does not change the potential gradient.)
      //
      // CMB scaling: 3D: (r_cmb/R)^(l+2),  2D: (r_cmb/R)^(n+1)

      const double delta_rho_cmb  = density_below_cmb - density_above_cmb;

      if (dim == 3)
        {
          auto [cos_topo, sin_topo] = sh_transform->analyze(
                                        theta_pts, phi_pts, weight_pts, topo_pts,
                                        this->get_mpi_communicator());
          const unsigned int n_coeff = sh_transform->n_coefficients();

          std::vector<double> cos_cmb(n_coeff, 0.0);
          std::vector<double> sin_cmb(n_coeff, 0.0);
          // analyze() performs MPI collectives, so every rank must call it.
          // Ranks without locally owned CMB faces contribute empty vectors,
          // which correctly produce a zero local contribution.
          if (include_cmb_contribution)
            {
              std::tie(cos_cmb, sin_cmb) = sh_transform->analyze(
                                             cmb_theta_pts, cmb_phi_pts,
                                             cmb_weight_pts, cmb_topo_pts,
                                             this->get_mpi_communicator());
              std::tie(cmb_committed_topography_cos_coeffs,
                       cmb_committed_topography_sin_coeffs) =
                         sh_transform->analyze(
                           cmb_theta_pts, cmb_phi_pts, cmb_weight_pts,
                           cmb_committed_topo_pts, this->get_mpi_communicator());
            }

          cmb_topography_cos_coeffs = cos_cmb;
          cmb_topography_sin_coeffs = sin_cmb;

          const unsigned int step = this->get_timestep_number();
          const double time = this->get_time();

          if (current_tracked_step != step)
            {
              current_tracked_step = step;
              printing_this_step = false;

              const double eff_time_interval = time_between_text_output;
              const unsigned int eff_step_interval = time_steps_between_text_output;

              if (eff_step_interval > 0 || eff_time_interval > 0.0)
                {
                  if (step == 0 || time == 0.0)
                    printing_this_step = true;
                  else if (eff_step_interval > 0 && (step - last_text_output_step >= eff_step_interval))
                    printing_this_step = true;
                  else if (eff_time_interval > 0 && (time - last_text_output_time >= eff_time_interval))
                    printing_this_step = true;
                }

              if (printing_this_step)
                {
                  last_text_output_step = step;
                  last_text_output_time = time;
                }
            }

          if (printing_this_step
              &&
              print_self_gravity_diagnostic_once("coefficient norms",
                                                 step,
                                                 potential_iteration_number))
            {
              const auto coefficient_l2_norm =
                [](const std::vector<double> &cos_coeffs,
                   const std::vector<double> &sin_coeffs)
              {
                double norm_squared = 0.0;
                for (const double value : cos_coeffs)
                  norm_squared += value * value;
                for (const double value : sin_coeffs)
                  norm_squared += value * value;
                return std::sqrt(norm_squared);
              };

              this->get_pcout()
                  << "      Self-gravity effective boundary SH coefficient L2 norm [m]:"
                  << std::scientific << std::setprecision(6)
                  << " surface=" << coefficient_l2_norm(cos_topo, sin_topo)
                  << ", CMB=" << coefficient_l2_norm(cos_cmb, sin_cmb)
                  << std::defaultfloat << std::endl;
            }

          // Phi/g at the surface.
          std::vector<double> surface_to_surface(max_degree + 1, 0.0);
          std::vector<double> cmb_to_surface(max_degree + 1, 0.0);
          // Phi/g at the CMB.
          std::vector<double> surface_to_cmb(max_degree + 1, 0.0);
          std::vector<double> cmb_to_cmb(max_degree + 1, 0.0);

          for (unsigned int l = min_degree; l <= max_degree; ++l)
            {
              const double common =
                3.0 / ((2.0 * l + 1.0) * planet_mean_density);
              surface_to_surface[l] = common * delta_rho_surf;
              cmb_to_surface[l] = common * delta_rho_cmb
                                  * std::pow(radius_ratio,
                                             static_cast<int>(l) + 2);
              surface_to_cmb[l] = common * delta_rho_surf
                                  * std::pow(radius_ratio,
                                             static_cast<int>(l));
              cmb_to_cmb[l] = common * delta_rho_cmb * radius_ratio;
            }

          surface_potential_cos_coeffs = cos_topo;
          surface_potential_sin_coeffs = sin_topo;
          sh_transform->apply_degree_filter(surface_potential_cos_coeffs,
                                            surface_potential_sin_coeffs,
                                            surface_to_surface);
          surface_mass_potential_cos_coeffs = surface_potential_cos_coeffs;
          surface_mass_potential_sin_coeffs = surface_potential_sin_coeffs;
          std::vector<double> cmb_at_surface_cos = cos_cmb;
          std::vector<double> cmb_at_surface_sin = sin_cmb;
          sh_transform->apply_degree_filter(cmb_at_surface_cos,
                                            cmb_at_surface_sin,
                                            cmb_to_surface);
          cmb_mass_potential_cos_coeffs = cmb_at_surface_cos;
          cmb_mass_potential_sin_coeffs = cmb_at_surface_sin;

          cmb_potential_cos_coeffs = cos_topo;
          cmb_potential_sin_coeffs = sin_topo;
          sh_transform->apply_degree_filter(cmb_potential_cos_coeffs,
                                            cmb_potential_sin_coeffs,
                                            surface_to_cmb);
          std::vector<double> cmb_at_cmb_cos = cos_cmb;
          std::vector<double> cmb_at_cmb_sin = sin_cmb;
          sh_transform->apply_degree_filter(cmb_at_cmb_cos,
                                            cmb_at_cmb_sin,
                                            cmb_to_cmb);

          for (unsigned int i = 0; i < n_coeff; ++i)
            {
              surface_potential_cos_coeffs[i] += cmb_at_surface_cos[i];
              surface_potential_sin_coeffs[i] += cmb_at_surface_sin[i];
              cmb_potential_cos_coeffs[i] += cmb_at_cmb_cos[i];
              cmb_potential_sin_coeffs[i] += cmb_at_cmb_sin[i];
            }

          tidal_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);

          tidal_potential.add_to_coefficients(
            *sh_transform,
            radius_ratio,
            surface_potential_cos_coeffs,
            surface_potential_sin_coeffs,
            cmb_potential_cos_coeffs,
            cmb_potential_sin_coeffs,
            tidal_surface_potential_cos_coeffs,
            tidal_surface_potential_sin_coeffs,
            tidal_cmb_potential_cos_coeffs,
            tidal_cmb_potential_sin_coeffs);
        }
      else
        {
          auto [cos_topo, sin_topo] = fourier_transform->analyze(
                                        phi_pts, weight_pts, topo_pts,
                                        this->get_mpi_communicator());
          const unsigned int n_coeff = fourier_transform->n_coefficients();

          std::vector<double> cos_cmb(n_coeff, 0.0);
          std::vector<double> sin_cmb(n_coeff, 0.0);
          // analyze() performs MPI collectives, so every rank must call it.
          // Ranks without locally owned CMB faces contribute empty vectors,
          // which correctly produce a zero local contribution.
          if (include_cmb_contribution)
            {
              std::tie(cos_cmb, sin_cmb) = fourier_transform->analyze(
                                             cmb_phi_pts, cmb_weight_pts, cmb_topo_pts,
                                             this->get_mpi_communicator());
              std::tie(cmb_committed_topography_cos_coeffs,
                       cmb_committed_topography_sin_coeffs) =
                         fourier_transform->analyze(
                           cmb_phi_pts, cmb_weight_pts, cmb_committed_topo_pts,
                           this->get_mpi_communicator());
            }

          cmb_topography_cos_coeffs = cos_cmb;
          cmb_topography_sin_coeffs = sin_cmb;

          std::vector<double> surface_to_surface(max_degree + 1, 0.0);
          std::vector<double> cmb_to_surface(max_degree + 1, 0.0);
          std::vector<double> surface_to_cmb(max_degree + 1, 0.0);
          std::vector<double> cmb_to_cmb(max_degree + 1, 0.0);
          for (unsigned int n = std::max(min_degree, 1u); n <= max_degree; ++n)
            {
              const double common =
                2.0 / (static_cast<double>(n) * planet_mean_density);
              surface_to_surface[n] = common * delta_rho_surf;
              cmb_to_surface[n] = common * delta_rho_cmb
                                  * std::pow(radius_ratio,
                                             static_cast<int>(n) + 1);
              surface_to_cmb[n] = common * delta_rho_surf
                                  * std::pow(radius_ratio,
                                             static_cast<int>(n));
              cmb_to_cmb[n] = common * delta_rho_cmb * radius_ratio;
            }

          surface_potential_cos_coeffs = cos_topo;
          surface_potential_sin_coeffs = sin_topo;
          fourier_transform->apply_degree_filter(surface_potential_cos_coeffs,
                                                 surface_potential_sin_coeffs,
                                                 surface_to_surface);
          surface_mass_potential_cos_coeffs = surface_potential_cos_coeffs;
          surface_mass_potential_sin_coeffs = surface_potential_sin_coeffs;
          std::vector<double> cmb_at_surface_cos = cos_cmb;
          std::vector<double> cmb_at_surface_sin = sin_cmb;
          fourier_transform->apply_degree_filter(cmb_at_surface_cos,
                                                 cmb_at_surface_sin,
                                                 cmb_to_surface);
          cmb_mass_potential_cos_coeffs = cmb_at_surface_cos;
          cmb_mass_potential_sin_coeffs = cmb_at_surface_sin;

          cmb_potential_cos_coeffs = cos_topo;
          cmb_potential_sin_coeffs = sin_topo;
          fourier_transform->apply_degree_filter(cmb_potential_cos_coeffs,
                                                 cmb_potential_sin_coeffs,
                                                 surface_to_cmb);
          std::vector<double> cmb_at_cmb_cos = cos_cmb;
          std::vector<double> cmb_at_cmb_sin = sin_cmb;
          fourier_transform->apply_degree_filter(cmb_at_cmb_cos,
                                                 cmb_at_cmb_sin,
                                                 cmb_to_cmb);

          for (unsigned int i = 0; i < n_coeff; ++i)
            {
              surface_potential_cos_coeffs[i] += cmb_at_surface_cos[i];
              surface_potential_sin_coeffs[i] += cmb_at_surface_sin[i];
              cmb_potential_cos_coeffs[i] += cmb_at_cmb_cos[i];
              cmb_potential_sin_coeffs[i] += cmb_at_cmb_sin[i];
            }

          tidal_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
        }

      if (include_current_velocity_increment &&
          !old_surface_potential_cos.empty())
        {
          double difference_squared = 0.0;
          double new_norm_squared = 0.0;
          const auto accumulate_change =
            [&difference_squared, &new_norm_squared](
              const std::vector<double> &old_values,
              const std::vector<double> &new_values)
          {
            AssertDimension(old_values.size(), new_values.size());
            for (unsigned int i=0; i<new_values.size(); ++i)
              {
                difference_squared +=
                  (new_values[i]-old_values[i]) *
                  (new_values[i]-old_values[i]);
                new_norm_squared += new_values[i] * new_values[i];
              }
          };

          accumulate_change(old_surface_potential_cos,
                            surface_potential_cos_coeffs);
          accumulate_change(old_surface_potential_sin,
                            surface_potential_sin_coeffs);
          accumulate_change(old_cmb_potential_cos,
                            cmb_potential_cos_coeffs);
          accumulate_change(old_cmb_potential_sin,
                            cmb_potential_sin_coeffs);

          potential_relative_change =
            std::sqrt(difference_squared) /
            std::max(std::sqrt(new_norm_squared),
                     std::numeric_limits<double>::min());

          if (print_self_gravity_diagnostic_once("relative change",
                                                 this->get_timestep_number(),
                                                 potential_iteration_number))
            {
              this->get_pcout()
                  << "      Self-gravity potential update: "
                  << "relative SH coefficient change="
                  << std::scientific << std::setprecision(6)
                  << potential_relative_change << std::defaultfloat << std::endl;

              if (potential_relative_change > potential_convergence_tolerance
                  && potential_iteration_number >= maximum_potential_iterations)
                this->get_pcout()
                    << "        status=maximum iterations reached" << std::endl;
            }
        }
    }


    template <int dim>
    bool
    SelfGravitation<dim>::potential_is_converged() const
    {
      return potential_relative_change <= potential_convergence_tolerance
             || potential_iteration_number >= maximum_potential_iterations;
    }


    template <int dim>
    double
    SelfGravitation<dim>::potential_relative_change_value() const
    {
      return potential_relative_change;
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::surface_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      const unsigned int index = sh_transform->index(degree, order);
      return {surface_mass_potential_cos_coeffs.at(index),
              surface_mass_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::cmb_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      const unsigned int index = sh_transform->index(degree, order);
      return {cmb_mass_potential_cos_coeffs.at(index),
              cmb_mass_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::tidal_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (tidal_surface_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {tidal_surface_potential_cos_coeffs.at(index),
              tidal_surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    double
    SelfGravitation<dim>::surface_density_jump() const
    {
      return density_below_surface - density_above_surface;
    }


    template <int dim>
    double
    SelfGravitation<dim>::cmb_density_jump() const
    {
      return density_below_cmb - density_above_cmb;
    }


    template <int dim>
    Tensor<1, dim>
    SelfGravitation<dim>::boundary_traction(
      const types::boundary_id boundary_indicator,
      const Point<dim> &position,
      const Tensor<1, dim> &normal_vector) const
    {
      // Convert position to spherical coordinates
      const std::array<double, dim> scoord =
        aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(
          position);
      const double ph = scoord[1]; // longitude / azimuthal angle

      if (surface_potential_cos_coeffs.empty())
        return Tensor<1, dim>();

      const bool is_surface = boundary_indicator == top_boundary_id;
      const bool is_cmb = boundary_indicator == bottom_boundary_id;
      if (!is_surface && !is_cmb)
        return Tensor<1, dim>();

      const std::vector<double> &potential_cos =
        (is_surface ? surface_potential_cos_coeffs : cmb_potential_cos_coeffs);
      const std::vector<double> &potential_sin =
        (is_surface ? surface_potential_sin_coeffs : cmb_potential_sin_coeffs);

      double potential_height = 0.0;
      double cmb_topography = 0.0;
      if (dim == 3)
        {
          const double th = scoord[2]; // colatitude
          const std::vector<double> th_vec = {th};
          const std::vector<double> ph_vec = {ph};
          const std::vector<double> potential =
            sh_transform->synthesize(potential_cos,
                                     potential_sin,
                                     th_vec, ph_vec);
          potential_height = potential[0];


          if (is_cmb && include_cmb_contribution)
            cmb_topography = sh_transform->synthesize(
                               cmb_committed_topography_cos_coeffs,
                               cmb_committed_topography_sin_coeffs,
                               th_vec, ph_vec)[0];
        }
      else
        {
          const std::vector<double> ph_vec = {ph};
          const std::vector<double> potential =
            fourier_transform->synthesize(potential_cos,
                                          potential_sin,
                                          ph_vec);
          potential_height = potential[0];


          if (is_cmb && include_cmb_contribution)
            cmb_topography = fourier_transform->synthesize(
                               cmb_committed_topography_cos_coeffs,
                               cmb_committed_topography_sin_coeffs,
                               ph_vec)[0];
        }

      const Tensor<1, dim> gravity =
        this->get_gravity_model().gravity_vector(position);
      const double g_magnitude = gravity.norm();
      const double delta_rho_cmb = density_below_cmb - density_above_cmb;

      if (is_surface)
        {
          double committed_surface_topography = 0.0;
          if (this->get_timestep_number() > 0)
            committed_surface_topography =
              this->get_geometry_model().height_above_reference_surface(position);

          // CitcomSVE keeps the current displacement increment in the local
          // restoring matrix and carries committed topography as an RHS load.
          return density_below_surface * g_magnitude
                 * (-committed_surface_topography
                    + (enable_surface_potential_traction
                       ? potential_height
                       : 0.0))
                 * normal_vector;
        }

      // Fluid-core CMB condition after subtracting the mantle hydrostatic
      // reference state: Delta rho * (g*h_b - Phi_b) n.
      return delta_rho_cmb * g_magnitude
             * (cmb_topography
                + (enable_cmb_potential_traction
                   ? -potential_height
                   : 0.0))
             * normal_vector;
    }



    template <int dim>
    void
    SelfGravitation<dim>::configure_from_potential_feedback_settings(
      const PotentialFeedback::Settings &settings)
    {
      max_degree = settings.self_gravity_max_degree;
      min_degree = settings.self_gravity_min_degree;
      density_above_surface =
        settings.interface_properties.surface.density_above;
      density_below_surface =
        settings.interface_properties.surface.density_below;
      density_above_cmb =
        settings.interface_properties.cmb.density_above;
      density_below_cmb =
        settings.interface_properties.cmb.density_below;
      planet_mean_density = settings.planet.planet_mean_density;
      include_cmb_contribution =
        self_gravity_list_contains(settings.self_gravity_source_interfaces,
                                   "CMB");
      iterate_with_stokes = settings.iterate_with_stokes;
      freeze_potential_after_timestep_zero =
        settings.freeze_feedback_after_timestep_zero;
      initial_displacement_timestep =
        settings.initial_displacement_timestep;
      potential_convergence_tolerance = settings.relative_tolerance;
      maximum_potential_iterations = settings.maximum_iterations;
      has_legacy_apply_boundaries = settings.has_legacy_apply_boundaries;
      legacy_apply_boundaries = settings.legacy_apply_boundaries;
      configured_from_potential_feedback = true;
      time_between_text_output = 0.0;
      time_steps_between_text_output = 0;
      potential_relative_change = std::numeric_limits<double>::infinity();
      current_potential_iteration_step = (unsigned int)-1;
      potential_iteration_number = 0;

      AssertThrow(min_degree <= max_degree,
                  ExcMessage("Potential feedback/Self gravity/Minimum degree "
                             "must not exceed Maximum degree."));
      AssertThrow(planet_mean_density > 0.0,
                  ExcMessage("Planet mean density must be positive."));
    }



    template <int dim>
    void
    SelfGravitation<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Self gravitation");
        {
          prm.declare_entry("Maximum degree", "40",
                            Patterns::Integer(0),
                            "Maximum spherical harmonic degree for the "
                            "self-gravitation calculation.");

          prm.declare_entry("Minimum degree", "0",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree for the "
                            "self-gravitation calculation.");

          prm.declare_entry("Density above surface", "0",
                            Patterns::Double(0),
                            "Density immediately above the deformed surface "
                            "boundary in kg/m^3. For a free surface in vacuum "
                            "or thin atmosphere, set to 0. For a seafloor "
                            "under ocean, set to water density (e.g., 1030).");

          prm.declare_entry("Density below surface", "3500",
                            Patterns::Double(0),
                            "Density immediately below the deformed surface "
                            "boundary in kg/m^3. For rock topography, use "
                            "crustal density (e.g., 3500). For an ice cap "
                            "sitting on rock, use ice density (e.g., 917).");

          prm.declare_entry("Density above CMB", "5500",
                            Patterns::Double(0),
                            "Density immediately above the CMB (lower mantle side) "
                            "in kg/m^3. Earth: ~5500, Mars: ~3800.");

          prm.declare_entry("Density below CMB", "9900",
                            Patterns::Double(0),
                            "Density immediately below the CMB (outer core side) "
                            "in kg/m^3. Earth: ~9900, Mars: ~6200.");

          prm.declare_entry("Planet mean density", "5515",
                            Patterns::Double(0),
                            "Mean density of the planet in kg/m^3. "
                            "Earth: 5515, Mars: 3390.");

          prm.declare_entry("Include CMB contribution", "true",
                            Patterns::Bool(),
                            "Whether to include the CMB topography contribution "
                            "to the self-gravitational potential perturbation. "
                            "Set to false if only surface topography feedback is needed.");

          prm.declare_entry("Iterate with Stokes", "true",
                            Patterns::Bool(),
                            "Recompute the non-local surface/CMB potential from "
                            "the current Stokes velocity after every Stokes solve. "
                            "The updated traction is used by the next nonlinear "
                            "iteration in the same time step.");
          prm.declare_entry("Freeze potential after timestep zero", "false",
                            Patterns::Bool(),
                            "Diagnostic switch that retains the converged "
                            "timestep-zero non-local potential coefficients "
                            "without recomputing them at later timesteps.");

          prm.declare_entry("Initial displacement time step", "0",
                            Patterns::Double(0),
                            "Displacement interval used to convert the timestep-0 "
                            "Stokes velocity into an incremental boundary displacement. "
                            "Set this to the elastic time step for an instantaneously "
                            "applied load. Units are years when 'Use years instead of "
                            "seconds' is enabled, otherwise seconds.");
          prm.declare_entry("Potential convergence tolerance", "1e-3",
                            Patterns::Double(0),
                            "Relative L2 change tolerance for the combined "
                            "surface and CMB Phi/g spherical-harmonic "
                            "coefficient vectors. Zhong et al. (2022) author "
                            "inputfile10 uses 1e-3 for its self-gravity "
                            "iteration cutoff.");
          prm.declare_entry("Maximum potential iterations", "10",
                            Patterns::Integer(1),
                            "Maximum number of self-consistent potential "
                            "updates per timestep. The iteration stops when "
                            "the potential coefficient change reaches the "
                            "tolerance or this limit is reached.");
          prm.declare_entry("Enable surface potential traction", "true",
                            Patterns::Bool(),
                            "Diagnostic switch controlling whether Phi/g is "
                            "applied as a non-local traction at the outer "
                            "surface. Harmonic analysis and output remain "
                            "active when this switch is false.");
          prm.declare_entry("Enable CMB potential traction", "true",
                            Patterns::Bool(),
                            "Diagnostic switch controlling whether Phi/g is "
                            "applied as a non-local traction at the CMB. The "
                            "local CMB topography term is unaffected.");
          prm.enter_subsection("Tidal potential");
          {
            TidalPotential::declare_parameters(prm);
          }
          prm.leave_subsection();
          prm.enter_subsection("Applied potential");
          {
            TidalPotential::declare_parameters(prm);
          }
          prm.leave_subsection();
          prm.declare_entry("Time between text output", "0.",
                            Patterns::Double(0.),
                            "The time interval in years between text outputs for self-gravity diagnostics. "
                            "If zero, this parameter is ignored.");
          prm.declare_entry("Time steps between text output", "0",
                            Patterns::Integer(0),
                            "The number of time steps between self-gravity diagnostic text outputs. "
                            "If zero, this parameter is ignored. If both output interval parameters are zero, no self-gravity diagnostic text is printed.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    SelfGravitation<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Self gravitation");
        {
          max_degree = prm.get_integer("Maximum degree");
          min_degree = prm.get_integer("Minimum degree");
          density_above_surface = prm.get_double("Density above surface");
          density_below_surface = prm.get_double("Density below surface");
          density_above_cmb = prm.get_double("Density above CMB");
          density_below_cmb = prm.get_double("Density below CMB");
          planet_mean_density = prm.get_double("Planet mean density");
          include_cmb_contribution = prm.get_bool("Include CMB contribution");
          iterate_with_stokes = prm.get_bool("Iterate with Stokes");
          freeze_potential_after_timestep_zero =
            prm.get_bool("Freeze potential after timestep zero");
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
          prm.enter_subsection("Tidal potential");
          {
            tidal_potential.parse_parameters(prm,
                                             min_degree,
                                             max_degree,
                                             dim);
          }
          prm.leave_subsection();
          prm.enter_subsection("Applied potential");
          {
            if (prm.get_bool("Enable"))
              tidal_potential.parse_parameters(prm,
                                               min_degree,
                                               max_degree,
                                               dim);
          }
          prm.leave_subsection();
          time_between_text_output = prm.get_double("Time between text output");
          time_steps_between_text_output = prm.get_integer("Time steps between text output");
          potential_relative_change = std::numeric_limits<double>::infinity();
          current_potential_iteration_step = (unsigned int)-1;
          potential_iteration_number = 0;

          if (this->convert_output_to_years())
            {
              initial_displacement_timestep *= year_in_seconds;
              time_between_text_output *= year_in_seconds;
            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(min_degree <= max_degree,
                  ExcMessage("Minimum degree must not exceed Maximum degree."));
      AssertThrow(planet_mean_density > 0.0,
                  ExcMessage("Planet mean density must be positive."));
    }
  }
}


// Explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    template class SelfGravitation<2>;
    template class SelfGravitation<3>;
  }
}
