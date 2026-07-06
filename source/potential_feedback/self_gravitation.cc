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

#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/simulator.h>
#include <aspect/postprocess/boundary_densities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <numeric>
#include <set>

namespace aspect
{
  namespace PotentialFeedback
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
      selected_load_source_contains(
        const std::map<std::string, std::vector<std::string>> &selection,
        const std::string &boundary_name,
        const std::string &plugin_name)
      {
        const auto boundary = selection.find(boundary_name);
        if (boundary == selection.end())
          return false;

        return std::find(boundary->second.begin(),
                         boundary->second.end(),
                         plugin_name) != boundary->second.end();
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

      const double mm_initial_elastic_dt =
        this->get_material_model().initial_elastic_time_step();
      if (mm_initial_elastic_dt > 0.0 && initial_displacement_timestep == 0.0)
        initial_displacement_timestep = mm_initial_elastic_dt;

      if (configured_from_potential_feedback)
        {
          enable_surface_potential_traction = true;
          enable_cmb_potential_traction = true;
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
      const double delta_rho_cmb = density_below_cmb - density_above_cmb;

      const auto &traction_manager =
        this->get_boundary_traction_manager();
      const auto &plugins = traction_manager.get_active_plugins();
      const auto &plugin_boundaries =
        traction_manager.get_active_plugin_boundary_indicators();
      const auto &plugin_names = traction_manager.get_active_plugin_names();

      const auto load_traction_on_boundary =
        [&plugins,
         &plugin_boundaries,
         &plugin_names,
         this]
        (const types::boundary_id boundary_id,
         const Point<dim> &position,
         const Tensor<1,dim> &face_normal,
         const std::string &boundary_name)
      {
        Tensor<1,dim> load_traction;
        if (external_load_source == "none")
          return load_traction;

        unsigned int plugin_index = 0;
        for (const auto &plugin : plugins)
          {
            const bool plugin_on_boundary =
              plugin_boundaries[plugin_index] == boundary_id;
            const bool is_feedback_plugin =
              dynamic_cast<const SelfGravitation<dim> *>(plugin.get()) != nullptr
              ||
              dynamic_cast<const PotentialFeedback::BoundaryTractionMarker *>(plugin.get()) != nullptr;

            bool use_plugin = false;
            if (plugin_on_boundary && !is_feedback_plugin)
              {
                if (external_load_source == "auto")
                  use_plugin = plugin->is_potential_feedback_load_source();
                else if (external_load_source == "selected")
                  use_plugin = selected_load_source_contains(
                                 selected_external_load_traction_indicators,
                                 boundary_name,
                                 plugin_names[plugin_index]);
                else
                  AssertThrow(false, ExcInternalError());
              }

            if (use_plugin)
              load_traction += plugin->boundary_traction(boundary_id,
                                                         position,
                                                         face_normal);

            ++plugin_index;
          }

        return load_traction;
      };

      // Surface topography data
      std::vector<double> phi_pts;
      std::vector<double> theta_pts; // only used in 3D
      std::vector<double> weight_pts;
      std::vector<double> topo_pts;
      std::vector<double> surface_deformation_topo_pts;
      std::vector<double> external_load_topo_pts;

      // CMB topography data
      std::vector<double> cmb_phi_pts;
      std::vector<double> cmb_theta_pts; // only used in 3D
      std::vector<double> cmb_weight_pts;
      std::vector<double> cmb_topo_pts;
      std::vector<double> cmb_deformation_topo_pts;
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
                  const bool is_top    = (bid == top_boundary_id)
                                         && include_surface_contribution;
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
                          const Tensor<1,dim> load_traction =
                            load_traction_on_boundary(top_boundary_id,
                                                      position,
                                                      face_normal,
                                                      "surface");

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
                          surface_deformation_topo_pts.push_back(h_rock);
                          external_load_topo_pts.push_back(h_load);
                        }
                      else // is_bottom
                        {
                          const double r = scoord[0];
                          const double committed_cmb_topography = r - inner_radius;
                          const double cmb_topography =
                            committed_cmb_topography + predicted_radial_displacement;

                          const Tensor<1,dim> face_normal =
                            fe_face_values.normal_vector(q);
                          const Tensor<1,dim> load_traction =
                            load_traction_on_boundary(bottom_boundary_id,
                                                      position,
                                                      face_normal,
                                                      "CMB");

                          const double g_magnitude =
                            this->get_gravity_model().gravity_vector(position).norm();

                          double h_cmb_load = 0.0;
                          if (g_magnitude > 0 && delta_rho_cmb > 0)
                            h_cmb_load = (load_traction * face_normal) /
                                         (delta_rho_cmb * g_magnitude);

                          const double cmb_effective_topography =
                            cmb_topography + h_cmb_load;
                          const double ref_radius = inner_radius;
                          const double w =
                            fe_face_values.JxW(q) /
                            (dim == 3 ? ref_radius *ref_radius : ref_radius);

                          cmb_phi_pts.push_back(ph);
                          if (dim == 3)
                            cmb_theta_pts.push_back(scoord[2]);
                          cmb_weight_pts.push_back(w);
                          cmb_topo_pts.push_back(cmb_effective_topography);
                          cmb_deformation_topo_pts.push_back(cmb_topography);
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

      if (dim == 3)
        {
          auto [cos_topo, sin_topo] = sh_transform->analyze(
                                        theta_pts, phi_pts, weight_pts, topo_pts,
                                        this->get_mpi_communicator());
          const unsigned int n_coeff = sh_transform->n_coefficients();
          auto [cos_surface_deformation, sin_surface_deformation] =
            sh_transform->analyze(theta_pts,
                                  phi_pts,
                                  weight_pts,
                                  surface_deformation_topo_pts,
                                  this->get_mpi_communicator());
          auto [cos_external_load, sin_external_load] =
            sh_transform->analyze(theta_pts,
                                  phi_pts,
                                  weight_pts,
                                  external_load_topo_pts,
                                  this->get_mpi_communicator());

          std::vector<double> cos_cmb(n_coeff, 0.0);
          std::vector<double> sin_cmb(n_coeff, 0.0);
          std::vector<double> cos_cmb_deformation(n_coeff, 0.0);
          std::vector<double> sin_cmb_deformation(n_coeff, 0.0);
          // analyze() performs MPI collectives, so every rank must call it.
          // Ranks without locally owned CMB faces contribute empty vectors,
          // which correctly produce a zero local contribution.
          if (include_cmb_contribution)
            {
              std::tie(cos_cmb, sin_cmb) = sh_transform->analyze(
                                             cmb_theta_pts, cmb_phi_pts,
                                             cmb_weight_pts, cmb_topo_pts,
                                             this->get_mpi_communicator());
              std::tie(cos_cmb_deformation, sin_cmb_deformation) =
                sh_transform->analyze(cmb_theta_pts,
                                      cmb_phi_pts,
                                      cmb_weight_pts,
                                      cmb_deformation_topo_pts,
                                      this->get_mpi_communicator());
              std::tie(cmb_committed_topography_cos_coeffs,
                       cmb_committed_topography_sin_coeffs) =
                         sh_transform->analyze(
                           cmb_theta_pts, cmb_phi_pts, cmb_weight_pts,
                           cmb_committed_topo_pts, this->get_mpi_communicator());
            }

          if (!self_gravity_mass_feedback_enabled)
            {
              std::fill(cos_topo.begin(), cos_topo.end(), 0.0);
              std::fill(sin_topo.begin(), sin_topo.end(), 0.0);
              std::fill(cos_surface_deformation.begin(),
                        cos_surface_deformation.end(),
                        0.0);
              std::fill(sin_surface_deformation.begin(),
                        sin_surface_deformation.end(),
                        0.0);
              std::fill(cos_external_load.begin(),
                        cos_external_load.end(),
                        0.0);
              std::fill(sin_external_load.begin(),
                        sin_external_load.end(),
                        0.0);
              std::fill(cos_cmb.begin(), cos_cmb.end(), 0.0);
              std::fill(sin_cmb.begin(), sin_cmb.end(), 0.0);
              std::fill(cos_cmb_deformation.begin(),
                        cos_cmb_deformation.end(),
                        0.0);
              std::fill(sin_cmb_deformation.begin(),
                        sin_cmb_deformation.end(),
                        0.0);
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
          external_load_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          external_load_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          surface_deformation_mass_potential_cos_coeffs =
            surface_mass_potential_cos_coeffs;
          surface_deformation_mass_potential_sin_coeffs =
            surface_mass_potential_sin_coeffs;

          external_load_surface_potential_cos_coeffs = cos_external_load;
          external_load_surface_potential_sin_coeffs = sin_external_load;
          sh_transform->apply_degree_filter(
            external_load_surface_potential_cos_coeffs,
            external_load_surface_potential_sin_coeffs,
            surface_to_surface);

          surface_deformation_mass_potential_cos_coeffs =
            cos_surface_deformation;
          surface_deformation_mass_potential_sin_coeffs =
            sin_surface_deformation;
          sh_transform->apply_degree_filter(
            surface_deformation_mass_potential_cos_coeffs,
            surface_deformation_mass_potential_sin_coeffs,
            surface_to_surface);

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
          reference_frame_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_cos_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_sin_coeffs.assign(n_coeff, 0.0);
          degree_one_load_replay_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          degree_one_load_replay_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          citcomsve_degree_one_load_replay_diagnostic =
            CitcomSVEDegreeOneLoadReplayDiagnostic();

          tidal_potential.add_to_coefficients(
            *sh_transform,
            radius_ratio,
            this->get_time(),
            surface_potential_cos_coeffs,
            surface_potential_sin_coeffs,
            cmb_potential_cos_coeffs,
            cmb_potential_sin_coeffs,
            tidal_surface_potential_cos_coeffs,
            tidal_surface_potential_sin_coeffs,
            tidal_cmb_potential_cos_coeffs,
            tidal_cmb_potential_sin_coeffs);

          cm_displacement_increment = Tensor<1,dim>();
          deformation_cm_displacement_increment = Tensor<1,dim>();
          external_load_cm_displacement_increment = Tensor<1,dim>();
          surface_deformation_cm_displacement_increment = Tensor<1,dim>();
          cmb_deformation_cm_displacement_increment = Tensor<1,dim>();

          reference_frame_acceleration = Tensor<1,dim>();
          if ((center_of_mass_correction
               || citcomsve_degree_one_load_compensation)
              && min_degree <= 1 && max_degree >= 1)
            {
              // With normalized real degree-1 harmonics, the center-of-mass
              // displacement associated with a Phi/g coefficient is Phi_1m/g
              // times sqrt(3/(4*pi)). This reproduces CitcomSVE's initial
              // l=1 load CM: a 6.37 m surface load gives about 2.604 m of
              // CM_z.
              const unsigned int idx10 = sh_transform->index(1, 0);
              const unsigned int idx11 = sh_transform->index(1, 1);
              const double y1_normalization =
                std::sqrt(3.0 / (4.0 * numbers::PI));

              const auto potential_to_cm =
                [idx10, idx11, y1_normalization]
                (const std::vector<double> &cos_coeffs,
                 const std::vector<double> &sin_coeffs)
              {
                Tensor<1,dim> result;
                result[0] = -cos_coeffs[idx11] * y1_normalization;
                result[1] = -sin_coeffs[idx11] * y1_normalization;
                result[2] =  cos_coeffs[idx10] * y1_normalization;
                return result;
              };

              cm_displacement_increment =
                potential_to_cm(surface_potential_cos_coeffs,
                                surface_potential_sin_coeffs);
              external_load_cm_displacement_increment =
                potential_to_cm(external_load_surface_potential_cos_coeffs,
                                external_load_surface_potential_sin_coeffs);
              surface_deformation_cm_displacement_increment =
                potential_to_cm(surface_deformation_mass_potential_cos_coeffs,
                                surface_deformation_mass_potential_sin_coeffs);
              cmb_deformation_cm_displacement_increment =
                potential_to_cm(cmb_mass_potential_cos_coeffs,
                                cmb_mass_potential_sin_coeffs);

              if (citcomsve_degree_one_load_compensation)
                {
                  const double compensation_scale = 1.0;

                  degree_one_load_compensation_cos_coeffs[idx10] =
                    -compensation_scale
                    * external_load_surface_potential_cos_coeffs[idx10];
                  degree_one_load_compensation_cos_coeffs[idx11] =
                    -compensation_scale
                    * external_load_surface_potential_cos_coeffs[idx11];
                  degree_one_load_compensation_sin_coeffs[idx11] =
                    -compensation_scale
                    * external_load_surface_potential_sin_coeffs[idx11];

                  for (unsigned int order = 0; order <= 1; ++order)
                    {
                      const unsigned int index = sh_transform->index(1, order);
                      degree_one_load_replay_cmb_potential_cos_coeffs[index] =
                        cmb_potential_cos_coeffs[index]
                        + (surface_to_cmb[1] + cmb_to_cmb[1])
                        * degree_one_load_compensation_cos_coeffs[index];
                      degree_one_load_replay_cmb_potential_sin_coeffs[index] =
                        cmb_potential_sin_coeffs[index]
                        + (surface_to_cmb[1] + cmb_to_cmb[1])
                        * degree_one_load_compensation_sin_coeffs[index];
                    }

                  citcomsve_degree_one_load_replay_diagnostic.valid = true;
                  citcomsve_degree_one_load_replay_diagnostic
                  .original_surface_load_height_10 = cos_external_load[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_external_10_over_g =
                    external_load_surface_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.citcomsve_cm_z =
                    external_load_surface_potential_cos_coeffs[idx10]
                    * y1_normalization;
                  citcomsve_degree_one_load_replay_diagnostic.citcomsve_h_comp_10 =
                    degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .corrected_surface_load_height_10 =
                    cos_external_load[idx10]
                    + degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .corrected_cmb_load_height_10 =
                    degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.surface_kernel_l1 =
                    surface_to_surface[1];
                  citcomsve_degree_one_load_replay_diagnostic.cmb_kernel_l1 =
                    cmb_to_surface[1];
                  citcomsve_degree_one_load_replay_diagnostic
                  .net_degree1_phi_over_g_after_load_compensation =
                    external_load_surface_potential_cos_coeffs[idx10]
                    + (surface_to_surface[1] + cmb_to_surface[1])
                    * degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_pre_cancellation_over_g_10 =
                    cmb_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.h_comp_10 =
                    degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .surface_deformation_topo_cos_10 =
                    cos_surface_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_deformation_topo_cos_10 =
                    cos_cmb_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.surface_to_cmb_l1 =
                    surface_to_cmb[1];
                  citcomsve_degree_one_load_replay_diagnostic.cmb_to_cmb_l1 =
                    cmb_to_cmb[1];
                  citcomsve_degree_one_load_replay_diagnostic
                  .surface_deformation_to_cmb_phi_over_g_10 =
                    surface_to_cmb[1] * cos_surface_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_deformation_to_cmb_phi_over_g_10 =
                    cmb_to_cmb[1] * cos_cmb_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .original_surface_load_to_cmb_l1_times_height =
                    surface_to_cmb[1] * cos_external_load[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .surface_to_cmb_l1_times_h_comp =
                    surface_to_cmb[1]
                    * degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_to_cmb_l1_times_h_comp =
                    cmb_to_cmb[1]
                    * degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_deformation_pre_compensation_over_g_10 =
                    citcomsve_degree_one_load_replay_diagnostic
                    .surface_deformation_to_cmb_phi_over_g_10
                    + citcomsve_degree_one_load_replay_diagnostic
                    .cmb_deformation_to_cmb_phi_over_g_10;
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_initial_load_pair_replay_over_g_10 =
                    citcomsve_degree_one_load_replay_diagnostic
                    .original_surface_load_to_cmb_l1_times_height
                    + citcomsve_degree_one_load_replay_diagnostic
                    .surface_to_cmb_l1_times_h_comp
                    + citcomsve_degree_one_load_replay_diagnostic
                    .cmb_to_cmb_l1_times_h_comp;
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_replay_over_g_10 =
                    degree_one_load_replay_cmb_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_intermediate_compensation_rhs_10 =
                    -degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_potential_append_rhs_10 =
                    degree_one_load_replay_cmb_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_final_rhs_10 =
                    citcomsve_degree_one_load_replay_diagnostic
                    .cmb_intermediate_compensation_rhs_10
                    + citcomsve_degree_one_load_replay_diagnostic
                    .cmb_potential_append_rhs_10;

                  Utilities::create_directory(
                    this->get_output_directory() + "self_gravity/",
                    this->get_mpi_communicator(),
                    /*silent=*/true);

                  if (Utilities::MPI::this_mpi_process(
                        this->get_mpi_communicator()) == 0)
                    {
                      const std::string filename =
                        this->get_output_directory()
                        + "self_gravity/"
                        + "citcomsve_degree1_load_replay_diagnostic."
                        + Utilities::int_to_string(this->get_timestep_number(), 5);
                      std::ofstream output(filename);
                      output << "# CitcomSVE incompressible l=1,m=0 "
                             << "initial-load center-of-mass replay diagnostic\n";
                      output << "# timestep: " << this->get_timestep_number()
                             << "\n";
                      output << "# potential_iteration: "
                             << potential_iteration_number << "\n";
                      output << "# include_current_velocity_increment: "
                             << include_current_velocity_increment << "\n";
                      output << "# deformation_topography_reference_frame: "
                             << "solution-frame displacement before "
                             << "degree-1 reference-frame potential "
                             << "cancellation; surface excludes external load "
                             << "height and CMB excludes non-self-gravity "
                             << "CMB load height.\n";
                      output
                          << "original_surface_load_height_10 "
                          << "phi_external_10_over_g "
                          << "citcomsve_cm_z "
                          << "citcomsve_h_comp_10 "
                          << "corrected_surface_load_height_10 "
                          << "corrected_cmb_load_height_10 "
                          << "surface_kernel_l1 "
                          << "cmb_kernel_l1 "
                          << "net_degree1_phi_over_g_after_load_compensation "
                          << "phi_cmb_pre_cancellation_over_g_10 "
                          << "h_comp_10 "
                          << "surface_deformation_topo_cos_10 "
                          << "cmb_deformation_topo_cos_10 "
                          << "surface_to_cmb_l1 "
                          << "cmb_to_cmb_l1 "
                          << "surface_deformation_to_cmb_phi_over_g_10 "
                          << "cmb_deformation_to_cmb_phi_over_g_10 "
                          << "original_surface_load_to_cmb_l1_times_height "
                          << "surface_to_cmb_l1_times_h_comp "
                          << "cmb_to_cmb_l1_times_h_comp "
                          << "phi_cmb_deformation_pre_compensation_over_g_10 "
                          << "phi_cmb_initial_load_pair_replay_over_g_10 "
                          << "phi_cmb_replay_over_g_10 "
                          << "cmb_intermediate_compensation_rhs_10 "
                          << "cmb_potential_append_rhs_10 "
                          << "cmb_final_rhs_10\n";
                      output << std::setprecision(16)
                             << citcomsve_degree_one_load_replay_diagnostic
                             .original_surface_load_height_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .phi_external_10_over_g << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .citcomsve_cm_z << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .citcomsve_h_comp_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .corrected_surface_load_height_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .corrected_cmb_load_height_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .surface_kernel_l1 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_kernel_l1 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .net_degree1_phi_over_g_after_load_compensation
                             << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .phi_cmb_pre_cancellation_over_g_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .h_comp_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .surface_deformation_topo_cos_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_deformation_topo_cos_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .surface_to_cmb_l1 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_to_cmb_l1 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .surface_deformation_to_cmb_phi_over_g_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_deformation_to_cmb_phi_over_g_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .original_surface_load_to_cmb_l1_times_height
                             << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .surface_to_cmb_l1_times_h_comp << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_to_cmb_l1_times_h_comp << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .phi_cmb_deformation_pre_compensation_over_g_10
                             << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .phi_cmb_initial_load_pair_replay_over_g_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .phi_cmb_replay_over_g_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_intermediate_compensation_rhs_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_potential_append_rhs_10 << ' '
                             << citcomsve_degree_one_load_replay_diagnostic
                             .cmb_final_rhs_10
                             << '\n';
                    }
                }

              std::vector<double> deformation_potential_cos =
                surface_deformation_mass_potential_cos_coeffs;
              std::vector<double> deformation_potential_sin =
                surface_deformation_mass_potential_sin_coeffs;
              for (unsigned int i = 0; i < n_coeff; ++i)
                {
                  deformation_potential_cos[i] += cmb_mass_potential_cos_coeffs[i];
                  deformation_potential_sin[i] += cmb_mass_potential_sin_coeffs[i];
                }
              deformation_cm_displacement_increment =
                potential_to_cm(deformation_potential_cos,
                                deformation_potential_sin);

              // Zero degree-1 from the potential coefficients used for
              // boundary traction (geoid/k cancellation).  This is the
              // existing center_of_mass_correction: it ensures k1 = -1.
              if (center_of_mass_correction)
                {
                  for (unsigned int order = 0; order <= 1; ++order)
                    {
                      const unsigned int index = sh_transform->index(1, order);

                      reference_frame_surface_potential_cos_coeffs[index] =
                        -surface_potential_cos_coeffs[index];
                      reference_frame_surface_potential_sin_coeffs[index] =
                        -surface_potential_sin_coeffs[index];
                      reference_frame_cmb_potential_cos_coeffs[index] =
                        -cmb_potential_cos_coeffs[index];
                      reference_frame_cmb_potential_sin_coeffs[index] =
                        -cmb_potential_sin_coeffs[index];

                      surface_potential_cos_coeffs[index] = 0.0;
                      surface_potential_sin_coeffs[index] = 0.0;
                      cmb_potential_cos_coeffs[index] = 0.0;
                      cmb_potential_sin_coeffs[index] = 0.0;
                    }
                }

              const std::vector<double> theta = {numbers::PI / 2.0,
                                                 numbers::PI / 2.0,
                                                 0.0
                                                };
              const std::vector<double> phi = {0.0,
                                               numbers::PI / 2.0,
                                               0.0
                                              };
              const std::vector<double> height =
                sh_transform->synthesize(
                  reference_frame_surface_potential_cos_coeffs,
                  reference_frame_surface_potential_sin_coeffs,
                  theta,
                  phi);
              const double surface_gravity =
                this->get_gravity_model()
                .gravity_vector(geometry.representative_point(1.0)).norm();
              for (unsigned int d = 0; d < dim; ++d)
                reference_frame_acceleration[d] =
                  -surface_gravity * height[d] / outer_radius;
            }
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
          reference_frame_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_cos_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_acceleration = Tensor<1,dim>();
          cm_displacement_increment = Tensor<1,dim>();
          deformation_cm_displacement_increment = Tensor<1,dim>();
          external_load_cm_displacement_increment = Tensor<1,dim>();
          surface_deformation_cm_displacement_increment = Tensor<1,dim>();
          cmb_deformation_cm_displacement_increment = Tensor<1,dim>();
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
    SelfGravitation<dim>::external_load_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      const unsigned int index = sh_transform->index(degree, order);
      return {external_load_surface_potential_cos_coeffs.at(index),
              external_load_surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::surface_deformation_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      const unsigned int index = sh_transform->index(degree, order);
      return {surface_deformation_mass_potential_cos_coeffs.at(index),
              surface_deformation_mass_potential_sin_coeffs.at(index)
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
    std::pair<double,double>
    SelfGravitation<dim>::reference_frame_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (reference_frame_surface_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {reference_frame_surface_potential_cos_coeffs.at(index),
              reference_frame_surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::reference_frame_body_force(
      const Point<dim> &position) const
    {
      (void)position;
      return reference_frame_acceleration;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_cm_displacement_increment() const
    {
      return cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_deformation_cm_displacement_increment() const
    {
      return deformation_cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_external_load_cm_displacement_increment() const
    {
      return external_load_cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_surface_deformation_cm_displacement_increment() const
    {
      return surface_deformation_cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_cmb_deformation_cm_displacement_increment() const
    {
      return cmb_deformation_cm_displacement_increment;
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
    bool
    SelfGravitation<dim>::has_citcomsve_degree_one_load_replay_diagnostic() const
    {
      return citcomsve_degree_one_load_replay_diagnostic.valid;
    }


    template <int dim>
    double
    SelfGravitation<dim>::citcomsve_degree_one_cmb_intermediate_compensation_rhs_10() const
    {
      return citcomsve_degree_one_load_replay_diagnostic
             .cmb_intermediate_compensation_rhs_10;
    }


    template <int dim>
    double
    SelfGravitation<dim>::citcomsve_degree_one_cmb_potential_append_rhs_10() const
    {
      return citcomsve_degree_one_load_replay_diagnostic
             .cmb_potential_append_rhs_10;
    }


    template <int dim>
    double
    SelfGravitation<dim>::citcomsve_degree_one_cmb_final_rhs_10() const
    {
      return citcomsve_degree_one_load_replay_diagnostic.cmb_final_rhs_10;
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
      double degree_one_load_compensation_topography = 0.0;
      double degree_one_load_replay_cmb_potential_height = 0.0;
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

          if (citcomsve_degree_one_load_compensation
              && !degree_one_load_compensation_cos_coeffs.empty())
            degree_one_load_compensation_topography =
              sh_transform->synthesize(
                degree_one_load_compensation_cos_coeffs,
                degree_one_load_compensation_sin_coeffs,
                th_vec, ph_vec)[0];

          if (is_cmb
              && citcomsve_degree_one_load_compensation
              && !degree_one_load_replay_cmb_potential_cos_coeffs.empty())
            degree_one_load_replay_cmb_potential_height =
              sh_transform->synthesize(
                degree_one_load_replay_cmb_potential_cos_coeffs,
                degree_one_load_replay_cmb_potential_sin_coeffs,
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
                 * normal_vector
                 - density_below_surface * g_magnitude
                 * degree_one_load_compensation_topography
                 * normal_vector;
        }

      // Fluid-core CMB condition after subtracting the mantle hydrostatic
      // reference state: Delta rho * (g*h_b - Phi_b) n.
      return delta_rho_cmb * g_magnitude
             * (cmb_topography
                + (enable_cmb_potential_traction
                   ? -potential_height
                   : 0.0)
                + degree_one_load_compensation_topography
                - degree_one_load_replay_cmb_potential_height)
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
      include_surface_contribution =
        self_gravity_list_contains(settings.self_gravity_boundary_indicators,
                                   "surface");
      include_cmb_contribution =
        self_gravity_list_contains(settings.self_gravity_boundary_indicators,
                                   "CMB");
      self_gravity_mass_feedback_enabled =
        self_gravity_list_contains(settings.feedback_mechanisms,
                                   "self gravity");
      iterate_with_stokes = settings.iterate_with_stokes;
      freeze_potential_after_timestep_zero =
        settings.freeze_feedback_after_timestep_zero;
      initial_displacement_timestep =
        settings.initial_displacement_timestep;
      potential_convergence_tolerance = settings.relative_tolerance;
      potential_iteration_relaxation_factor =
        settings.potential_iteration_relaxation_factor;
      maximum_potential_iterations = settings.maximum_iterations;
      include_internal_density_anomalies = settings.include_internal_density_anomalies;
      reference_density_for_internal_anomalies = settings.reference_density_for_internal_anomalies;
      internal_density_anomaly_tolerance = settings.internal_density_anomaly_tolerance;
      center_of_mass_correction = settings.center_of_mass_correction;
      citcomsve_degree_one_load_compensation =
        settings.citcomsve_degree_one_load_compensation;
      external_load_source = settings.external_load_source;
      selected_external_load_traction_indicators =
        settings.selected_external_load_traction_indicators;
      tidal_potential.configure_from_settings(settings,
                                              min_degree,
                                              max_degree,
                                              dim);
      configured_from_potential_feedback = true;
      time_between_text_output =
        (settings.write_self_gravity_diagnostics
         || settings.write_coefficient_diagnostics
         ? settings.time_between_diagnostic_output
         : 0.0);
      time_steps_between_text_output =
        (settings.write_self_gravity_diagnostics
         || settings.write_coefficient_diagnostics
         ? settings.time_steps_between_diagnostic_output
         : 0);
      potential_relative_change = std::numeric_limits<double>::infinity();
      current_potential_iteration_step = (unsigned int)-1;
      potential_iteration_number = 0;

      AssertThrow(min_degree <= max_degree,
                  ExcMessage("Potential feedback/Self gravity/Minimum degree "
                             "must not exceed Maximum degree."));
      AssertThrow(planet_mean_density > 0.0,
                  ExcMessage("Planet mean density must be positive."));
      AssertThrow(include_surface_contribution || include_cmb_contribution,
                  ExcMessage("Potential feedback/Self gravity/Boundary "
                             "indicators must include at least one of the "
                             "surface or CMB aliases."));
      AssertThrow(!citcomsve_degree_one_load_compensation
                  || (min_degree <= 1 && max_degree >= 1),
                  ExcMessage("CitcomSVE degree 1 load compensation requires "
                             "degree 1 to be included in the self-gravity "
                             "spherical-harmonic range."));
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
          prm.declare_entry("Center of mass correction", "false",
                            Patterns::Bool(),
                            "Whether to apply the degree-1 center-of-mass "
                            "reference-frame correction. This correction only "
                            "affects degree 1 and is separate from ASPECT "
                            "nullspace removal.");
          prm.declare_entry("CitcomSVE degree 1 load compensation", "false",
                            Patterns::Bool(),
                            "Whether to apply the CitcomSVE-style degree-1 "
                            "center-of-mass compensating load before solving "
                            "the displacement response. This diagnostic option "
                            "is disabled by default and is separate from "
                            "ASPECT nullspace removal and from the degree-1 "
                            "geoid reference-frame correction.");
          prm.declare_entry("Include internal density anomalies", "auto",
                            Patterns::Selection("true|false|auto"),
                            "Whether to include the internal mantle density anomalies "
                            "contribution to the gravitational potential. Default is auto.");
          prm.declare_entry("Reference density for internal anomalies", "0",
                            Patterns::Double(),
                            "Reference density used to define mantle density anomalies (kg/m^3).");
          prm.declare_entry("Internal density anomaly tolerance", "0",
                            Patterns::Double(0),
                            "Density anomaly threshold below which the volume integral is skipped.");
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
          center_of_mass_correction =
            prm.get_bool("Center of mass correction");
          citcomsve_degree_one_load_compensation =
            prm.get_bool("CitcomSVE degree 1 load compensation");
          include_surface_contribution = true;
          self_gravity_mass_feedback_enabled = true;
          external_load_source = "auto";
          selected_external_load_traction_indicators.clear();
          potential_iteration_relaxation_factor = 1.0;
          include_internal_density_anomalies =
            prm.get("Include internal density anomalies");
          reference_density_for_internal_anomalies =
            prm.get_double("Reference density for internal anomalies");
          internal_density_anomaly_tolerance =
            prm.get_double("Internal density anomaly tolerance");
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
      AssertThrow(!citcomsve_degree_one_load_compensation
                  || (min_degree <= 1 && max_degree >= 1),
                  ExcMessage("Boundary traction model/Self gravitation/"
                             "CitcomSVE degree 1 load compensation requires "
                             "degree 1 to be included in the self-gravity "
                             "spherical-harmonic range."));
    }


    template <int dim>
    std::string
    SelfGravitation<dim>::get_include_internal_density_anomalies() const
    {
      return include_internal_density_anomalies;
    }

    template <int dim>
    double
    SelfGravitation<dim>::get_reference_density_for_internal_anomalies() const
    {
      return reference_density_for_internal_anomalies;
    }

    template <int dim>
    double
    SelfGravitation<dim>::get_internal_density_anomaly_tolerance() const
    {
      return internal_density_anomaly_tolerance;
    }

    template <int dim>
    std::pair<std::vector<double>, std::vector<double>>
    SelfGravitation<dim>::to_spherical_harmonic_coefficients(
      const std::vector<std::vector<double>> &spherical_function) const
    {
      std::vector<double> cosi(spherical_function.size(), 0.0);
      std::vector<double> sini(spherical_function.size(), 0.0);
      std::vector<double> coecos;
      std::vector<double> coesin;

      for (unsigned int ideg = min_degree; ideg < max_degree + 1; ++ideg)
        {
          for (unsigned int iord = 0; iord < ideg + 1; ++iord)
            {
              // Do the spherical harmonic expansion.
              for (unsigned int ds_num = 0; ds_num < spherical_function.size(); ++ds_num)
                {
                  // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                  const std::pair<double, double> sph_harm_vals =
                    aspect::Utilities::real_spherical_harmonic(ideg, iord, spherical_function.at(ds_num).at(0), spherical_function.at(ds_num).at(1));
                  const double cos_component = sph_harm_vals.first;
                  const double sin_component = sph_harm_vals.second;

                  cosi.at(ds_num) = (spherical_function.at(ds_num).at(3) * cos_component);
                  sini.at(ds_num) = (spherical_function.at(ds_num).at(3) * sin_component);
                }
              // Integrate the contribution of each spherical infinitesimal.
              double cosii = 0;
              double sinii = 0;
              for (unsigned int ds_num = 0; ds_num < spherical_function.size(); ++ds_num)
                {
                  cosii += cosi.at(ds_num) * spherical_function.at(ds_num).at(2);
                  sinii += sini.at(ds_num) * spherical_function.at(ds_num).at(2);
                }
              coecos.push_back(cosii);
              coesin.push_back(sinii);
            }
        }
      // Sum over each processor.
      dealii::Utilities::MPI::sum (coecos, this->get_mpi_communicator(), coecos);
      dealii::Utilities::MPI::sum (coesin, this->get_mpi_communicator(), coesin);

      return std::make_pair(coecos, coesin);
    }

    template <int dim>
    std::pair<std::vector<double>, std::vector<double>>
    SelfGravitation<dim>::compute_internal_density_potential(const double /*outer_radius*/) const
    {
      AssertThrow(false, ExcNotImplemented());
      return std::make_pair(std::vector<double>(), std::vector<double>());
    }

    template <>
    std::pair<std::vector<double>, std::vector<double>>
    SelfGravitation<3>::compute_internal_density_potential(const double outer_radius) const
    {
      unsigned int n_coefficients = 0;
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
        n_coefficients += degree + 1;

      std::vector<double> SH_density_coecos(n_coefficients, 0.0);
      std::vector<double> SH_density_coesin(n_coefficients, 0.0);

      // Map "auto" to either true or false depending on whether there are temperature or compositional fields
      bool actual_include_internal = false;
      if (include_internal_density_anomalies == "true")
        actual_include_internal = true;
      else if (include_internal_density_anomalies == "false")
        actual_include_internal = false;
      else if (include_internal_density_anomalies == "auto")
        {
          actual_include_internal = (this->introspection().n_compositional_fields > 0 ||
                                     this->introspection().variable_exists("temperature"));
        }

      if (!actual_include_internal)
        {
          return std::make_pair(SH_density_coecos, SH_density_coesin);
        }

      const unsigned int quadrature_degree = this->introspection().polynomial_degree.temperature;

      // Need to evaluate density contribution of each volume quadrature point.
      const QGauss<3> quadrature_formula(quadrature_degree);

      FEValues<3> fe_values(this->get_mapping(),
                            this->get_fe(),
                            quadrature_formula,
                            update_values |
                            update_quadrature_points |
                            update_JxW_values |
                            update_gradients);

      MaterialModel::MaterialModelInputs<3> in(fe_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<3> out(fe_values.n_quadrature_points, this->n_compositional_fields());
      in.requested_properties = MaterialModel::MaterialProperties::density;

      const double effective_tolerance =
        (internal_density_anomaly_tolerance > 0.0
         ? internal_density_anomaly_tolerance
         : 1e-12 * std::max(1.0, std::abs(reference_density_for_internal_anomalies)));

      if (include_internal_density_anomalies == "auto")
        {
          double local_max_density_anomaly = 0.0;

          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                in.reinit(fe_values, cell, this->introspection(), this->get_solution());
                this->get_material_model().evaluate(in, out);

                for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                  local_max_density_anomaly =
                    std::max(local_max_density_anomaly,
                             std::abs(out.densities[q] - reference_density_for_internal_anomalies));
              }

          const double global_max_density_anomaly =
            Utilities::MPI::max(local_max_density_anomaly,
                                this->get_mpi_communicator());

          if (global_max_density_anomaly <= effective_tolerance)
            {
              return std::make_pair(SH_density_coecos, SH_density_coesin);
            }
        }

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            in.reinit(fe_values, cell, this->introspection(), this->get_solution());

            this->get_material_model().evaluate(in, out);

            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              {
                const double density_anomaly = out.densities[q] - reference_density_for_internal_anomalies;

                if (density_anomaly == 0.0)
                  continue;

                const std::array<double, 3> scoord =
                  aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(in.position[q]);
                const double r_q = in.position[q].norm();
                const double JxW = fe_values.JxW(q);

                unsigned int coefficient_index = 0;
                for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                  {
#if DEAL_II_VERSION_GTE(9,6,0)
                    const double radial_kernel =
                      (1.0 / r_q) * Utilities::pow(r_q / outer_radius, degree + 1);
#else
                    const double radial_kernel =
                      (1.0 / r_q) * std::pow(r_q / outer_radius, degree + 1);
#endif

                    for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                      {
                        const std::pair<double, double> sph_harm_vals =
                          aspect::Utilities::real_spherical_harmonic(
                            degree, order, scoord[2], scoord[1]);

                        const double weighted_density =
                          density_anomaly * radial_kernel * JxW;
                        SH_density_coecos[coefficient_index] +=
                          weighted_density * sph_harm_vals.first;
                        SH_density_coesin[coefficient_index] +=
                          weighted_density * sph_harm_vals.second;
                      }
                  }
              }
          }

      dealii::Utilities::MPI::sum(SH_density_coecos, this->get_mpi_communicator(), SH_density_coecos);
      dealii::Utilities::MPI::sum(SH_density_coesin, this->get_mpi_communicator(), SH_density_coesin);

      return std::make_pair(SH_density_coecos, SH_density_coesin);
    }

    template <int dim>
    std::pair<std::pair<double, std::pair<std::vector<double>, std::vector<double>>>, std::pair<double, std::pair<std::vector<double>, std::vector<double>>>>
    SelfGravitation<dim>::compute_topography_potential(const double /*outer_radius*/, const double /*inner_radius*/) const
    {
      AssertThrow(false, ExcNotImplemented());
      std::pair<double, std::pair<std::vector<double>, std::vector<double>>> temp;
      return std::make_pair(temp, temp);
    }

    template <>
    std::pair<std::pair<double, std::pair<std::vector<double>, std::vector<double>>>, std::pair<double, std::pair<std::vector<double>, std::vector<double>>>>
    SelfGravitation<3>::compute_topography_potential(const double outer_radius, const double inner_radius) const
    {
      const Postprocess::BoundaryDensities<3> &boundary_densities =
        this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::BoundaryDensities<3>>();

      const double top_layer_average_density = boundary_densities.density_at_top();
      const double bottom_layer_average_density = boundary_densities.density_at_bottom();

      const types::boundary_id top_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      const unsigned int quadrature_degree = this->introspection().polynomial_degree.temperature;
      const QGauss<2> quadrature_formula_face(quadrature_degree);

      FEFaceValues<3> fe_face_values(this->get_mapping(),
                                     this->get_fe(),
                                     quadrature_formula_face,
                                     update_values |
                                     update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

      std::vector<std::pair<Point<3>, std::pair<double, double>>> surface_stored_values;
      std::vector<std::pair<Point<3>, std::pair<double, double>>> CMB_stored_values;

      const bool use_free_surface_topography = true;
      const bool use_free_CMB_topography = include_cmb_contribution;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            unsigned int face_idx = numbers::invalid_unsigned_int;
            bool at_upper_surface = false;
            {
              for (const unsigned int f : cell->face_indices())
                {
                  if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
                    {
                      face_idx = f;
                      at_upper_surface = true;
                      break;
                    }
                  else if (cell->at_boundary(f) && cell->face(f)->boundary_id() == bottom_boundary_id)
                    {
                      face_idx = f;
                      at_upper_surface = false;
                      break;
                    }
                }
              if (face_idx == numbers::invalid_unsigned_int)
                continue;
            }

            fe_face_values.reinit(cell, face_idx);

            if (at_upper_surface)
              {
                if (use_free_surface_topography)
                  {
                    const auto &boundary_traction_manager = this->get_boundary_traction_manager();
                    const std::set<types::boundary_id> &prescribed_traction_boundary_indicators =
                      boundary_traction_manager.get_prescribed_boundary_traction_indicators();
                    const bool has_active_boundary_traction = (boundary_traction_manager.get_active_plugins().empty() == false);

                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        const Point<3> current_position = fe_face_values.quadrature_point(q);
                        double topography = this->get_geometry_model().height_above_reference_surface(current_position);

                        if (has_active_boundary_traction &&
                            prescribed_traction_boundary_indicators.find(cell->face(face_idx)->boundary_id()) !=
                            prescribed_traction_boundary_indicators.end())
                          {
                            const Tensor<1, 3> traction = boundary_traction_manager.boundary_traction(
                                                            cell->face(face_idx)->boundary_id(), fe_face_values.quadrature_point(q), fe_face_values.normal_vector(q));
                            const double normal_traction = traction * fe_face_values.normal_vector(q);

                            if (std::abs(normal_traction) > 1e-10)
                              {
                                const double gravity = this->get_gravity_model().gravity_vector(current_position).norm();
                                const double delta_rho = top_layer_average_density;
                                if (std::abs(delta_rho) > 0.0)
                                  topography -= normal_traction / (gravity * delta_rho);
                              }
                          }

                        surface_stored_values.emplace_back(current_position, std::make_pair(fe_face_values.JxW(q), topography));
                      }
                  }
                else
                  {
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        surface_stored_values.emplace_back(fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), 0.0));
                      }
                  }
              }

            if (at_upper_surface == false)
              {
                if (use_free_CMB_topography)
                  {
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        const Point<3> current_position = fe_face_values.quadrature_point(q);
                        const double topography = this->get_geometry_model().height_above_reference_surface(current_position) + (outer_radius - inner_radius);
                        CMB_stored_values.emplace_back(current_position, std::make_pair(fe_face_values.JxW(q), topography));
                      }
                  }
                else
                  {
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        CMB_stored_values.emplace_back(fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), 0.0));
                      }
                  }
              }
          }

      std::vector<std::vector<double>> surface_topo_spherical_function;
      std::vector<std::vector<double>> CMB_topo_spherical_function;

      for (const auto &surface_stored_value : surface_stored_values)
        {
          const std::array<double, 3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(surface_stored_value.first);
          const double infinitesimal = surface_stored_value.second.first / (outer_radius * outer_radius);
          surface_topo_spherical_function.emplace_back(std::vector<double> {scoord[2],
                                                                            scoord[1],
                                                                            infinitesimal,
                                                                            surface_stored_value.second.second
                                                                           });
        }

      for (const auto &CMB_stored_value : CMB_stored_values)
        {
          const std::array<double, 3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(CMB_stored_value.first);
          const double infinitesimal = CMB_stored_value.second.first / (inner_radius * inner_radius);
          CMB_topo_spherical_function.emplace_back(std::vector<double> {scoord[2],
                                                                        scoord[1],
                                                                        infinitesimal,
                                                                        CMB_stored_value.second.second
                                                                       });
        }

      std::pair<double, std::pair<std::vector<double>, std::vector<double>>> SH_surface_topo_coes
        = std::make_pair(top_layer_average_density, to_spherical_harmonic_coefficients(surface_topo_spherical_function));
      std::pair<double, std::pair<std::vector<double>, std::vector<double>>> SH_CMB_topo_coes
        = std::make_pair(bottom_layer_average_density, to_spherical_harmonic_coefficients(CMB_topo_spherical_function));

      return std::make_pair(SH_surface_topo_coes, SH_CMB_topo_coes);
    }
  }
}


// Explicit instantiations
namespace aspect
{
  namespace PotentialFeedback
  {
    template class SelfGravitation<2>;
    template class SelfGravitation<3>;
  }
}
