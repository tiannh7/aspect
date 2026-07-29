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

#include <aspect/potential_feedback/glacial_isostatic_adjustment.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace GIA
    {
      double
      barystatic_sea_level(
        const double ocean_area,
        const double ice_mass_change,
        const double relative_sea_level_volume,
        const double density_water)
      {
        AssertThrow(ocean_area > 0.0,
                    ExcMessage("The GIA ocean function contains no ocean "
                               "area at the current model time."));
        AssertThrow(density_water > 0.0,
                    ExcMessage("The GIA water density must be positive."));

        return (-ice_mass_change / density_water
                - relative_sea_level_volume)
               / ocean_area;
      }



      double
      sea_level_change(
        const double ocean_function,
        const double relative_geoid,
        const double barystatic_sea_level)
      {
        return ocean_function
               * (relative_geoid + barystatic_sea_level);
      }



      double
      ice_load_mass_density(
        const double current_ice_thickness,
        const double initial_ice_thickness,
        const double density_ice,
        const IceLoadReference reference)
      {
        AssertThrow(density_ice > 0.0,
                    ExcMessage("The GIA ice density must be positive."));

        if (reference == IceLoadReference::signed_anomaly)
          return density_ice * current_ice_thickness;

        const double current_thickness =
          std::max(0.0, current_ice_thickness);
        const double reference_thickness =
          (reference == IceLoadReference::first_history_file
           ? std::max(0.0, initial_ice_thickness)
           : 0.0);
        return density_ice * (current_thickness - reference_thickness);
      }



      double
      uniform_ocean_sea_level_coefficient(
        const double relative_sea_level_response,
        const double ice_load_coefficient,
        const double density_water)
      {
        AssertThrow(density_water > 0.0,
                    ExcMessage("The GIA water density must be positive."));

        const double denominator =
          1.0 - density_water * relative_sea_level_response;
        AssertThrow(std::abs(denominator)
                    > 100.0 * std::numeric_limits<double>::epsilon(),
                    ExcMessage("The uniform-ocean sea-level response is "
                               "singular because 1-rho_w*A_l is zero."));

        return relative_sea_level_response
               * ice_load_coefficient / denominator;
      }
    }



    namespace
    {
      bool
      feedback_list_contains(const std::vector<std::string> &values,
                             const std::string &name)
      {
        return std::find(values.begin(), values.end(), name) != values.end();
      }
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::
    configure_from_potential_feedback_settings(
      const PotentialFeedback::Settings &settings)
    {
      enabled = feedback_list_contains(settings.feedback_mechanisms,
                                       "glacial isostatic adjustment");
      ice_load_reference = settings.gia.ice_load_reference;
      ice_history_configuration = settings.gia.ice_history;
      ocean_history_configuration = settings.gia.ocean_history;
      density_ice = settings.gia.density_ice;
      density_water = settings.gia.density_water;
      maximum_degree = settings.gia.maximum_degree;
      diagnostic_degrees = settings.gia.diagnostic_degrees;
      absolute_coefficient_tolerance =
        settings.gia.absolute_coefficient_tolerance;
      projection_longitude_samples =
        settings.gia.projection_longitude_samples;
      projection_latitude_samples =
        settings.gia.projection_latitude_samples;
      iterate_with_stokes = settings.iterate_with_stokes;
      freeze_feedback_after_timestep_zero =
        settings.freeze_feedback_after_timestep_zero;
      initial_displacement_timestep = settings.initial_displacement_timestep;
      potential_convergence_tolerance = settings.relative_tolerance;
      potential_relaxation_factor =
        settings.potential_iteration_relaxation_factor;
      maximum_potential_iterations = settings.maximum_iterations;

      AssertThrow(!enabled || maximum_degree <= settings.self_gravity_max_degree,
                  ExcMessage("Potential feedback/Glacial isostatic "
                             "adjustment/Maximum degree must not exceed "
                             "Potential feedback/Self gravity/Maximum "
                             "degree."));
      AssertThrow(!enabled
                  || ((projection_longitude_samples == 0
                       && projection_latitude_samples == 0)
                      || (projection_longitude_samples >= 2
                          && projection_latitude_samples >= 2)),
                  ExcMessage("GIA history projection requires both sample "
                             "counts to be zero, or both to be at least two."));
      AssertThrow(!enabled || dim == 3
                  || projection_longitude_samples == 0,
                  ExcMessage("GIA history projection is implemented only in "
                             "three dimensions."));
      AssertThrow(!enabled
                  || (potential_relaxation_factor > 0.0
                      && potential_relaxation_factor <= 1.0),
                  ExcMessage("The potential iteration relaxation factor must "
                             "be in the interval (0,1] for GIA."));
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::set_surface_potential_height_function(
      const std::function<double(const Point<dim> &)> &function)
    {
      surface_potential_height_function = function;
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::initialize()
    {
      if (!enabled)
        return;

      AssertThrow(dim == 3,
                  ExcMessage("Glacial isostatic adjustment is implemented "
                             "only for 3D spherical-shell models."));
      AssertThrow(Plugins::plugin_type_matches<
                  const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Glacial isostatic adjustment requires a "
                             "spherical shell geometry."));
      AssertThrow(this->get_parameters().mesh_deformation_enabled,
                  ExcMessage("Glacial isostatic adjustment requires mesh "
                             "deformation on the top boundary."));
      AssertThrow(static_cast<bool>(surface_potential_height_function),
                  ExcMessage("The GIA surface potential provider was not "
                             "configured."));

      top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");

      ice_history.initialize_simulator(this->get_simulator());
      ice_history.configure(ice_history_configuration);
      ice_history.initialize(top_boundary_id);

      ocean_history.initialize_simulator(this->get_simulator());
      ocean_history.configure(ocean_history_configuration);
      ocean_history.initialize(top_boundary_id);

      sh_transform = std::make_unique<Utilities::SphericalHarmonicTransform>(
                       maximum_degree, 0);

      const double material_initial_timestep =
        this->get_material_model().initial_elastic_time_step();
      if (initial_displacement_timestep == 0.0
          && material_initial_timestep > 0.0)
        initial_displacement_timestep = material_initial_timestep;

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
        this->compute_surface_load(false);
      });
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::update()
    {
      if (!enabled
          || (freeze_feedback_after_timestep_zero
              && this->get_timestep_number() > 0))
        return;

      ice_history.update();
      ocean_history.update();

      compute_surface_load(false);
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::update_load_from_current_potential()
    {
      if (!enabled
          || (freeze_feedback_after_timestep_zero
              && this->get_timestep_number() > 0))
        return;

      compute_surface_load(false);
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::update_after_stokes_solve()
    {
      if (!freeze_feedback_after_timestep_zero
          || this->get_timestep_number() == 0)
        compute_surface_load(true);
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::compute_surface_load(
      const bool include_current_velocity_increment)
    {
      TimerOutput::Scope timer(this->get_computing_timer(),
                               "Potential feedback: GIA surface load");

      const unsigned int timestep_number = this->get_timestep_number();
      if (current_potential_iteration_step != timestep_number)
        {
          current_potential_iteration_step = timestep_number;
          potential_iteration_number = 0;
          potential_relative_change =
            std::numeric_limits<double>::infinity();
          potential_absolute_change =
            std::numeric_limits<double>::infinity();
        }
      if (include_current_velocity_increment)
        ++potential_iteration_number;

      const GeometryModel::SphericalShell<dim> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          this->get_geometry_model());
      const double outer_radius = geometry.outer_radius();
      double displacement_timestep = this->get_timestep();
      if (displacement_timestep == 0.0)
        displacement_timestep = initial_displacement_timestep;

      const unsigned int quadrature_degree =
        std::max(2u,
                 this->introspection().polynomial_degree.velocities + 1u);
      const QGauss<dim-1> face_quadrature(quadrature_degree);
      FEFaceValues<dim> face_values(this->get_mapping(),
                                    this->get_fe(),
                                    face_quadrature,
                                    update_quadrature_points |
                                    update_JxW_values);

      const auto &mesh_deformation_handler =
        this->get_mesh_deformation_handler();
      const DoFHandler<dim> &mesh_dof_handler =
        mesh_deformation_handler.get_mesh_deformation_dof_handler();
      FEFaceValues<dim> mesh_face_values(
        this->get_mapping(),
        mesh_dof_handler.get_fe(),
        face_quadrature,
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

      std::vector<double> theta;
      std::vector<double> phi;
      std::vector<double> weights;
      std::vector<double> current_ice_thickness;
      std::vector<double> initial_ice_thickness;
      std::vector<double> current_geoid;
      std::vector<double> current_displacement;
      std::vector<double> prescribed_ocean_values;

      auto mesh_cell = mesh_dof_handler.begin_active();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        {
          const auto current_mesh_cell = mesh_cell;
          ++mesh_cell;

          if (!cell->is_locally_owned() || !cell->at_boundary())
            continue;

          for (const unsigned int face_index : cell->face_indices())
            {
              if (!cell->at_boundary(face_index)
                  || cell->face(face_index)->boundary_id() != top_boundary_id)
                continue;

              face_values.reinit(cell, face_index);
              if (include_current_velocity_increment)
                Assert(projected_mesh_velocity != nullptr,
                       ExcInternalError());

              mesh_face_values.reinit(current_mesh_cell, face_index);
              mesh_face_values[mesh_velocity_extractor].get_function_values(
                mesh_deformation_handler.get_mesh_displacements(),
                mesh_displacement_values);
              if (include_current_velocity_increment)
                mesh_face_values[mesh_velocity_extractor]
                .get_function_values(*projected_mesh_velocity,
                                     projected_mesh_velocity_values);

              for (unsigned int q = 0;
                   q < face_values.n_quadrature_points;
                   ++q)
                {
                  const Point<dim> position =
                    face_values.quadrature_point(q);
                  const std::array<double,dim> spherical_coordinates =
                    Utilities::Coordinates::cartesian_to_spherical_coordinates(
                      position);
                  const Tensor<1,dim> radial_unit =
                    position / spherical_coordinates[0];
                  const double predicted_displacement =
                    (include_current_velocity_increment
                     ? displacement_timestep
                     * (projected_mesh_velocity_values[q] * radial_unit)
                     : 0.0);

                  phi.push_back(spherical_coordinates[1]);
                  theta.push_back(spherical_coordinates[2]);
                  weights.push_back(face_values.JxW(q)
                                    / (outer_radius * outer_radius));
                  current_ice_thickness.push_back(
                    ice_history.value(position));
                  initial_ice_thickness.push_back(
                    ice_history.initial_value(position));
                  current_geoid.push_back(
                    surface_potential_height_function(position));
                  current_displacement.push_back(
                    mesh_displacement_values[q] * radial_unit
                    + predicted_displacement);
                  prescribed_ocean_values.push_back(
                    ocean_history.value(position));
                }
            }
        }
      Assert(mesh_cell == mesh_dof_handler.end(), ExcInternalError());

      if (projection_longitude_samples > 0)
        {
          const auto solid_coefficients = sh_transform->analyze_multiple(
                                            theta,
                                            phi,
                                            weights,
          {
            current_geoid,
            current_displacement
          },
          this->get_mpi_communicator());
          AssertDimension(solid_coefficients.size(), 2);

          theta.clear();
          phi.clear();
          weights.clear();
          current_ice_thickness.clear();
          initial_ice_thickness.clear();
          current_geoid.clear();
          current_displacement.clear();
          prescribed_ocean_values.clear();

          const unsigned int mpi_rank =
            Utilities::MPI::this_mpi_process(
              this->get_mpi_communicator());
          const unsigned int n_mpi_processes =
            Utilities::MPI::n_mpi_processes(
              this->get_mpi_communicator());
          const double longitude_spacing =
            2.0 * numbers::PI / projection_longitude_samples;
          const double latitude_spacing =
            numbers::PI / projection_latitude_samples;

          for (unsigned int latitude_index = 0;
               latitude_index < projection_latitude_samples;
               ++latitude_index)
            for (unsigned int longitude_index = 0;
                 longitude_index < projection_longitude_samples;
                 ++longitude_index)
              {
                const std::size_t global_index =
                  static_cast<std::size_t>(longitude_index)
                  + static_cast<std::size_t>(
                      projection_longitude_samples)
                  * latitude_index;
                if (global_index % n_mpi_processes != mpi_rank)
                  continue;

                const double colatitude =
                  (latitude_index + 0.5) * latitude_spacing;
                const double longitude =
                  (longitude_index + 0.5) * longitude_spacing;
                const double cell_weight =
                  longitude_spacing
                  * (std::cos(latitude_index * latitude_spacing)
                     - std::cos((latitude_index + 1)
                                * latitude_spacing));
                std::array<double,dim> spherical_position;
                spherical_position[0] = outer_radius;
                spherical_position[1] = longitude;
                if constexpr (dim == 3)
                  spherical_position[2] = colatitude;
                const Point<dim> position =
                  Utilities::Coordinates::
                  spherical_to_cartesian_coordinates<dim>(
                    spherical_position);

                theta.push_back(colatitude);
                phi.push_back(longitude);
                weights.push_back(cell_weight);
                current_ice_thickness.push_back(
                  ice_history.value(position));
                initial_ice_thickness.push_back(
                  ice_history.initial_value(position));
                prescribed_ocean_values.push_back(
                  ocean_history.value(position));
              }

          current_geoid =
            sh_transform->synthesize(solid_coefficients[0].first,
                                     solid_coefficients[0].second,
                                     theta,
                                     phi);
          current_displacement =
            sh_transform->synthesize(solid_coefficients[1].first,
                                     solid_coefficients[1].second,
                                     theta,
                                     phi);
        }

      const unsigned int n_points = weights.size();
      std::vector<double> ice_load(n_points, 0.0);
      std::vector<double> ocean_load(n_points, 0.0);
      std::vector<double> total_load(n_points, 0.0);
      std::vector<double> sea_level(n_points, 0.0);
      std::vector<double> ocean_function_values(n_points, 0.0);

      double local_ocean_area = 0.0;
      double local_surface_area = 0.0;
      double local_ice_mass_change = 0.0;
      double local_relative_sea_level_volume = 0.0;
      for (unsigned int point_index = 0;
           point_index < n_points;
           ++point_index)
        {
          const double relative_geoid =
            current_geoid[point_index]
            - current_displacement[point_index];
          const double ocean_function =
            std::max(0.0,
                     std::min(1.0,
                              prescribed_ocean_values[point_index]));
          const double ice_mass_density =
            GIA::ice_load_mass_density(
              current_ice_thickness[point_index],
              initial_ice_thickness[point_index],
              density_ice,
              ice_load_reference);
          const double surface_area_weight =
            weights[point_index] * outer_radius * outer_radius;

          ice_load[point_index] = ice_mass_density;
          ocean_function_values[point_index] = ocean_function;
          local_surface_area += surface_area_weight;
          local_ocean_area += ocean_function * surface_area_weight;
          local_ice_mass_change +=
            ice_mass_density * surface_area_weight;
          local_relative_sea_level_volume +=
            relative_geoid * ocean_function * surface_area_weight;
        }

      current_ocean_area =
        Utilities::MPI::sum(local_ocean_area,
                            this->get_mpi_communicator());
      current_ice_mass_change =
        Utilities::MPI::sum(local_ice_mass_change,
                            this->get_mpi_communicator());
      if (ice_load_reference == IceLoadReference::signed_anomaly)
        {
          const double surface_area =
            Utilities::MPI::sum(local_surface_area,
                                this->get_mpi_communicator());
          AssertThrow(surface_area > 0.0, ExcInternalError());
          const double discrete_mean_surface_mass =
            current_ice_mass_change / surface_area;

          local_ice_mass_change = 0.0;
          for (unsigned int point_index = 0;
               point_index < n_points;
               ++point_index)
            {
              ice_load[point_index] -= discrete_mean_surface_mass;
              const double surface_area_weight =
                weights[point_index] * outer_radius * outer_radius;
              local_ice_mass_change +=
                ice_load[point_index] * surface_area_weight;
            }
          current_ice_mass_change =
            Utilities::MPI::sum(local_ice_mass_change,
                                this->get_mpi_communicator());
        }
      const double relative_sea_level_volume =
        Utilities::MPI::sum(local_relative_sea_level_volume,
                            this->get_mpi_communicator());
      const double sea_level_offset =
        GIA::barystatic_sea_level(current_ocean_area,
                                  current_ice_mass_change,
                                  relative_sea_level_volume,
                                  density_water);
      const double new_eustatic_sea_level =
        -current_ice_mass_change
        / (density_water * current_ocean_area);

      for (unsigned int point_index = 0;
           point_index < n_points;
           ++point_index)
        {
          const double relative_geoid =
            current_geoid[point_index]
            - current_displacement[point_index];
          const double local_sea_level =
            GIA::sea_level_change(
              ocean_function_values[point_index],
              relative_geoid,
              sea_level_offset);

          sea_level[point_index] = local_sea_level;
          ocean_load[point_index] = density_water * local_sea_level;
          total_load[point_index] =
            ice_load[point_index] + ocean_load[point_index];
        }

      const auto analyzed_fields = sh_transform->analyze_multiple(
                                     theta,
                                     phi,
                                     weights,
      {
        total_load,
        ice_load,
        ocean_load,
        sea_level,
        ocean_function_values,
        current_geoid,
        current_displacement
      },
      this->get_mpi_communicator());
      AssertDimension(analyzed_fields.size(), 7);

      const std::vector<double> old_total_cos =
        total_load_cos_coefficients;
      const std::vector<double> old_total_sin =
        total_load_sin_coefficients;

      relax_coefficients(total_load_cos_coefficients,
                         total_load_sin_coefficients,
                         analyzed_fields[0].first,
                         analyzed_fields[0].second);
      relax_coefficients(ice_load_cos_coefficients,
                         ice_load_sin_coefficients,
                         analyzed_fields[1].first,
                         analyzed_fields[1].second);
      relax_coefficients(ocean_load_cos_coefficients,
                         ocean_load_sin_coefficients,
                         analyzed_fields[2].first,
                         analyzed_fields[2].second);
      relax_coefficients(sea_level_cos_coefficients,
                         sea_level_sin_coefficients,
                         analyzed_fields[3].first,
                         analyzed_fields[3].second);
      relax_coefficients(ocean_function_cos_coefficients,
                         ocean_function_sin_coefficients,
                         analyzed_fields[4].first,
                         analyzed_fields[4].second);

      current_barystatic_sea_level +=
        potential_relaxation_factor
        * (sea_level_offset - current_barystatic_sea_level);
      current_eustatic_sea_level +=
        potential_relaxation_factor
        * (new_eustatic_sea_level - current_eustatic_sea_level);

      const std::vector<double> applied_ice_load =
        sh_transform->synthesize(ice_load_cos_coefficients,
                                 ice_load_sin_coefficients,
                                 theta,
                                 phi);
      std::vector<double> applied_ocean_load =
        sh_transform->synthesize(ocean_load_cos_coefficients,
                                 ocean_load_sin_coefficients,
                                 theta,
                                 phi);
      const std::vector<double> applied_ocean_function =
        sh_transform->synthesize(ocean_function_cos_coefficients,
                                 ocean_function_sin_coefficients,
                                 theta,
                                 phi);

      double local_applied_ice_mass_change;
      double local_ocean_water_mass_change;
      double local_applied_mass_scale;
      const auto integrate_applied_loads =
        [&]()
      {
        local_applied_ice_mass_change = 0.0;
        local_ocean_water_mass_change = 0.0;
        local_applied_mass_scale = 0.0;
        for (unsigned int point_index = 0;
             point_index < n_points;
             ++point_index)
          {
            const double surface_area_weight =
              weights[point_index] * outer_radius * outer_radius;
            local_applied_ice_mass_change +=
              applied_ice_load[point_index] * surface_area_weight;
            local_ocean_water_mass_change +=
              applied_ocean_load[point_index] * surface_area_weight;
            local_applied_mass_scale +=
              (std::abs(applied_ice_load[point_index])
               + std::abs(applied_ocean_load[point_index]))
              * surface_area_weight;
          }
      };
      integrate_applied_loads();

      double applied_ice_mass_change =
        Utilities::MPI::sum(local_applied_ice_mass_change,
                            this->get_mpi_communicator());
      current_ocean_water_mass_change =
        Utilities::MPI::sum(local_ocean_water_mass_change,
                            this->get_mpi_communicator());
      current_water_mass_residual =
        applied_ice_mass_change + current_ocean_water_mass_change;
      double applied_mass_scale =
        Utilities::MPI::sum(local_applied_mass_scale,
                            this->get_mpi_communicator());

      double local_effective_ocean_area = 0.0;
      for (unsigned int point_index = 0;
           point_index < n_points;
           ++point_index)
        local_effective_ocean_area +=
          applied_ocean_function[point_index]
          * weights[point_index] * outer_radius * outer_radius;
      const double effective_ocean_area =
        Utilities::MPI::sum(local_effective_ocean_area,
                            this->get_mpi_communicator());
      AssertThrow(effective_ocean_area > 0.0,
                  ExcMessage("The spectrally represented GIA ocean function "
                             "contains no ocean area."));

      if (std::abs(current_water_mass_residual)
          > std::numeric_limits<double>::epsilon()
          * std::max(applied_mass_scale, 1.0))
        {
          const double mass_closure_sea_level =
            -current_water_mass_residual
            / (density_water * effective_ocean_area);
          AssertDimension(total_load_cos_coefficients.size(),
                          ocean_function_cos_coefficients.size());
          for (unsigned int coefficient_index = 0;
               coefficient_index < total_load_cos_coefficients.size();
               ++coefficient_index)
            {
              const double cosine_sea_level_correction =
                mass_closure_sea_level
                * ocean_function_cos_coefficients[coefficient_index];
              const double sine_sea_level_correction =
                mass_closure_sea_level
                * ocean_function_sin_coefficients[coefficient_index];
              sea_level_cos_coefficients[coefficient_index] +=
                cosine_sea_level_correction;
              sea_level_sin_coefficients[coefficient_index] +=
                sine_sea_level_correction;
              ocean_load_cos_coefficients[coefficient_index] +=
                density_water * cosine_sea_level_correction;
              ocean_load_sin_coefficients[coefficient_index] +=
                density_water * sine_sea_level_correction;
              total_load_cos_coefficients[coefficient_index] +=
                density_water * cosine_sea_level_correction;
              total_load_sin_coefficients[coefficient_index] +=
                density_water * sine_sea_level_correction;
            }
          current_barystatic_sea_level += mass_closure_sea_level;

          applied_ocean_load =
            sh_transform->synthesize(ocean_load_cos_coefficients,
                                     ocean_load_sin_coefficients,
                                     theta,
                                     phi);
          integrate_applied_loads();
          applied_ice_mass_change =
            Utilities::MPI::sum(local_applied_ice_mass_change,
                                this->get_mpi_communicator());
          current_ocean_water_mass_change =
            Utilities::MPI::sum(local_ocean_water_mass_change,
                                this->get_mpi_communicator());
          current_water_mass_residual =
            applied_ice_mass_change + current_ocean_water_mass_change;
          applied_mass_scale =
            Utilities::MPI::sum(local_applied_mass_scale,
                                this->get_mpi_communicator());
        }

      potential_relative_change = coefficient_relative_change(
                                    old_total_cos,
                                    old_total_sin,
                                    total_load_cos_coefficients,
                                    total_load_sin_coefficients);
      potential_absolute_change = coefficient_absolute_change(
                                    old_total_cos,
                                    old_total_sin,
                                    total_load_cos_coefficients,
                                    total_load_sin_coefficients);
      current_relative_water_mass_residual =
        std::abs(current_water_mass_residual)
        / std::max(applied_mass_scale,
                   std::numeric_limits<double>::min());

      if (include_current_velocity_increment)
        {
          this->get_pcout()
              << "      GIA/SLE surface-load update: "
              << "relative SH coefficient change="
              << std::scientific << std::setprecision(6)
              << potential_relative_change
              << ", absolute SH coefficient change="
              << potential_absolute_change
              << ", prescribed ice mass change="
              << current_ice_mass_change
              << ", applied ice mass change="
              << applied_ice_mass_change
              << ", ocean water mass change="
              << current_ocean_water_mass_change
              << ", water mass residual="
              << current_water_mass_residual
              << ", relative water mass residual="
              << current_relative_water_mass_residual
              << ", applied absolute mass scale="
              << applied_mass_scale
              << ", ocean area=" << current_ocean_area
              << ", barystatic sea level="
              << current_barystatic_sea_level
              << ", eustatic sea level="
              << current_eustatic_sea_level
              << std::defaultfloat << std::endl;

          for (const unsigned int degree : diagnostic_degrees)
            {
              const unsigned int diagnostic_index =
                sh_transform->index(degree, 0);
              const double geoid_coefficient =
                analyzed_fields[5].first[diagnostic_index];
              const double displacement_coefficient =
                analyzed_fields[6].first[diagnostic_index];
              const double relative_sea_level_coefficient =
                geoid_coefficient - displacement_coefficient;

              this->get_pcout()
                  << "        l" << degree << "m0 cosine coefficients: "
                  << std::scientific << std::setprecision(6)
                  << "ice load=" << ice_load_coefficient(degree, 0).first
                  << ", ocean load="
                  << ocean_load_coefficient(degree, 0).first
                  << ", total load="
                  << total_load_coefficient(degree, 0).first
                  << ", relative sea level="
                  << sea_level_coefficient(degree, 0).first
                  << ", potential height=" << geoid_coefficient
                  << ", surface displacement=" << displacement_coefficient
                  << ", raw potential-minus-displacement="
                  << relative_sea_level_coefficient
                  << std::defaultfloat << std::endl;
            }
        }
    }



    template <int dim>
    void
    GlacialIsostaticAdjustment<dim>::relax_coefficients(
      std::vector<double> &stored_cos,
      std::vector<double> &stored_sin,
      const std::vector<double> &new_cos,
      const std::vector<double> &new_sin)
    {
      if (stored_cos.empty())
        {
          stored_cos = new_cos;
          stored_sin = new_sin;
          return;
        }

      AssertDimension(stored_cos.size(), new_cos.size());
      AssertDimension(stored_sin.size(), new_sin.size());
      for (unsigned int index = 0; index < stored_cos.size(); ++index)
        {
          stored_cos[index] +=
            potential_relaxation_factor
            * (new_cos[index] - stored_cos[index]);
          stored_sin[index] +=
            potential_relaxation_factor
            * (new_sin[index] - stored_sin[index]);
        }
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::coefficient_relative_change(
      const std::vector<double> &old_cos,
      const std::vector<double> &old_sin,
      const std::vector<double> &new_cos,
      const std::vector<double> &new_sin) const
    {
      if (old_cos.empty())
        return std::numeric_limits<double>::infinity();

      AssertDimension(old_cos.size(), new_cos.size());
      AssertDimension(old_sin.size(), new_sin.size());
      double difference_norm_square = 0.0;
      double new_norm_square = 0.0;
      for (unsigned int index = 0; index < new_cos.size(); ++index)
        {
          difference_norm_square +=
            (new_cos[index] - old_cos[index])
            * (new_cos[index] - old_cos[index])
            + (new_sin[index] - old_sin[index])
            * (new_sin[index] - old_sin[index]);
          new_norm_square += new_cos[index] * new_cos[index]
                             + new_sin[index] * new_sin[index];
        }

      return std::sqrt(difference_norm_square)
             / std::max(std::sqrt(new_norm_square),
                        std::numeric_limits<double>::min());
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::coefficient_absolute_change(
      const std::vector<double> &old_cos,
      const std::vector<double> &old_sin,
      const std::vector<double> &new_cos,
      const std::vector<double> &new_sin) const
    {
      if (old_cos.empty())
        return std::numeric_limits<double>::infinity();

      AssertDimension(old_cos.size(), new_cos.size());
      AssertDimension(old_sin.size(), new_sin.size());
      double difference_norm_square = 0.0;
      for (unsigned int index = 0; index < new_cos.size(); ++index)
        difference_norm_square +=
          (new_cos[index] - old_cos[index])
          * (new_cos[index] - old_cos[index])
          + (new_sin[index] - old_sin[index])
          * (new_sin[index] - old_sin[index]);

      return std::sqrt(difference_norm_square);
    }



    template <int dim>
    Tensor<1,dim>
    GlacialIsostaticAdjustment<dim>::boundary_traction(
      const types::boundary_id boundary_indicator,
      const Point<dim> &position,
      const Tensor<1,dim> &normal_vector) const
    {
      if (!enabled || boundary_indicator != top_boundary_id)
        return Tensor<1,dim>();

      const double gravity =
        this->get_gravity_model().gravity_vector(position).norm();
      return -gravity * surface_mass_density(position) * normal_vector;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::synthesize(
      const std::vector<double> &cos_coefficients,
      const std::vector<double> &sin_coefficients,
      const Point<dim> &position) const
    {
      if (cos_coefficients.empty())
        return 0.0;

      const std::array<double,dim> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      return sh_transform->synthesize(cos_coefficients,
                                      sin_coefficients,
      {spherical_coordinates[2]},
      {spherical_coordinates[1]})[0];
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::surface_mass_density(
      const Point<dim> &position) const
    {
      return synthesize(total_load_cos_coefficients,
                        total_load_sin_coefficients,
                        position);
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::ice_load_mass_density(
      const Point<dim> &position) const
    {
      return synthesize(ice_load_cos_coefficients,
                        ice_load_sin_coefficients,
                        position);
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::ocean_load_mass_density(
      const Point<dim> &position) const
    {
      return synthesize(ocean_load_cos_coefficients,
                        ocean_load_sin_coefficients,
                        position);
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::sea_level_change(
      const Point<dim> &position) const
    {
      return synthesize(sea_level_cos_coefficients,
                        sea_level_sin_coefficients,
                        position);
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::ocean_function(
      const Point<dim> &position) const
    {
      return std::max(0.0,
                      std::min(1.0,
                               synthesize(
                                 ocean_function_cos_coefficients,
                                 ocean_function_sin_coefficients,
                                 position)));
    }



    template <int dim>
    std::pair<double,double>
    GlacialIsostaticAdjustment<dim>::coefficient(
      const std::vector<double> &cos_coefficients,
      const std::vector<double> &sin_coefficients,
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(degree <= maximum_degree && order <= degree,
                  ExcMessage("The requested GIA spherical-harmonic "
                             "coefficient is outside the configured range."));

      if (cos_coefficients.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      AssertIndexRange(index, cos_coefficients.size());
      AssertIndexRange(index, sin_coefficients.size());
      return {cos_coefficients[index], sin_coefficients[index]};
    }



    template <int dim>
    std::pair<double,double>
    GlacialIsostaticAdjustment<dim>::total_load_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      return coefficient(total_load_cos_coefficients,
                         total_load_sin_coefficients,
                         degree,
                         order);
    }



    template <int dim>
    std::pair<double,double>
    GlacialIsostaticAdjustment<dim>::ice_load_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      return coefficient(ice_load_cos_coefficients,
                         ice_load_sin_coefficients,
                         degree,
                         order);
    }



    template <int dim>
    std::pair<double,double>
    GlacialIsostaticAdjustment<dim>::ocean_load_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      return coefficient(ocean_load_cos_coefficients,
                         ocean_load_sin_coefficients,
                         degree,
                         order);
    }



    template <int dim>
    std::pair<double,double>
    GlacialIsostaticAdjustment<dim>::sea_level_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      return coefficient(sea_level_cos_coefficients,
                         sea_level_sin_coefficients,
                         degree,
                         order);
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::barystatic_sea_level() const
    {
      return current_barystatic_sea_level;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::eustatic_sea_level() const
    {
      return current_eustatic_sea_level;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::ice_mass_change() const
    {
      return current_ice_mass_change;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::ocean_water_mass_change() const
    {
      return current_ocean_water_mass_change;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::water_mass_residual() const
    {
      return current_water_mass_residual;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::relative_water_mass_residual() const
    {
      return current_relative_water_mass_residual;
    }



    template <int dim>
    bool
    GlacialIsostaticAdjustment<dim>::potential_is_converged() const
    {
      if (!enabled
          || (freeze_feedback_after_timestep_zero
              && this->get_timestep_number() > 0))
        return true;

      const bool converged =
        potential_relative_change <= potential_convergence_tolerance
        || (absolute_coefficient_tolerance > 0.0
            && potential_absolute_change
            <= absolute_coefficient_tolerance);
      AssertThrow(converged
                  || potential_iteration_number
                  < maximum_potential_iterations,
                  ExcMessage(
                    "The coupled GIA/sea-level-equation solve reached the "
                    "maximum number of potential iterations without "
                    "satisfying either the relative or absolute surface-load "
                    "coefficient-change tolerance."));
      return converged;
    }



    template <int dim>
    double
    GlacialIsostaticAdjustment<dim>::potential_relative_change_value() const
    {
      return potential_relative_change;
    }



    template <int dim>
    bool
    GlacialIsostaticAdjustment<dim>::is_enabled() const
    {
      return enabled;
    }
  }
}

namespace aspect
{
  namespace PotentialFeedback
  {
    template class GlacialIsostaticAdjustment<2>;
    template class GlacialIsostaticAdjustment<3>;
  }
}
