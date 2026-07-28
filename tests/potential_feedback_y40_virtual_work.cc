/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/boundary_traction/potential_feedback_traction.h>
#include <aspect/density_source_manager.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

namespace aspect
{
  namespace
  {
    enum Channel
    {
      volume_potential_rhs,
      volume_history_rhs,
      volume_current_lhs,
      internal_potential_rhs,
      internal_history_rhs,
      internal_current_lhs,
      boundary_potential_rhs,
      n_channels
    };



    template <int dim>
    const PotentialFeedback::SelfGravitation<dim> *
    get_self_gravity(const SimulatorAccess<dim> &simulator_access)
    {
      const auto &traction_manager =
        simulator_access.get_boundary_traction_manager();
      if (traction_manager.template has_matching_active_plugin<
          PotentialFeedback::SelfGravitation<dim>>())
        return &traction_manager.template get_matching_active_plugin<
               PotentialFeedback::SelfGravitation<dim>>();

      if (traction_manager.template has_matching_active_plugin<
          BoundaryTraction::PotentialFeedbackTraction<dim>>())
        {
          const auto &potential_feedback =
            traction_manager.template get_matching_active_plugin<
            BoundaryTraction::PotentialFeedbackTraction<dim>>();
          if (potential_feedback.has_self_gravity_feedback())
            return &potential_feedback.get_self_gravity();
        }

      return nullptr;
    }



    template <int dim>
    double
    zonal_harmonic(const Point<dim> &point,
                   const unsigned int degree)
    {
      if constexpr (dim == 3)
        {
          const std::array<double,dim> coordinates =
            Utilities::Coordinates::cartesian_to_spherical_coordinates(point);
          return Utilities::real_spherical_harmonic(
                   degree, 0, coordinates[2], coordinates[1]).first;
        }
      else
        return 0.0;
    }



    template <int dim>
    double
    full_domain_potential(const SimulatorAccess<dim> &simulator_access,
                          const Point<dim> &point)
    {
      const auto &traction_manager =
        simulator_access.get_boundary_traction_manager();
      if (traction_manager.template has_matching_active_plugin<
          BoundaryTraction::PotentialFeedbackTraction<dim>>())
        return traction_manager.template get_matching_active_plugin<
               BoundaryTraction::PotentialFeedbackTraction<dim>>()
               .full_domain_potential(point);

      const auto *self_gravity = get_self_gravity(simulator_access);
      return (self_gravity != nullptr && self_gravity->has_full_domain_potential()
              ? self_gravity->full_domain_potential(point)
              : 0.0);
    }



    template <int dim>
    void
    observe_y40_virtual_work(const SimulatorAccess<dim> &simulator_access)
    {
      if constexpr (dim != 3)
        return;

      const auto &parameters = simulator_access.get_parameters();
      if (parameters.density_source_law
          != Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation)
        return;

      const auto &density_sources = simulator_access.get_density_source_manager();
      const auto &introspection = simulator_access.introspection();
      const unsigned int displacement_index =
        introspection.compositional_index_for_name("ve_radial_displacement");
      const double mechanical_time_step =
        density_sources.effective_mechanical_time_step();
      const unsigned int quadrature_degree =
        std::max(2u, introspection.polynomial_degree.velocities + 1u);
      const QGauss<dim> quadrature(quadrature_degree);
      const QGauss<dim-1> face_quadrature(quadrature_degree);
      FEValues<dim> values(simulator_access.get_mapping(),
                           simulator_access.get_fe(),
                           quadrature,
                           update_values |
                           update_quadrature_points |
                           update_JxW_values);
      FEFaceValues<dim> face_values(simulator_access.get_mapping(),
                                    simulator_access.get_fe(),
                                    face_quadrature,
                                    update_values |
                                    update_quadrature_points |
                                    update_normal_vectors |
                                    update_JxW_values);
      const auto &mesh_deformation_handler =
        simulator_access.get_mesh_deformation_handler();
      const DoFHandler<dim> &mesh_dof_handler =
        mesh_deformation_handler.get_mesh_deformation_dof_handler();
      FEFaceValues<dim> mesh_face_values(simulator_access.get_mapping(),
                                         mesh_dof_handler.get_fe(),
                                         face_quadrature,
                                         update_values);
      const FEValuesExtractors::Vector mesh_velocity_extractor(0);
      const auto mesh_dual_solution =
        [&mesh_deformation_handler] (const unsigned int degree)
      {
        return mesh_deformation_handler.project_free_surface_boundary_field(
                 [degree] (const Point<dim> &point, const Tensor<1,dim> &)
        {
          return (point / point.norm()) * zonal_harmonic(point, degree);
        },
        false);
      };
      const std::array<LinearAlgebra::Vector,2> adjoint_mesh_fields = {{
          mesh_dual_solution(2), mesh_dual_solution(4)
        }
      };
      const LinearAlgebra::Vector &projected_velocity =
        mesh_deformation_handler.get_projected_free_surface_velocity();

      std::vector<Tensor<1,dim>> velocities(quadrature.size());
      std::vector<double> old_displacements(quadrature.size());
      std::vector<Tensor<1,dim>> face_velocities(face_quadrature.size());
      std::vector<double> face_old_displacements(face_quadrature.size());
      std::array<double,4> local_gram = {};
      std::array<std::array<double,n_channels>,2> local_raw_work = {};

      const bool volume_restoring = parameters.enable_mechanical_volume_restoring
                                    && parameters.enable_mechanical_radial_volume_restoring;
      const bool potential_force = parameters.enable_full_domain_potential_force;

      for (const auto &cell :
           simulator_access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            values[introspection.extractors.velocities].get_function_values(
              simulator_access.get_solution(), velocities);
            values[introspection.extractors.compositional_fields[displacement_index]]
            .get_function_values(simulator_access.get_current_linearization_point(),
                                 old_displacements);

            for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
              {
                const Point<dim> point = values.quadrature_point(q);
                const double radius = point.norm();
                AssertThrow(radius > 0.0,
                            ExcMessage("The Y40 virtual-work test is undefined at radius zero."));
                const Tensor<1,dim> radial_unit = point / radius;
                const std::array<double,2> harmonics = {{
                    zonal_harmonic(point, 2), zonal_harmonic(point, 4)
                  }
                };
                const double JxW = values.JxW(q);
                const double reference_density =
                  density_sources.reference_density(point);
                const double gravity_magnitude =
                  density_sources.mechanical_gravity_magnitude(
                    point,
                    simulator_access.get_gravity_model().gravity_vector(point).norm());

                for (unsigned int row = 0; row < 2; ++row)
                  for (unsigned int column = 0; column < 2; ++column)
                    local_gram[2 * row + column] +=
                      harmonics[row] * harmonics[column] * JxW;

                if (potential_force)
                  {
                    const double potential =
                      full_domain_potential(simulator_access, point);
                    const Tensor<1,dim> reference_density_gradient =
                      density_sources.reference_density_gradient(point);
                    const double work_density =
                      -potential
                      * (reference_density * 2.0 / radius
                         + reference_density_gradient * radial_unit) * JxW;
                    for (unsigned int degree = 0; degree < 2; ++degree)
                      local_raw_work[degree][volume_potential_rhs] +=
                        work_density * harmonics[degree];
                  }

                if (volume_restoring)
                  {
                    const double coefficient = reference_density * gravity_magnitude;
                    const double history_density =
                      coefficient * 2.0 / radius * old_displacements[q] * JxW;
                    const double current_density =
                      -coefficient * mechanical_time_step * 2.0 / radius
                      * (velocities[q] * radial_unit) * JxW;
                    for (unsigned int degree = 0; degree < 2; ++degree)
                      {
                        local_raw_work[degree][volume_history_rhs] +=
                          history_density * harmonics[degree];
                        local_raw_work[degree][volume_current_lhs] +=
                          current_density * harmonics[degree];
                      }
                  }
              }

            if (parameters.enable_internal_density_jump_restoring
                && density_sources.has_internal_density_jumps())
              for (const unsigned int face_no : cell->face_indices())
                {
                  if (cell->at_boundary(face_no)
                      || cell->has_periodic_neighbor(face_no))
                    continue;

                  const auto neighbor = cell->neighbor(face_no);
                  const Point<dim> inner_cell_center =
                    density_sources.radial_cell_representative_point(cell);
                  const Point<dim> outer_cell_center =
                    density_sources.radial_cell_representative_point(neighbor);
                  if (inner_cell_center.norm() >= outer_cell_center.norm())
                    continue;

                  const auto face = cell->face(face_no);
                  const double density_contrast =
                    density_sources.internal_density_jump_across_face(
                      inner_cell_center,
                      outer_cell_center,
                      face->vertex(0).norm());
                  if (density_contrast == 0.0)
                    continue;

                  bool all_vertices_match = true;
                  for (unsigned int vertex = 1;
                       vertex < face->n_vertices();
                       ++vertex)
                    all_vertices_match = all_vertices_match
                                         && (density_sources.internal_density_jump_across_face(
                                               inner_cell_center,
                                               outer_cell_center,
                                               face->vertex(vertex).norm())
                                             == density_contrast);
                  if (!all_vertices_match)
                    continue;

                  face_values.reinit(cell, face_no);
                  face_values[introspection.extractors.velocities]
                  .get_function_values(simulator_access.get_solution(),
                                       face_velocities);
                  face_values[introspection.extractors.compositional_fields[displacement_index]]
                  .get_function_values(simulator_access.get_current_linearization_point(),
                                       face_old_displacements);

                  for (unsigned int q = 0;
                       q < face_values.n_quadrature_points;
                       ++q)
                    {
                      const Point<dim> point = face_values.quadrature_point(q);
                      const Tensor<1,dim> radial_unit = point / point.norm();
                      const std::array<double,2> harmonics = {{
                          zonal_harmonic(point, 2), zonal_harmonic(point, 4)
                        }
                      };
                      const double gravity_magnitude =
                        simulator_access.get_gravity_model().gravity_vector(point).norm();
                      const double JxW = face_values.JxW(q);

                      for (unsigned int degree = 0; degree < 2; ++degree)
                        {
                          if (potential_force)
                            local_raw_work[degree][internal_potential_rhs] +=
                              density_contrast
                              * full_domain_potential(simulator_access, point)
                              * harmonics[degree] * JxW;
                          local_raw_work[degree][internal_history_rhs] +=
                            -density_contrast * gravity_magnitude
                            * face_old_displacements[q] * harmonics[degree] * JxW;
                          local_raw_work[degree][internal_current_lhs] +=
                            density_contrast * gravity_magnitude
                            * mechanical_time_step
                            * (face_velocities[q] * radial_unit)
                            * harmonics[degree] * JxW;
                        }
                    }
                }

            if (cell->at_boundary())
              for (const unsigned int face_no : cell->face_indices())
                if (cell->at_boundary(face_no))
                  {
                    face_values.reinit(cell, face_no);
                    const types::boundary_id boundary_id =
                      cell->face(face_no)->boundary_id();
                    const auto *self_gravity = get_self_gravity(simulator_access);
                    if (self_gravity == nullptr)
                      continue;
                    for (unsigned int q = 0;
                         q < face_values.n_quadrature_points;
                         ++q)
                      {
                        const Point<dim> point = face_values.quadrature_point(q);
                        const Tensor<1,dim> radial_unit = point / point.norm();
                        const Tensor<1,dim> traction =
                          self_gravity->boundary_traction(
                            boundary_id,
                            point,
                            face_values.normal_vector(q));
                        for (unsigned int degree = 0; degree < 2; ++degree)
                          local_raw_work[degree][boundary_potential_rhs] +=
                            (traction * radial_unit)
                            * zonal_harmonic(point, degree == 0 ? 2 : 4)
                            * face_values.JxW(q);
                      }
                  }
          }

      std::array<double,2> local_source_action = {};
      std::array<double,2> local_reverse_action = {};
      std::array<double,2> local_direct_normal_action = {};
      std::vector<Tensor<1,dim>> projected_velocity_values(face_quadrature.size());
      std::array<std::vector<Tensor<1,dim>>,2> adjoint_mesh_values = {{
          std::vector<Tensor<1,dim>>(face_quadrature.size()),
          std::vector<Tensor<1,dim>>(face_quadrature.size())
        }
      };
      auto cell = simulator_access.get_dof_handler().begin_active();
      const auto end_cell = simulator_access.get_dof_handler().end();
      auto mesh_cell = mesh_dof_handler.begin_active();
      for (; cell != end_cell; ++cell, ++mesh_cell)
        if (cell->is_locally_owned() && cell->at_boundary())
          for (const unsigned int face_no : cell->face_indices())
            if (cell->at_boundary(face_no)
                && mesh_deformation_handler.get_free_surface_boundary_indicators().count(
                  cell->face(face_no)->boundary_id()) > 0)
              {
                face_values.reinit(cell, face_no);
                mesh_face_values.reinit(mesh_cell, face_no);
                face_values[introspection.extractors.velocities]
                .get_function_values(simulator_access.get_solution(), face_velocities);
                mesh_face_values[mesh_velocity_extractor].get_function_values(
                  projected_velocity, projected_velocity_values);
                for (unsigned int degree = 0; degree < 2; ++degree)
                  mesh_face_values[mesh_velocity_extractor].get_function_values(
                    adjoint_mesh_fields[degree], adjoint_mesh_values[degree]);

                for (unsigned int q = 0; q < face_values.n_quadrature_points; ++q)
                  {
                    const Point<dim> point = face_values.quadrature_point(q);
                    const Tensor<1,dim> radial = point / point.norm();
                    const Tensor<1,dim> direction =
                      mesh_deformation_handler.free_surface_projection_direction(
                        point, face_values.normal_vector(q));
                    const double JxW = face_values.JxW(q);
                    for (unsigned int degree = 0; degree < 2; ++degree)
                      {
                        const double harmonic =
                          zonal_harmonic(point, degree == 0 ? 2 : 4);
                        local_source_action[degree] +=
                          harmonic * (projected_velocity_values[q] * radial) * JxW;
                        local_reverse_action[degree] +=
                          (face_velocities[q] * direction)
                          * (adjoint_mesh_values[degree][q] * direction) * JxW;
                        local_direct_normal_action[degree] +=
                          harmonic * (face_velocities[q] * face_values.normal_vector(q))
                          * JxW;
                      }
                  }
              }

      std::array<double,4> gram = {};
      for (unsigned int entry = 0; entry < 4; ++entry)
        gram[entry] = Utilities::MPI::sum(
                        local_gram[entry],
                        simulator_access.get_mpi_communicator());
      AssertThrow(gram[0] > 0.0,
                  ExcMessage("The sampled Y20 norm is zero."));
      const double y40_y20_projection = gram[2] / gram[0];

      std::array<double,n_channels> global = {};
      for (unsigned int channel = 0; channel < n_channels; ++channel)
        {
          const double raw_y20 = Utilities::MPI::sum(
                                   local_raw_work[0][channel],
                                   simulator_access.get_mpi_communicator());
          const double raw_y40 = Utilities::MPI::sum(
                                   local_raw_work[1][channel],
                                   simulator_access.get_mpi_communicator());
          global[channel] = raw_y40
                            - y40_y20_projection * raw_y20;
        }

      std::array<double,2> source_actions = {};
      std::array<double,2> reverse_actions = {};
      std::array<double,2> direct_normal_actions = {};
      for (unsigned int degree = 0; degree < 2; ++degree)
        {
          source_actions[degree] = Utilities::MPI::sum(
                                     local_source_action[degree],
                                     simulator_access.get_mpi_communicator());
          reverse_actions[degree] = Utilities::MPI::sum(
                                      local_reverse_action[degree],
                                      simulator_access.get_mpi_communicator());
          direct_normal_actions[degree] = Utilities::MPI::sum(
                                            local_direct_normal_action[degree],
                                            simulator_access.get_mpi_communicator());
        }

      static unsigned int solve = 0;
      if (Utilities::MPI::this_mpi_process(
            simulator_access.get_mpi_communicator()) == 0)
        {
          const std::array<const char *,n_channels> labels = {{
              "volume_full_domain_potential_rhs",
              "volume_local_history_rhs",
              "volume_local_current_lhs",
              "internal_jump_potential_rhs",
              "internal_jump_history_rhs",
              "internal_jump_current_lhs",
              "boundary_self_gravity_potential_rhs"
            }
          };
          std::cout << std::scientific << std::setprecision(16);
          for (unsigned int degree = 0; degree < 2; ++degree)
            {
              const double source_action = source_actions[degree];
              const double reverse_action = reverse_actions[degree];
              const double scale = std::max(std::abs(source_action),
                                            std::abs(reverse_action));
              const double residual = (scale > 0.0
                                       ? std::abs(source_action - reverse_action) / scale
                                       : 0.0);
              std::cout << "PROJECTION_ADJOINT_WORK"
                        << " solve=" << solve
                        << " timestep=" << simulator_access.get_timestep_number()
                        << " nonlinear=" << simulator_access.get_nonlinear_iteration()
                        << " degree=" << (degree == 0 ? 2 : 4)
                        << " source_action=" << source_action
                        << " reverse_action=" << reverse_action
                        << " direct_normal_action=" << direct_normal_actions[degree]
                        << " relative_residual=" << residual
                        << std::endl;
            }

          for (unsigned int channel = 0; channel < n_channels; ++channel)
            std::cout << "Y40WORK"
                      << " solve=" << solve
                      << " timestep=" << simulator_access.get_timestep_number()
                      << " nonlinear=" << simulator_access.get_nonlinear_iteration()
                      << " timing=post_stokes_solution_current_feedback_state"
                      << " projection=y40_orthogonal_to_sampled_y20"
                      << " channel=" << labels[channel]
                      << " work=" << global[channel]
                      << std::endl;
        }
      ++solve;
    }



    template <int dim>
    void
    connect_y40_virtual_work_observer(SimulatorSignals<dim> &signals)
    {
      signals.post_set_initial_state.connect(
        [&signals] (const SimulatorAccess<dim> &)
      {
        signals.post_stokes_solver.connect(
          [] (const SimulatorAccess<dim> &simulator_access,
              const unsigned int,
              const unsigned int,
              const SolverControl &,
              const SolverControl &)
        {
          observe_y40_virtual_work(simulator_access);
        });
      });
    }
  }



  ASPECT_REGISTER_SIGNALS_CONNECTOR(
    connect_y40_virtual_work_observer<2>,
    connect_y40_virtual_work_observer<3>)
}
