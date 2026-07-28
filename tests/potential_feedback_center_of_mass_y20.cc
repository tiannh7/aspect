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

#include <aspect/boundary_traction/potential_feedback_traction.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

namespace aspect
{
  namespace
  {
    struct BoundaryHarmonicCoefficients
    {
      std::array<double,9> raw_radial = {};
      std::array<double,9> raw_normal = {};
      std::array<double,9> projected_radial = {};
      std::array<double,9> projected_normal = {};
      std::array<double,9> sampled_y20 = {};
      std::array<double,81> gram = {};
    };



    struct PotentialDiagnosticState
    {
      bool valid = false;
      unsigned int timestep = numbers::invalid_unsigned_int;
      unsigned int nonlinear_iteration = numbers::invalid_unsigned_int;
      unsigned int potential_update = 0;
      std::array<double,2> external_load = {};
      std::array<double,2> surface_deformation = {};
      std::array<double,2> cmb = {};
      std::array<double,2> total_surface = {};
      std::array<double,2> representative_radius_direct = {};
      std::array<double,2> representative_radius_gram = {};
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



    std::array<double,9>
    solve_gram_system(const std::array<double,81> &gram_entries,
                      const std::array<double,9> &right_hand_side)
    {
      FullMatrix<double> gram(9, 9);
      Vector<double> rhs(9);
      Vector<double> solution(9);
      for (unsigned int i = 0; i < 9; ++i)
        {
          rhs[i] = right_hand_side[i];
          for (unsigned int j = 0; j < 9; ++j)
            gram(i,j) = gram_entries[9 * i + j];
        }

      const FullMatrix<double> original_gram(gram);
      gram.gauss_jordan();
      gram.vmult(solution, rhs);

      double residual = 0.0;
      double scale = 0.0;
      for (unsigned int i = 0; i < 9; ++i)
        {
          double row_residual = -rhs[i];
          for (unsigned int j = 0; j < 9; ++j)
            row_residual += original_gram(i,j) * solution[j];
          residual = std::max(residual, std::abs(row_residual));
          scale = std::max(scale, std::abs(rhs[i]));
        }
      AssertThrow(residual <= 1e-11 * std::max(1.0, scale),
                  ExcMessage("The Y20 discrete Gram solve did not converge."));

      std::array<double,9> coefficients = {};
      for (unsigned int i = 0; i < 9; ++i)
        coefficients[i] = solution[i];
      return coefficients;
    }



    template <int dim>
    std::array<double,4>
    observe_representative_radius_potential(
      const SimulatorAccess<dim> &simulator_access,
      const PotentialFeedback::SelfGravitation<dim> &self_gravity,
      const double representative_radius)
    {
      std::array<double,9> local_direct = {};
      std::array<double,81> local_gram = {};
      const QGauss<dim> quadrature(
        std::max(2u,
                 simulator_access.introspection().polynomial_degree.velocities
                 + 1u));
      FEValues<dim> values(simulator_access.get_mapping(),
                           simulator_access.get_fe(),
                           quadrature,
                           update_quadrature_points |
                           update_JxW_values);

      for (const auto &cell :
           simulator_access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
              {
                const Point<dim> volume_point = values.quadrature_point(q);
                const std::array<double,dim> coordinates =
                  Utilities::Coordinates::cartesian_to_spherical_coordinates(
                    volume_point);
                const Point<dim> representative_point =
                  volume_point * (representative_radius / coordinates[0]);
                const double angular_weight =
                  values.JxW(q) / std::pow(coordinates[0], dim);
                const double potential_height =
                  self_gravity.full_domain_potential(representative_point)
                  / simulator_access.get_gravity_model().gravity_vector(
                    representative_point).norm();

                std::array<double,9> harmonics;
                for (unsigned int degree = 0; degree <= 8; ++degree)
                  harmonics[degree] =
                    Utilities::real_spherical_harmonic(
                      degree,
                      0,
                      coordinates[2],
                      coordinates[1]).first;

                for (unsigned int degree = 0; degree <= 8; ++degree)
                  {
                    local_direct[degree] += angular_weight
                                            * potential_height
                                            * harmonics[degree];
                    for (unsigned int other_degree = 0;
                         other_degree <= 8;
                         ++other_degree)
                      local_gram[9 * degree + other_degree] +=
                        angular_weight * harmonics[degree]
                        * harmonics[other_degree];
                  }
              }
          }

      std::array<double,9> direct = {};
      std::array<double,81> gram = {};
      for (unsigned int degree = 0; degree <= 8; ++degree)
        direct[degree] = Utilities::MPI::sum(
                           local_direct[degree],
                           simulator_access.get_mpi_communicator());
      for (unsigned int i = 0; i < 81; ++i)
        gram[i] = Utilities::MPI::sum(
                    local_gram[i],
                    simulator_access.get_mpi_communicator());
      const std::array<double,9> corrected = solve_gram_system(gram, direct);
      return {{direct[2], direct[4], corrected[2], corrected[4]}};
    }



    template <int dim>
    void
    observe_boundary_velocity_harmonics(
      const SimulatorAccess<dim> &simulator_access)
    {
      if constexpr (dim != 3)
        return;

      const unsigned int quadrature_degree =
        std::max(2u,
                 simulator_access.introspection().polynomial_degree.velocities
                 + 1u);
      const QGauss<dim-1> face_quadrature(quadrature_degree);
      FEFaceValues<dim> face_values(
        simulator_access.get_mapping(),
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
      FEFaceValues<dim> mesh_face_values(
        simulator_access.get_mapping(),
        mesh_dof_handler.get_fe(),
        face_quadrature,
        update_values);
      const FEValuesExtractors::Vector mesh_velocity_extractor(0);
      const LinearAlgebra::Vector &projected_velocity =
        mesh_deformation_handler.get_projected_free_surface_velocity(true);

      std::vector<Tensor<1,dim>> raw_velocity_values(
        face_quadrature.size());
      std::vector<Tensor<1,dim>> projected_velocity_values(
        face_quadrature.size());
      const types::boundary_id top_boundary_id =
        simulator_access.get_geometry_model()
        .translate_symbolic_boundary_name_to_id("top");

      BoundaryHarmonicCoefficients local_coefficients;
      auto mesh_cell = mesh_dof_handler.begin_active();
      for (const auto &cell :
           simulator_access.get_dof_handler().active_cell_iterators())
        {
          const auto current_mesh_cell = mesh_cell;
          ++mesh_cell;
          if (!cell->is_locally_owned() || !cell->at_boundary())
            continue;

          for (const unsigned int face_no : cell->face_indices())
            if (cell->at_boundary(face_no)
                && cell->face(face_no)->boundary_id() == top_boundary_id)
              {
                face_values.reinit(cell, face_no);
                mesh_face_values.reinit(current_mesh_cell, face_no);
                face_values[simulator_access.introspection().extractors.velocities]
                .get_function_values(simulator_access.get_solution(),
                                     raw_velocity_values);
                mesh_face_values[mesh_velocity_extractor]
                .get_function_values(projected_velocity,
                                     projected_velocity_values);

                for (unsigned int q = 0;
                     q < face_values.n_quadrature_points;
                     ++q)
                  {
                    const Point<dim> position =
                      face_values.quadrature_point(q);
                    const std::array<double,dim> spherical_coordinates =
                      Utilities::Coordinates::
                      cartesian_to_spherical_coordinates(position);
                    const Tensor<1,dim> radial_unit =
                      position / spherical_coordinates[0];
                    const Tensor<1,dim> normal =
                      face_values.normal_vector(q);
                    const double angular_weight =
                      face_values.JxW(q)
                      / (spherical_coordinates[0]
                         * spherical_coordinates[0]);

                    std::array<double,9> harmonics;
                    for (unsigned int degree = 0; degree <= 8; ++degree)
                      harmonics[degree] =
                        Utilities::real_spherical_harmonic(
                          degree,
                          0,
                          spherical_coordinates[2],
                          spherical_coordinates[1]).first;

                    for (unsigned int degree = 0;
                         degree <= 8;
                         ++degree)
                      {
                        const double weighted_harmonic =
                          angular_weight * harmonics[degree];
                        local_coefficients.raw_radial[degree] +=
                          (raw_velocity_values[q] * radial_unit)
                          * weighted_harmonic;
                        local_coefficients.raw_normal[degree] +=
                          (raw_velocity_values[q] * normal)
                          * weighted_harmonic;
                        local_coefficients.projected_radial[degree] +=
                          (projected_velocity_values[q] * radial_unit)
                          * weighted_harmonic;
                        local_coefficients.projected_normal[degree] +=
                          (projected_velocity_values[q] * normal)
                          * weighted_harmonic;
                        local_coefficients.sampled_y20[degree] +=
                          harmonics[2] * weighted_harmonic;
                        for (unsigned int other_degree = 0;
                             other_degree <= 8;
                             ++other_degree)
                          local_coefficients.gram[9 * degree + other_degree] +=
                            weighted_harmonic * harmonics[other_degree];
                      }
                  }
              }
        }

      BoundaryHarmonicCoefficients coefficients;
      for (unsigned int degree = 0; degree <= 8; ++degree)
        {
          coefficients.raw_radial[degree] = Utilities::MPI::sum(
                                              local_coefficients.raw_radial[degree],
                                              simulator_access.get_mpi_communicator());
          coefficients.raw_normal[degree] = Utilities::MPI::sum(
                                              local_coefficients.raw_normal[degree],
                                              simulator_access.get_mpi_communicator());
          coefficients.projected_radial[degree] = Utilities::MPI::sum(
                                                    local_coefficients.projected_radial[degree],
                                                    simulator_access.get_mpi_communicator());
          coefficients.projected_normal[degree] = Utilities::MPI::sum(
                                                    local_coefficients.projected_normal[degree],
                                                    simulator_access.get_mpi_communicator());
          coefficients.sampled_y20[degree] = Utilities::MPI::sum(
                                               local_coefficients.sampled_y20[degree],
                                               simulator_access.get_mpi_communicator());
        }
      for (unsigned int i = 0; i < 81; ++i)
        coefficients.gram[i] = Utilities::MPI::sum(
                                 local_coefficients.gram[i],
                                 simulator_access.get_mpi_communicator());

      const std::array<double,9> raw_radial_gram =
        solve_gram_system(coefficients.gram, coefficients.raw_radial);
      const std::array<double,9> projected_radial_gram =
        solve_gram_system(coefficients.gram, coefficients.projected_radial);
      const std::array<double,9> sampled_y20_gram =
        solve_gram_system(coefficients.gram, coefficients.sampled_y20);
      for (unsigned int degree = 0; degree <= 8; ++degree)
        AssertThrow(std::abs(sampled_y20_gram[degree]
                             - (degree == 2 ? 1.0 : 0.0)) <= 1e-10,
                    ExcMessage("The discrete Gram oracle did not recover pure Y20."));

      AssertThrow(std::abs(coefficients.raw_radial[2]) > 0.0,
                  ExcMessage("The Y20 boundary-velocity oracle observed no target mode."));

      if (Utilities::MPI::this_mpi_process(
            simulator_access.get_mpi_communicator()) == 0)
        {
          std::cout << std::scientific << std::setprecision(6);
          for (const unsigned int degree :
          {
            2u, 4u, 6u, 8u
          })
          std::cout
              << "Y20DIAG velocity"
              << " timestep=" << simulator_access.get_timestep_number()
              << " nonlinear="
              << simulator_access.get_nonlinear_iteration()
              << " degree=" << degree
              << " raw_radial="
              << coefficients.raw_radial[degree]
              << " raw_normal="
              << coefficients.raw_normal[degree]
              << " projected_radial="
              << coefficients.projected_radial[degree]
              << " projected_normal="
              << coefficients.projected_normal[degree]
              << " raw_radial_gram="
              << raw_radial_gram[degree]
              << " projected_radial_gram="
              << projected_radial_gram[degree]
              << std::endl;
        }
    }



    template <int dim>
    void
    observe_y20_diagnostic(const SimulatorAccess<dim> &simulator_access)
    {
      if constexpr (dim != 3)
        return;
      else
        {

          static PotentialDiagnosticState source_used;
          static unsigned int potential_update = 0;
          const unsigned int timestep = simulator_access.get_timestep_number();
          const unsigned int nonlinear_iteration =
            simulator_access.get_nonlinear_iteration();
          if (!source_used.valid || source_used.timestep != timestep)
            {
              source_used = PotentialDiagnosticState();
              potential_update = 0;
            }

          const PotentialFeedback::SelfGravitation<dim> *self_gravity =
            get_self_gravity(simulator_access);
          PotentialDiagnosticState response;
          response.valid = true;
          response.timestep = timestep;
          response.nonlinear_iteration = nonlinear_iteration;
          response.potential_update = ++potential_update;
          if (self_gravity != nullptr)
            {
              for (const unsigned int index :
              {
                0u, 1u
              })
              {
                const unsigned int degree = (index == 0 ? 2 : 4);
                response.external_load[index] =
                  self_gravity->external_load_surface_potential_coefficient(degree, 0).first;
                response.surface_deformation[index] =
                  self_gravity->surface_deformation_mass_potential_coefficient(degree, 0).first;
                response.cmb[index] =
                  self_gravity->cmb_mass_potential_coefficient(degree, 0).first;
                response.total_surface[index] =
                  self_gravity->total_surface_potential_coefficient(degree, 0).first;
              }

              if (self_gravity->has_full_domain_potential())
                {
                  const auto &geometry =
                    Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
                      simulator_access.get_geometry_model());
                  const double representative_radius =
                    0.5 * (geometry.inner_radius() + geometry.outer_radius());
                  const std::array<double,4> representative =
                    observe_representative_radius_potential(
                      simulator_access, *self_gravity, representative_radius);
                  response.representative_radius_direct =
                  {{representative[0], representative[1]}};
                  response.representative_radius_gram =
                  {{representative[2], representative[3]}};
                }
            }

          if (Utilities::MPI::this_mpi_process(
                simulator_access.get_mpi_communicator()) == 0)
            {
              std::cout << std::scientific << std::setprecision(6);
              const auto print_state =
                [&simulator_access, &response](const char *timing,
                                               const PotentialDiagnosticState &state)
              {
                for (const unsigned int index :
                {
                  0u, 1u
                })
                {
                  const unsigned int degree = (index == 0 ? 2 : 4);
                  std::cout
                      << "Y20DIAG potential"
                      << " timestep=" << simulator_access.get_timestep_number()
                      << " nonlinear="
                      << simulator_access.get_nonlinear_iteration()
                      << " potential=" << response.potential_update
                      << " source_potential=" << state.potential_update
                      << " timing=" << timing
                      << " degree=" << degree
                      << " external_load=" << state.external_load[index]
                      << " surface_deformation="
                      << state.surface_deformation[index]
                      << " cmb=" << state.cmb[index]
                      << " total_surface=" << state.total_surface[index]
                      << " representative_radius_direct="
                      << state.representative_radius_direct[index]
                      << " representative_radius_gram="
                      << state.representative_radius_gram[index]
                      << std::endl;
                }
              };
              if (source_used.valid)
                print_state("source_used", source_used);
              print_state("response", response);
              print_state("next_source", response);
            }

          source_used = response;
          observe_boundary_velocity_harmonics(simulator_access);
        }
    }



    template <int dim>
    void
    connect_y20_boundary_velocity_observer(SimulatorSignals<dim> &signals)
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
          observe_y20_diagnostic(simulator_access);
        });
      });
    }
  }



  ASPECT_REGISTER_SIGNALS_CONNECTOR(
    connect_y20_boundary_velocity_observer<2>,
    connect_y20_boundary_velocity_observer<3>)
}
