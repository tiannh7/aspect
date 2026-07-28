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
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/simulator/assemblers/stokes.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace aspect
{
  namespace
  {
    template <int dim>
    class InternalDensityJumpWorkIdentityAssembler :
      public Assemblers::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void
        initialize_simulator(const Simulator<dim> &simulator_object) override
        {
          SimulatorAccess<dim>::initialize_simulator(simulator_object);
          production.initialize_simulator(simulator_object);
        }

        void
        execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override
        {
          internal::Assembly::Scratch::StokesSystem<dim> &scratch =
            dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
          internal::Assembly::CopyData::StokesSystem<dim> &data =
            dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

          const FullMatrix<double> matrix_before(data.local_matrix);
          const Vector<double> rhs_before(data.local_rhs);

          data.local_matrix = 0;
          data.local_rhs = 0;
          production.execute(scratch_base, data_base);

          const FullMatrix<double> production_matrix(data.local_matrix);
          const Vector<double> production_rhs(data.local_rhs);
          data.local_matrix.add(1.0, matrix_before);
          data.local_rhs += rhs_before;

          FullMatrix<double> expected_matrix(data.local_matrix.m(), data.local_matrix.n());
          Vector<double> expected_rhs(data.local_rhs.size());
          assemble_expected(scratch, data, expected_matrix, expected_rhs);

          double matrix_error = 0.0;
          double matrix_scale = 0.0;
          double rhs_error = 0.0;
          double rhs_scale = 0.0;

          for (unsigned int i = 0; i < data.local_matrix.m(); ++i)
            for (unsigned int j = 0; j < data.local_matrix.n(); ++j)
              {
                const double actual = production_matrix(i,j);
                const double expected = expected_matrix(i,j);
                matrix_error = std::max(matrix_error, std::abs(actual - expected));
                matrix_scale = std::max(matrix_scale, std::abs(expected));
              }

          for (unsigned int i = 0; i < data.local_rhs.size(); ++i)
            {
              const double actual = production_rhs(i);
              const double expected = expected_rhs(i);
              rhs_error = std::max(rhs_error, std::abs(actual - expected));
              rhs_scale = std::max(rhs_scale, std::abs(expected));
            }

          const double matrix_tolerance = 1e-12 * std::max(1.0, matrix_scale);
          const double rhs_tolerance = 1e-12 * std::max(1.0, rhs_scale);
          AssertThrow(matrix_error <= matrix_tolerance,
                      ExcMessage("Internal density jump restoring matrix violates the independent work-identity assembly."));
          AssertThrow(rhs_error <= rhs_tolerance,
                      ExcMessage("Internal density jump restoring RHS violates the independent history work assembly."));

          if (local_face_visits > 0)
            {
              global_face_visits += local_face_visits;
              global_quadrature_visits += local_quadrature_visits;
              global_matrix_error = std::max(global_matrix_error, matrix_error);
              global_rhs_error = std::max(global_rhs_error, rhs_error);
            }
        }

      private:
        void
        assemble_expected(internal::Assembly::Scratch::StokesSystem<dim> &scratch,
                          const internal::Assembly::CopyData::StokesSystem<dim> &data,
                          FullMatrix<double> &expected_matrix,
                          Vector<double> &expected_rhs) const
        {
          local_face_visits = 0;
          local_quadrature_visits = 0;

          const auto &density_sources = this->get_density_source_manager();
          if (!density_sources.has_internal_density_jumps())
            return;

          const Introspection<dim> &introspection = this->introspection();
          const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();
          const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
          const unsigned int displacement_index =
            introspection.compositional_index_for_name("ve_radial_displacement");
          const double mechanical_time_step =
            density_sources.effective_mechanical_time_step();
          const auto &traction_manager = this->get_boundary_traction_manager();
          const PotentialFeedback::SelfGravitation<dim> *self_gravity = nullptr;
          const BoundaryTraction::PotentialFeedbackTraction<dim> *potential_feedback = nullptr;
          if (traction_manager.template has_matching_active_plugin<
              PotentialFeedback::SelfGravitation<dim>>())
            self_gravity = &traction_manager.template get_matching_active_plugin<
                           PotentialFeedback::SelfGravitation<dim>>();
          else if (traction_manager.template has_matching_active_plugin<
                   BoundaryTraction::PotentialFeedbackTraction<dim>>())
            {
              potential_feedback = &traction_manager.template get_matching_active_plugin<
                                   BoundaryTraction::PotentialFeedbackTraction<dim>>();
              if (potential_feedback->has_self_gravity_feedback())
                self_gravity = &potential_feedback->get_self_gravity();
            }

          std::vector<double> committed_displacements(
            scratch.face_finite_element_values.n_quadrature_points);

          for (const unsigned int face_no : scratch.cell->face_indices())
            {
              if (scratch.cell->at_boundary(face_no)
                  || scratch.cell->has_periodic_neighbor(face_no))
                continue;

              const auto neighbor = scratch.cell->neighbor(face_no);
              const Point<dim> inner_cell_center =
                density_sources.radial_cell_representative_point(scratch.cell);
              const Point<dim> outer_cell_center =
                density_sources.radial_cell_representative_point(neighbor);
              if (inner_cell_center.norm() >= outer_cell_center.norm())
                continue;

              const auto face = scratch.cell->face(face_no);
              const double density_contrast =
                density_sources.internal_density_jump_across_face(
                  inner_cell_center,
                  outer_cell_center,
                  face->vertex(0).norm());
              if (density_contrast == 0.0)
                continue;

              bool all_vertices_match = true;
              for (unsigned int vertex = 1; vertex < face->n_vertices(); ++vertex)
                all_vertices_match = all_vertices_match
                                     && (density_sources.internal_density_jump_across_face(
                                           inner_cell_center,
                                           outer_cell_center,
                                           face->vertex(vertex).norm()) == density_contrast);
              if (!all_vertices_match)
                continue;

              scratch.face_finite_element_values.reinit(scratch.cell, face_no);
              scratch.face_finite_element_values
              [introspection.extractors.compositional_fields[displacement_index]]
              .get_function_values(this->get_current_linearization_point(),
                                   committed_displacements);

              ++local_face_visits;
              local_quadrature_visits += scratch.face_finite_element_values.n_quadrature_points;

              for (unsigned int q = 0;
                   q < scratch.face_finite_element_values.n_quadrature_points;
                   ++q)
                {
                  const Point<dim> point =
                    scratch.face_finite_element_values.quadrature_point(q);
                  const double radius = point.norm();
                  AssertThrow(radius > 0.0,
                              ExcMessage("Internal density jump work identity is undefined at radius zero."));
                  const Tensor<1,dim> radial_unit = point / radius;
                  const double gravity_magnitude =
                    this->get_gravity_model().gravity_vector(point).norm();
                  const double potential =
                    (potential_feedback != nullptr
                     ? potential_feedback->full_domain_potential(point)
                     : (self_gravity != nullptr
                        && self_gravity->has_full_domain_potential()
                        ? self_gravity->full_domain_potential(point)
                        : 0.0));
                  const double JxW = scratch.face_finite_element_values.JxW(q);

                  for (unsigned int i = 0, i_stokes = 0;
                       i_stokes < stokes_dofs_per_cell;
                       ++i)
                    if (introspection.is_stokes_component(
                          fe.system_to_component_index(i).first))
                      {
                        const Tensor<1,dim> phi_i =
                          scratch.face_finite_element_values
                          [introspection.extractors.velocities].value(i, q);
                        const double phi_i_radial = phi_i * radial_unit;

                        expected_rhs(i_stokes) -=
                          density_contrast * gravity_magnitude
                          * phi_i_radial * committed_displacements[q] * JxW;
                        expected_rhs(i_stokes) +=
                          density_contrast * potential * phi_i_radial * JxW;

                        if (scratch.rebuild_stokes_matrix)
                          for (unsigned int j = 0, j_stokes = 0;
                               j_stokes < stokes_dofs_per_cell;
                               ++j)
                            if (introspection.is_stokes_component(
                                  fe.system_to_component_index(j).first))
                              {
                                const Tensor<1,dim> phi_j =
                                  scratch.face_finite_element_values
                                  [introspection.extractors.velocities].value(j, q);
                                expected_matrix(i_stokes, j_stokes) +=
                                  density_contrast * gravity_magnitude
                                  * mechanical_time_step * phi_i_radial
                                  * (phi_j * radial_unit) * JxW;
                                ++j_stokes;
                              }

                        ++i_stokes;
                      }
                }
            }
        }

        Assemblers::StokesInternalDensityJumpRestoring<dim> production;
        mutable unsigned int local_face_visits = 0;
        mutable unsigned int local_quadrature_visits = 0;

      public:
        static inline unsigned int global_face_visits = 0;
        static inline unsigned int global_quadrature_visits = 0;
        static inline double global_matrix_error = 0.0;
        static inline double global_rhs_error = 0.0;
    };



    struct InternalSheetConsumerValues
    {
      DensitySourceManager<3>::InternalMassMoments moments;
      DensitySourceManager<3>::InternalMassMoments volume_moments;
      DensitySourceManager<3>::InternalMassMoments sheet_moments;
      std::vector<double> potential_cos = std::vector<double>(5, 0.0);
      std::vector<double> potential_sin = std::vector<double>(5, 0.0);
      std::vector<double> volume_potential_cos = std::vector<double>(5, 0.0);
      std::vector<double> volume_potential_sin = std::vector<double>(5, 0.0);
      std::vector<double> sheet_potential_cos = std::vector<double>(5, 0.0);
      std::vector<double> sheet_potential_sin = std::vector<double>(5, 0.0);
      unsigned int face_visits = 0;
      unsigned int quadrature_visits = 0;
    };



    void
    accumulate_internal_source(
      DensitySourceManager<3>::InternalMassMoments &moments,
      std::vector<double> &potential_cos,
      std::vector<double> &potential_sin,
      const double mass,
      const Point<3> &position,
      const double outer_radius)
    {
      for (unsigned int d = 0; d < 3; ++d)
        moments.mass_dipole[d] += mass * position[d];
      moments.inertia_tensor +=
        mass * (position.norm_square() * unit_symmetric_tensor<3>()
                - symmetrize(outer_product(position, position)));

      const double radius = position.norm();
      AssertThrow(radius > 0.0, ExcInternalError());
      const std::array<double,3> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      unsigned int coefficient_index = 0;
      for (unsigned int degree = 1; degree <= 2; ++degree)
        {
          const double radial_kernel =
            (1.0 / radius) * Utilities::pow(radius / outer_radius, degree + 1);
          for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
            {
              const std::pair<double,double> spherical_harmonic =
                Utilities::real_spherical_harmonic(degree,
                                                   order,
                                                   spherical_coordinates[2],
                                                   spherical_coordinates[1]);
              potential_cos[coefficient_index] +=
                mass * radial_kernel * spherical_harmonic.first;
              potential_sin[coefficient_index] +=
                mass * radial_kernel * spherical_harmonic.second;
            }
        }
    }



    void
    accumulate_internal_source(InternalSheetConsumerValues &values,
                               const bool is_volume_source,
                               const double mass,
                               const Point<3> &position,
                               const double outer_radius)
    {
      accumulate_internal_source(values.moments,
                                 values.potential_cos,
                                 values.potential_sin,
                                 mass,
                                 position,
                                 outer_radius);
      accumulate_internal_source(is_volume_source
                                 ? values.volume_moments
                                 : values.sheet_moments,
                                 is_volume_source
                                 ? values.volume_potential_cos
                                 : values.sheet_potential_cos,
                                 is_volume_source
                                 ? values.volume_potential_sin
                                 : values.sheet_potential_sin,
                                 mass,
                                 position,
                                 outer_radius);
    }



    template <int dim>
    InternalSheetConsumerValues
    compute_expected_internal_sheet_consumers(
      const SimulatorAccess<dim> &simulator_access,
      const Quadrature<dim-1> &face_quadrature,
      const unsigned int volume_quadrature_degree,
      const double outer_radius,
      const bool include_current_velocity_increment,
      const bool use_production_accessors = false,
      const Mapping<dim> *mapping_override = nullptr)
    {
      AssertThrow(dim == 3, ExcNotImplemented());
      InternalSheetConsumerValues values;
      const auto &density_sources = simulator_access.get_density_source_manager();
      const auto &introspection = simulator_access.introspection();
      const Mapping<dim> &mapping =
        (mapping_override != nullptr
         ? *mapping_override
         : simulator_access.get_mapping());

      const QGauss<dim> volume_quadrature(volume_quadrature_degree);
      FEValues<dim> fe_values(mapping,
                              simulator_access.get_fe(),
                              volume_quadrature,
                              update_values |
                              update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
      MaterialModel::MaterialModelInputs<dim> inputs(volume_quadrature.size(),
                                                     simulator_access.n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> outputs(volume_quadrature.size(),
                                                       simulator_access.n_compositional_fields());
      density_sources.create_additional_material_model_outputs(outputs);
      inputs.requested_properties = MaterialModel::MaterialProperties::density;

      for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            inputs.reinit(fe_values,
                          cell,
                          introspection,
                          simulator_access.get_solution());
            simulator_access.get_material_model().evaluate(inputs, outputs);
            for (unsigned int q = 0; q < volume_quadrature.size(); ++q)
              accumulate_internal_source(
                values,
                true,
                (use_production_accessors
                 ? density_sources.self_gravity_source_density(
                   inputs,
                   outputs,
                   q,
                   0.0,
                   include_current_velocity_increment)
                 : (density_sources.reference_density(inputs.position[q])
                    * inputs.pressure[q]
                    / density_sources.elastic_bulk_modulus(outputs, q)
                    - (inputs.composition[q][
                         introspection.compositional_index_for_name(
                           "ve_radial_displacement")]
                       + (include_current_velocity_increment
                          ? density_sources.effective_mechanical_time_step()
                          * (inputs.velocity[q]
                             * (inputs.position[q]
                                / inputs.position[q].norm()))
                          : 0.0))
                    * (density_sources.reference_density_gradient(
                         inputs.position[q])
                       * (inputs.position[q]
                          / inputs.position[q].norm()))))
                * fe_values.JxW(q),
                fe_values.quadrature_point(q),
                outer_radius);
          }

      FEFaceValues<dim> face_values(mapping,
                                    simulator_access.get_fe(),
                                    face_quadrature,
                                    update_values |
                                    update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);
      MaterialModel::MaterialModelInputs<dim> face_inputs(face_quadrature.size(),
                                                          simulator_access.n_compositional_fields());

      for (const auto &cell : simulator_access.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
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
                  inner_cell_center, outer_cell_center, face->vertex(0).norm());
              if (density_contrast == 0.0)
                continue;

              bool all_vertices_match = true;
              for (unsigned int vertex = 1; vertex < face->n_vertices(); ++vertex)
                all_vertices_match =
                  all_vertices_match
                  && (density_sources.internal_density_jump_across_face(
                        inner_cell_center,
                        outer_cell_center,
                        face->vertex(vertex).norm()) == density_contrast);
              if (!all_vertices_match)
                continue;

              face_values.reinit(cell, face_no);
              face_inputs.reinit(face_values,
                                 cell,
                                 introspection,
                                 simulator_access.get_solution());
              ++values.face_visits;
              values.quadrature_visits += face_quadrature.size();
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                accumulate_internal_source(
                  values,
                  false,
                  density_contrast
                  * (use_production_accessors
                     ? density_sources.mechanical_radial_displacement(
                       face_inputs,
                       q,
                       include_current_velocity_increment)
                     : (face_inputs.composition[q][
                          introspection.compositional_index_for_name(
                            "ve_radial_displacement")]
                        + (include_current_velocity_increment
                           ? density_sources.effective_mechanical_time_step()
                           * (face_inputs.velocity[q]
                              * (face_inputs.position[q]
                                 / face_inputs.position[q].norm()))
                           : 0.0)))
                  * face_values.JxW(q),
                  face_inputs.position[q],
                  outer_radius);
            }

      const MPI_Comm mpi_communicator =
        simulator_access.get_mpi_communicator();
      values.moments.mass_dipole =
        Utilities::MPI::sum(values.moments.mass_dipole, mpi_communicator);
      values.moments.inertia_tensor =
        Utilities::MPI::sum(values.moments.inertia_tensor, mpi_communicator);
      values.volume_moments.mass_dipole =
        Utilities::MPI::sum(values.volume_moments.mass_dipole,
                            mpi_communicator);
      values.volume_moments.inertia_tensor =
        Utilities::MPI::sum(values.volume_moments.inertia_tensor,
                            mpi_communicator);
      values.sheet_moments.mass_dipole =
        Utilities::MPI::sum(values.sheet_moments.mass_dipole,
                            mpi_communicator);
      values.sheet_moments.inertia_tensor =
        Utilities::MPI::sum(values.sheet_moments.inertia_tensor,
                            mpi_communicator);
      Utilities::MPI::sum(values.potential_cos,
                          mpi_communicator,
                          values.potential_cos);
      Utilities::MPI::sum(values.potential_sin,
                          mpi_communicator,
                          values.potential_sin);
      Utilities::MPI::sum(values.volume_potential_cos,
                          mpi_communicator,
                          values.volume_potential_cos);
      Utilities::MPI::sum(values.volume_potential_sin,
                          mpi_communicator,
                          values.volume_potential_sin);
      Utilities::MPI::sum(values.sheet_potential_cos,
                          mpi_communicator,
                          values.sheet_potential_cos);
      Utilities::MPI::sum(values.sheet_potential_sin,
                          mpi_communicator,
                          values.sheet_potential_sin);
      values.face_visits =
        Utilities::MPI::sum(values.face_visits,
                            simulator_access.get_mpi_communicator());
      values.quadrature_visits =
        Utilities::MPI::sum(values.quadrature_visits,
                            simulator_access.get_mpi_communicator());
      return values;
    }



    template <int dim>
    InternalSheetConsumerValues
    compute_production_internal_consumers(
      const SimulatorAccess<dim> &simulator_access,
      const double outer_radius,
      const bool include_current_velocity_increment)
    {
      InternalSheetConsumerValues values;
      simulator_access.get_density_source_manager().for_each_internal_mass_source(
        0.0,
        "quadrature point",
        [&values, outer_radius](const double mass, const Point<dim> &position)
      {
        accumulate_internal_source(values.moments,
                                   values.potential_cos,
                                   values.potential_sin,
                                   mass,
                                   position,
                                   outer_radius);
      },
      include_current_velocity_increment);

      values.moments.mass_dipole =
        Utilities::MPI::sum(values.moments.mass_dipole,
                            simulator_access.get_mpi_communicator());
      values.moments.inertia_tensor =
        Utilities::MPI::sum(values.moments.inertia_tensor,
                            simulator_access.get_mpi_communicator());
      Utilities::MPI::sum(values.potential_cos,
                          simulator_access.get_mpi_communicator(),
                          values.potential_cos);
      Utilities::MPI::sum(values.potential_sin,
                          simulator_access.get_mpi_communicator(),
                          values.potential_sin);
      return values;
    }



    double
    moment_difference(
      const DensitySourceManager<3>::InternalMassMoments &left,
      const DensitySourceManager<3>::InternalMassMoments &right)
    {
      double difference = 0.0;
      for (unsigned int i = 0; i < 3; ++i)
        {
          difference = std::max(difference,
                                std::abs(left.mass_dipole[i]
                                         - right.mass_dipole[i]));
          for (unsigned int j = 0; j < 3; ++j)
            difference = std::max(difference,
                                  std::abs(left.inertia_tensor[i][j]
                                           - right.inertia_tensor[i][j]));
        }
      return difference;
    }



    double
    coefficient_difference(const std::vector<double> &left_cos,
                           const std::vector<double> &left_sin,
                           const std::vector<double> &right_cos,
                           const std::vector<double> &right_sin)
    {
      AssertDimension(left_cos.size(), right_cos.size());
      AssertDimension(left_sin.size(), right_sin.size());
      double difference = 0.0;
      for (unsigned int i = 0; i < left_cos.size(); ++i)
        difference = std::max(
                       difference,
                       std::max(std::abs(left_cos[i] - right_cos[i]),
                                std::abs(left_sin[i] - right_sin[i])));
      return difference;
    }



    double
    evaluate_outer_radius_potential(const InternalSheetConsumerValues &values,
                                    const Point<3> &position,
                                    const double outer_radius,
                                    const bool volume_only = false,
                                    const bool sheet_only = false)
    {
      AssertThrow(std::abs(position.norm() - outer_radius)
                  <= 1e-14 * outer_radius,
                  ExcMessage("The independent cache oracle requires an outer-radius support point."));
      const std::vector<double> &cos_coefficients =
        (volume_only ? values.volume_potential_cos
         : (sheet_only ? values.sheet_potential_cos : values.potential_cos));
      const std::vector<double> &sin_coefficients =
        (volume_only ? values.volume_potential_sin
         : (sheet_only ? values.sheet_potential_sin : values.potential_sin));
      const std::array<double,3> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

      double potential = 0.0;
      unsigned int coefficient_index = 0;
      for (unsigned int degree = 1; degree <= 2; ++degree)
        {
          const double scale = 4.0 * numbers::PI * constants::big_g
                               / (2.0 * degree + 1.0);
          for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
            {
              const std::pair<double,double> spherical_harmonic =
                Utilities::real_spherical_harmonic(degree,
                                                   order,
                                                   spherical_coordinates[2],
                                                   spherical_coordinates[1]);
              potential += scale
                           * (cos_coefficients[coefficient_index]
                              * spherical_harmonic.first
                              + sin_coefficients[coefficient_index]
                              * spherical_harmonic.second);
            }
        }
      return potential;
    }



    struct BoundarySourceValues
    {
      std::vector<double> surface_cos = std::vector<double>(15, 0.0);
      std::vector<double> surface_sin = std::vector<double>(15, 0.0);
      std::vector<double> cmb_cos = std::vector<double>(15, 0.0);
      std::vector<double> cmb_sin = std::vector<double>(15, 0.0);
      unsigned int surface_face_visits = 0;
      unsigned int cmb_face_visits = 0;
    };



    void
    accumulate_boundary_source(std::vector<double> &cos_coefficients,
                               std::vector<double> &sin_coefficients,
                               const Point<3> &position,
                               const double weighted_topography)
    {
      const std::array<double,3> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      unsigned int coefficient_index = 0;
      for (unsigned int degree = 0; degree <= 4; ++degree)
        for (unsigned int order = 0; order <= degree;
             ++order, ++coefficient_index)
          {
            const std::pair<double,double> spherical_harmonic =
              Utilities::real_spherical_harmonic(
                degree,
                order,
                spherical_coordinates[2],
                spherical_coordinates[1]);
            cos_coefficients[coefficient_index] +=
              weighted_topography * spherical_harmonic.first;
            sin_coefficients[coefficient_index] +=
              weighted_topography * spherical_harmonic.second;
          }
    }



    template <int dim>
    BoundarySourceValues
    compute_expected_boundary_sources(
      const SimulatorAccess<dim> &simulator_access,
      const Mapping<dim> &mapping,
      const double surface_density_contrast,
      const double cmb_density_contrast)
    {
      AssertThrow(dim == 3, ExcNotImplemented());
      BoundarySourceValues values;
      const auto &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          simulator_access.get_geometry_model());
      const types::boundary_id top_boundary_id =
        simulator_access.get_geometry_model()
        .translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id =
        simulator_access.get_geometry_model()
        .translate_symbolic_boundary_name_to_id("bottom");
      const unsigned int quadrature_degree =
        std::max(2u,
                 simulator_access.introspection().polynomial_degree.velocities
                 + 1u);
      const QGauss<dim-1> face_quadrature(quadrature_degree);
      FEFaceValues<dim> face_values(mapping,
                                    simulator_access.get_fe(),
                                    face_quadrature,
                                    update_quadrature_points |
                                    update_JxW_values);

      for (const auto &cell : simulator_access.get_dof_handler()
           .active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          for (const unsigned int face_no : cell->face_indices())
            if (cell->at_boundary(face_no))
              {
                const types::boundary_id boundary_id =
                  cell->face(face_no)->boundary_id();
                if (boundary_id != top_boundary_id
                    && boundary_id != bottom_boundary_id)
                  continue;

                face_values.reinit(cell, face_no);
                if (boundary_id == top_boundary_id)
                  ++values.surface_face_visits;
                else
                  ++values.cmb_face_visits;

                for (unsigned int q = 0;
                     q < face_values.n_quadrature_points;
                     ++q)
                  {
                    const Point<dim> position =
                      face_values.quadrature_point(q);
                    if (boundary_id == top_boundary_id)
                      accumulate_boundary_source(
                        values.surface_cos,
                        values.surface_sin,
                        position,
                        surface_density_contrast
                        * geometry.height_above_reference_surface(position)
                        * face_values.JxW(q)
                        / Utilities::fixed_power<2>(geometry.outer_radius()));
                    else
                      accumulate_boundary_source(
                        values.cmb_cos,
                        values.cmb_sin,
                        position,
                        cmb_density_contrast
                        * (position.norm() - geometry.inner_radius())
                        * face_values.JxW(q)
                        / Utilities::fixed_power<2>(geometry.inner_radius()));
                  }
              }

      const MPI_Comm mpi_communicator =
        simulator_access.get_mpi_communicator();
      Utilities::MPI::sum(values.surface_cos,
                          mpi_communicator,
                          values.surface_cos);
      Utilities::MPI::sum(values.surface_sin,
                          mpi_communicator,
                          values.surface_sin);
      Utilities::MPI::sum(values.cmb_cos,
                          mpi_communicator,
                          values.cmb_cos);
      Utilities::MPI::sum(values.cmb_sin,
                          mpi_communicator,
                          values.cmb_sin);
      values.surface_face_visits =
        Utilities::MPI::sum(values.surface_face_visits, mpi_communicator);
      values.cmb_face_visits =
        Utilities::MPI::sum(values.cmb_face_visits, mpi_communicator);
      return values;
    }



    double
    degree_coefficient_difference(const std::vector<double> &left_cos,
                                  const std::vector<double> &left_sin,
                                  const std::vector<double> &right_cos,
                                  const std::vector<double> &right_sin,
                                  const unsigned int selected_degree)
    {
      double difference = 0.0;
      unsigned int coefficient_index = 0;
      for (unsigned int degree = 0; degree <= 4; ++degree)
        for (unsigned int order = 0; order <= degree;
             ++order, ++coefficient_index)
          if (degree == selected_degree)
            difference = std::max(
                           difference,
                           std::max(
                             std::abs(left_cos[coefficient_index]
                                      - right_cos[coefficient_index]),
                             std::abs(left_sin[coefficient_index]
                                      - right_sin[coefficient_index])));
      return difference;
    }



    template <int dim>
    void
    validate_zero_increment_geometry_remap(
      const SimulatorAccess<dim> &simulator_access)
    {
      if constexpr (dim != 3)
        return;

      static bool validated = false;
      if (validated || simulator_access.get_timestep_number() == 0)
        return;
      validated = true;

      const auto &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          simulator_access.get_geometry_model());
      const auto &mesh_handler =
        simulator_access.get_mesh_deformation_handler();
      const auto &mesh_dof_handler =
        mesh_handler.get_mesh_deformation_dof_handler();
      LinearAlgebra::Vector zero_displacement;
      zero_displacement.reinit(mesh_handler.get_mesh_displacements());
      zero_displacement = 0.0;
      MappingQEulerian<dim,LinearAlgebra::Vector> reference_mapping(
        simulator_access.get_geometry_model().has_curved_elements()
        ? mesh_dof_handler.get_fe().degree + 1
        : 1,
        mesh_dof_handler,
        zero_displacement);

      const auto &face_quadrature =
        simulator_access.introspection().face_quadratures.velocities;
      const unsigned int volume_quadrature_degree =
        std::max(2u,
                 simulator_access.introspection().polynomial_degree.temperature
                 + 1u);
      const InternalSheetConsumerValues current =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          face_quadrature,
          volume_quadrature_degree,
          geometry.outer_radius(),
          false);
      const InternalSheetConsumerValues reference =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          face_quadrature,
          volume_quadrature_degree,
          geometry.outer_radius(),
          false,
          false,
          &reference_mapping);

      const double total_difference =
        coefficient_difference(current.potential_cos,
                               current.potential_sin,
                               reference.potential_cos,
                               reference.potential_sin);
      const double volume_difference =
        coefficient_difference(current.volume_potential_cos,
                               current.volume_potential_sin,
                               reference.volume_potential_cos,
                               reference.volume_potential_sin);
      const double sheet_difference =
        coefficient_difference(current.sheet_potential_cos,
                               current.sheet_potential_sin,
                               reference.sheet_potential_cos,
                               reference.sheet_potential_sin);
      const double surface_density_contrast = 3700.0;
      const double cmb_density_contrast = 4100.0;
      const BoundarySourceValues current_boundaries =
        compute_expected_boundary_sources(simulator_access,
                                          simulator_access.get_mapping(),
                                          surface_density_contrast,
                                          cmb_density_contrast);
      const BoundarySourceValues reference_boundaries =
        compute_expected_boundary_sources(simulator_access,
                                          reference_mapping,
                                          surface_density_contrast,
                                          cmb_density_contrast);
      const double surface_difference =
        coefficient_difference(current_boundaries.surface_cos,
                               current_boundaries.surface_sin,
                               reference_boundaries.surface_cos,
                               reference_boundaries.surface_sin);
      const double cmb_difference =
        coefficient_difference(current_boundaries.cmb_cos,
                               current_boundaries.cmb_sin,
                               reference_boundaries.cmb_cos,
                               reference_boundaries.cmb_sin);
      const double surface_l4_difference =
        degree_coefficient_difference(current_boundaries.surface_cos,
                                      current_boundaries.surface_sin,
                                      reference_boundaries.surface_cos,
                                      reference_boundaries.surface_sin,
                                      4);
      const double cmb_l4_difference =
        degree_coefficient_difference(current_boundaries.cmb_cos,
                                      current_boundaries.cmb_sin,
                                      reference_boundaries.cmb_cos,
                                      reference_boundaries.cmb_sin,
                                      4);
      std::cout << std::scientific << std::setprecision(6)
                << "Internal-source geometry remap: timestep="
                << simulator_access.get_timestep_number()
                << ", total_difference=" << total_difference
                << ", volume_difference=" << volume_difference
                << ", sheet_difference=" << sheet_difference
                << std::endl
                << "Density-weighted boundary-source geometry remap: timestep="
                << simulator_access.get_timestep_number()
                << ", surface_density_contrast=" << surface_density_contrast
                << ", cmb_density_contrast=" << cmb_density_contrast
                << ", surface_difference=" << surface_difference
                << ", surface_l4_difference=" << surface_l4_difference
                << ", cmb_difference=" << cmb_difference
                << ", cmb_l4_difference=" << cmb_l4_difference
                << std::endl;

      AssertThrow(current.face_visits > 0 && reference.face_visits > 0,
                  ExcMessage("The geometry-remap oracle visited no internal sheet."));
      AssertThrow(total_difference > 0.0,
                  ExcMessage("The geometry-remap oracle did not distinguish current and reference source geometry."));
      AssertThrow(current_boundaries.surface_face_visits > 0
                  && reference_boundaries.surface_face_visits > 0
                  && current_boundaries.cmb_face_visits > 0
                  && reference_boundaries.cmb_face_visits > 0,
                  ExcMessage("The boundary geometry-remap oracle visited no surface or CMB face."));
      AssertThrow(surface_difference > 0.0 && cmb_difference > 0.0,
                  ExcMessage("The boundary geometry-remap oracle did not distinguish current and reference source geometry."));
    }



    template <int dim>
    void
    validate_self_gravity_cache_state(
      const SimulatorAccess<dim> &simulator_access,
      const bool include_current_velocity_increment,
      const char *const phase)
    {
      AssertThrow(dim == 3,
                  ExcMessage("The self-gravity cache state observer requires 3D."));
      const auto &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          simulator_access.get_geometry_model());
      const double outer_radius = geometry.outer_radius();
      const auto &potential_feedback =
        simulator_access.get_boundary_traction_manager()
        .template get_matching_active_plugin<
        BoundaryTraction::PotentialFeedbackTraction<dim>>();
      const auto &self_gravity = potential_feedback.get_self_gravity();
      AssertThrow(self_gravity.has_full_domain_potential(),
                  ExcMessage("The production full-domain self-gravity cache was not populated."));

      const unsigned int temperature_degree =
        simulator_access.introspection().polynomial_degree.temperature;
      const auto &velocity_face_quadrature =
        simulator_access.introspection().face_quadratures.velocities;
      const unsigned int volume_quadrature_degree =
        std::max(2u, temperature_degree + 1u);
      const InternalSheetConsumerValues expected =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          velocity_face_quadrature,
          volume_quadrature_degree,
          outer_radius,
          include_current_velocity_increment);
      const InternalSheetConsumerValues counterfactual =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          velocity_face_quadrature,
          volume_quadrature_degree,
          outer_radius,
          !include_current_velocity_increment);

      const std::array<Point<3>,2> sample_points =
      {
        {
          Point<3>(outer_radius / std::sqrt(2.0),
          outer_radius / std::sqrt(3.0),
          outer_radius / std::sqrt(6.0)),
          Point<3>(-outer_radius / std::sqrt(3.0),
          outer_radius / std::sqrt(2.0),
          outer_radius / std::sqrt(6.0))
        }
      };

      double cache_error = 0.0;
      double cache_scale = 0.0;
      double zero_source_difference = 0.0;
      double state_counterfactual_difference = 0.0;
      double volume_magnitude = 0.0;
      double sheet_magnitude = 0.0;
      for (const Point<3> &point : sample_points)
        {
          const double actual = self_gravity.full_domain_potential(point);
          const double expected_potential =
            evaluate_outer_radius_potential(expected, point, outer_radius);
          const double counterfactual_potential =
            evaluate_outer_radius_potential(counterfactual, point, outer_radius);
          const double volume_potential =
            evaluate_outer_radius_potential(expected, point, outer_radius, true);
          const double sheet_potential =
            evaluate_outer_radius_potential(expected, point, outer_radius, false, true);
          cache_error = std::max(cache_error,
                                 std::abs(actual - expected_potential));
          cache_scale = std::max(cache_scale, std::abs(expected_potential));
          zero_source_difference = std::max(zero_source_difference,
                                            std::abs(expected_potential));
          state_counterfactual_difference =
            std::max(state_counterfactual_difference,
                     std::abs(expected_potential - counterfactual_potential));
          volume_magnitude = std::max(volume_magnitude, std::abs(volume_potential));
          sheet_magnitude = std::max(sheet_magnitude, std::abs(sheet_potential));
        }

      const double tolerance = 1e-11 * cache_scale;
      std::cout << std::scientific << std::setprecision(6)
                << "Self-gravity cache state: phase=" << phase
                << ", timestep=" << simulator_access.get_timestep_number()
                << ", cache_error=" << cache_error
                << ", zero_source_difference=" << zero_source_difference
                << ", state_counterfactual_difference="
                << state_counterfactual_difference
                << ", volume_magnitude=" << volume_magnitude
                << ", sheet_magnitude=" << sheet_magnitude
                << std::endl;

      AssertThrow(expected.face_visits > 0,
                  ExcMessage("The self-gravity cache observer visited no internal jump face."));
      AssertThrow(cache_error <= tolerance,
                  ExcMessage("The production full-domain self-gravity cache disagrees with the independent internal-source Green oracle."));
      AssertThrow(volume_magnitude > 100.0 * tolerance
                  && sheet_magnitude > 100.0 * tolerance,
                  ExcMessage("The self-gravity cache regression does not distinguish volume and sheet internal sources."));
      AssertThrow(zero_source_difference > 100.0 * tolerance,
                  ExcMessage("The self-gravity cache regression does not distinguish the old pre-Stokes zero-source state."));
      if (include_current_velocity_increment
          || simulator_access.get_timestep_number() != 0)
        AssertThrow(state_counterfactual_difference > 100.0 * tolerance,
                    ExcMessage("The self-gravity cache regression does not distinguish committed and trial source states."));
    }



    template <int dim>
    class SelfGravityCacheStateObserver :
      public BoundaryTraction::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim>
        boundary_traction(const types::boundary_id,
                          const Point<dim> &,
                          const Tensor<1,dim> &) const override
        {
          return Tensor<1,dim>();
        }

        void
        initialize() override
        {
          this->get_signals().post_mesh_deformation.connect(
            [this](const SimulatorAccess<dim> &)
          {
            if constexpr (dim == 3)
              validate_zero_increment_geometry_remap(*this);
          });
          this->get_signals().post_stokes_solver.connect(
            [this](const SimulatorAccess<dim> &,
                   const unsigned int,
                   const unsigned int,
                   const SolverControl &,
                   const SolverControl &)
          {
            if constexpr (dim == 3)
              validate_self_gravity_cache_state(*this, true, "post-Stokes");
          });
        }

        void
        update() override
        {
          if constexpr (dim == 3)
            validate_self_gravity_cache_state(*this, false, "pre-Stokes");
        }
    };



    template <int dim>
    void
    validate_internal_sheet_consumers(const SimulatorAccess<dim> &simulator_access)
    {
      static bool validated_timestep_zero = false;
      static bool validated_nonzero_timestep = false;
      if (validated_nonzero_timestep)
        return;

      AssertThrow(dim == 3,
                  ExcMessage("The displaced internal-sheet consumer test requires 3D."));
      const auto &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          simulator_access.get_geometry_model());
      const double outer_radius = geometry.outer_radius();
      const auto &density_sources = simulator_access.get_density_source_manager();
      const auto &traction_manager = simulator_access.get_boundary_traction_manager();

      const PotentialFeedback::SelfGravitation<dim> *self_gravity = nullptr;
      if (traction_manager.template has_matching_active_plugin<
          PotentialFeedback::SelfGravitation<dim>>())
        self_gravity = &traction_manager.template get_matching_active_plugin<
                       PotentialFeedback::SelfGravitation<dim>>();
      else if (traction_manager.template has_matching_active_plugin<
               BoundaryTraction::PotentialFeedbackTraction<dim>>())
        {
          const auto &potential_feedback =
            traction_manager.template get_matching_active_plugin<
            BoundaryTraction::PotentialFeedbackTraction<dim>>();
          AssertThrow(potential_feedback.has_self_gravity_feedback(),
                      ExcMessage("The active potential-feedback adapter has no self gravity."));
          self_gravity = &potential_feedback.get_self_gravity();
        }
      AssertThrow(self_gravity != nullptr,
                  ExcMessage("No active self-gravity feedback was found."));
      AssertThrow(self_gravity->minimum_degree() == 1,
                  ExcMessage("The displaced-sheet test expects minimum self-gravity degree one."));

      const unsigned int temperature_degree =
        simulator_access.introspection().polynomial_degree.temperature;
      const unsigned int moment_volume_quadrature_degree =
        std::max(2u, temperature_degree + 1u);
      const bool is_timestep_zero =
        simulator_access.get_timestep_number() == 0;
      if (is_timestep_zero && validated_timestep_zero)
        return;
      if (is_timestep_zero)
        validated_timestep_zero = true;
      else
        validated_nonzero_timestep = true;
      const auto &velocity_face_quadrature =
        simulator_access.introspection().face_quadratures.velocities;
      const InternalSheetConsumerValues expected_moments =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          velocity_face_quadrature,
          moment_volume_quadrature_degree,
          outer_radius,
          true);
      const InternalSheetConsumerValues expected_potential =
        compute_expected_internal_sheet_consumers(simulator_access,
                                                  velocity_face_quadrature,
                                                  temperature_degree,
                                                  outer_radius,
                                                  true,
                                                  true);
      const InternalSheetConsumerValues expected_committed =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          velocity_face_quadrature,
          moment_volume_quadrature_degree,
          outer_radius,
          false);
      const InternalSheetConsumerValues actual_trial =
        compute_production_internal_consumers(simulator_access,
                                              outer_radius,
                                              true);
      const InternalSheetConsumerValues actual_committed =
        compute_production_internal_consumers(simulator_access,
                                              outer_radius,
                                              false);
      const QGauss<dim-1> old_face_quadrature(temperature_degree);
      const InternalSheetConsumerValues old_expected_moments =
        compute_expected_internal_sheet_consumers(
          simulator_access,
          old_face_quadrature,
          moment_volume_quadrature_degree,
          outer_radius,
          true);
      const InternalSheetConsumerValues old_expected_potential =
        compute_expected_internal_sheet_consumers(simulator_access,
                                                  old_face_quadrature,
                                                  temperature_degree,
                                                  outer_radius,
                                                  true);
      const auto actual_moments =
        density_sources.compute_internal_mass_moments(0.0,
                                                      "quadrature point",
                                                      true);
      const auto actual_committed_moments =
        density_sources.compute_internal_mass_moments(0.0,
                                                      "quadrature point",
                                                      false);
      const auto actual_potential =
        self_gravity->compute_internal_density_potential(outer_radius);

      AssertDimension(actual_potential.first.size(), expected_potential.potential_cos.size());
      AssertDimension(actual_potential.second.size(), expected_potential.potential_sin.size());
      double dipole_error = 0.0;
      double inertia_error = 0.0;
      double potential_error = 0.0;
      double old_moment_difference = 0.0;
      double old_potential_difference = 0.0;
      double moment_scale = 1.0;
      double potential_scale = 1.0;
      for (unsigned int i = 0; i < 3; ++i)
        {
          dipole_error = std::max(dipole_error,
                                  std::abs(actual_moments.mass_dipole[i]
                                           - expected_moments.moments.mass_dipole[i]));
          old_moment_difference = std::max(old_moment_difference,
                                           std::abs(old_expected_moments.moments.mass_dipole[i]
                                                    - expected_moments.moments.mass_dipole[i]));
          moment_scale = std::max(moment_scale,
                                  std::abs(expected_moments.moments.mass_dipole[i]));
          for (unsigned int j = 0; j < 3; ++j)
            {
              inertia_error = std::max(inertia_error,
                                       std::abs(actual_moments.inertia_tensor[i][j]
                                                - expected_moments.moments.inertia_tensor[i][j]));
              old_moment_difference = std::max(old_moment_difference,
                                               std::abs(old_expected_moments.moments.inertia_tensor[i][j]
                                                        - expected_moments.moments.inertia_tensor[i][j]));
              moment_scale = std::max(moment_scale,
                                      std::abs(expected_moments.moments.inertia_tensor[i][j]));
            }
        }
      for (unsigned int i = 0; i < expected_potential.potential_cos.size(); ++i)
        {
          potential_error = std::max(
                              potential_error,
                              std::max(std::abs(actual_potential.first[i] - expected_potential.potential_cos[i]),
                                       std::abs(actual_potential.second[i] - expected_potential.potential_sin[i])));
          old_potential_difference = std::max(
                                       old_potential_difference,
                                       std::max(std::abs(old_expected_potential.potential_cos[i] - expected_potential.potential_cos[i]),
                                                std::abs(old_expected_potential.potential_sin[i] - expected_potential.potential_sin[i])));
          potential_scale = std::max(
                              potential_scale,
                              std::max(std::abs(expected_potential.potential_cos[i]),
                                       std::abs(expected_potential.potential_sin[i])));
        }

      const double moment_tolerance = 1e-11 * moment_scale;
      const double potential_tolerance = 1e-11 * potential_scale;
      const double committed_moment_error =
        moment_difference(actual_committed_moments,
                          expected_committed.moments);
      const double production_trial_moment_error =
        moment_difference(actual_trial.moments, expected_moments.moments);
      const double production_committed_moment_error =
        moment_difference(actual_committed.moments,
                          expected_committed.moments);
      const double production_trial_coefficient_error =
        coefficient_difference(actual_trial.potential_cos,
                               actual_trial.potential_sin,
                               expected_moments.potential_cos,
                               expected_moments.potential_sin);
      const double production_committed_coefficient_error =
        coefficient_difference(actual_committed.potential_cos,
                               actual_committed.potential_sin,
                               expected_committed.potential_cos,
                               expected_committed.potential_sin);
      const double volume_delta =
        moment_difference(expected_moments.volume_moments,
                          expected_committed.volume_moments);
      const double sheet_delta =
        moment_difference(expected_moments.sheet_moments,
                          expected_committed.sheet_moments);
      DensitySourceManager<3>::InternalMassMoments actual_delta_moments;
      DensitySourceManager<3>::InternalMassMoments expected_delta_moments;
      for (unsigned int i = 0; i < 3; ++i)
        {
          actual_delta_moments.mass_dipole[i] =
            actual_trial.moments.mass_dipole[i]
            - actual_committed.moments.mass_dipole[i];
          expected_delta_moments.mass_dipole[i] =
            expected_moments.moments.mass_dipole[i]
            - expected_committed.moments.mass_dipole[i];
          for (unsigned int j = 0; j < 3; ++j)
            {
              actual_delta_moments.inertia_tensor[i][j] =
                actual_trial.moments.inertia_tensor[i][j]
                - actual_committed.moments.inertia_tensor[i][j];
              expected_delta_moments.inertia_tensor[i][j] =
                expected_moments.moments.inertia_tensor[i][j]
                - expected_committed.moments.inertia_tensor[i][j];
            }
        }
      const double delta_error =
        moment_difference(actual_delta_moments, expected_delta_moments);
      const double zero_committed_counterfactual =
        moment_difference(actual_committed.moments,
                          DensitySourceManager<3>::InternalMassMoments());
      const double reused_trial_counterfactual =
        moment_difference(actual_committed.moments,
                          expected_moments.moments);
      const double omitted_volume_counterfactual =
        coefficient_difference(expected_moments.potential_cos,
                               expected_moments.potential_sin,
                               expected_moments.sheet_potential_cos,
                               expected_moments.sheet_potential_sin);
      const double omitted_sheet_counterfactual =
        coefficient_difference(expected_moments.potential_cos,
                               expected_moments.potential_sin,
                               expected_moments.volume_potential_cos,
                               expected_moments.volume_potential_sin);
      std::vector<double> wrong_sign_cos(expected_moments.potential_cos.size());
      std::vector<double> wrong_sign_sin(expected_moments.potential_sin.size());
      std::vector<double> recombined_cos(expected_moments.potential_cos.size());
      std::vector<double> recombined_sin(expected_moments.potential_sin.size());
      for (unsigned int i = 0; i < wrong_sign_cos.size(); ++i)
        {
          wrong_sign_cos[i] = expected_moments.volume_potential_cos[i]
                              - expected_moments.sheet_potential_cos[i];
          wrong_sign_sin[i] = expected_moments.volume_potential_sin[i]
                              - expected_moments.sheet_potential_sin[i];
          recombined_cos[i] = expected_moments.volume_potential_cos[i]
                              + expected_moments.sheet_potential_cos[i];
          recombined_sin[i] = expected_moments.volume_potential_sin[i]
                              + expected_moments.sheet_potential_sin[i];
        }
      const double wrong_sign_counterfactual =
        coefficient_difference(expected_moments.potential_cos,
                               expected_moments.potential_sin,
                               wrong_sign_cos,
                               wrong_sign_sin);
      const double combined_source_reciprocity_error =
        coefficient_difference(expected_moments.potential_cos,
                               expected_moments.potential_sin,
                               recombined_cos,
                               recombined_sin);
      std::cout << std::scientific << std::setprecision(6)
                << "Internal displaced-sheet consumers: faces="
                << expected_moments.face_visits
                << ", q=" << expected_moments.quadrature_visits
                << ", dipole_error=" << dipole_error
                << ", inertia_error=" << inertia_error
                << ", potential_error=" << potential_error
                << ", old_moment_difference=" << old_moment_difference
                << ", old_potential_difference=" << old_potential_difference
                << std::endl
                << "Internal source state: timestep="
                << simulator_access.get_timestep_number()
                << ", committed_moment_error=" << committed_moment_error
                << ", trial_moment_error=" << production_trial_moment_error
                << ", committed_coefficient_error="
                << production_committed_coefficient_error
                << ", trial_coefficient_error="
                << production_trial_coefficient_error
                << ", volume_delta=" << volume_delta
                << ", sheet_delta=" << sheet_delta
                << ", delta_error=" << delta_error
                << ", zero_committed_difference="
                << zero_committed_counterfactual
                << ", reused_trial_difference="
                << reused_trial_counterfactual
                << ", combined_reciprocity_error="
                << combined_source_reciprocity_error
                << ", omitted_volume_difference="
                << omitted_volume_counterfactual
                << ", omitted_sheet_difference="
                << omitted_sheet_counterfactual
                << ", wrong_sheet_sign_difference="
                << wrong_sign_counterfactual
                << std::endl;

      AssertThrow(expected_moments.face_visits > 0,
                  ExcMessage("The displaced-sheet test visited no internal jump face."));
      AssertThrow(dipole_error <= moment_tolerance
                  && inertia_error <= moment_tolerance,
                  ExcMessage("The displaced-sheet mass moments violate the velocity-face projection."));
      AssertThrow(potential_error <= potential_tolerance,
                  ExcMessage("The displaced-sheet potential violates the velocity-face projection."));
      AssertThrow(old_moment_difference > 100.0 * moment_tolerance,
                  ExcMessage("The displaced-sheet moment test does not distinguish the old face quadrature."));
      AssertThrow(old_potential_difference > 100.0 * potential_tolerance,
                  ExcMessage("The displaced-sheet potential test does not distinguish the old face quadrature."));
      AssertThrow(committed_moment_error <= moment_tolerance
                  && production_trial_moment_error <= moment_tolerance
                  && production_committed_moment_error <= moment_tolerance,
                  ExcMessage("Committed/trial production mass moments disagree with independent integration."));
      AssertThrow(production_trial_coefficient_error <= potential_tolerance
                  && production_committed_coefficient_error <= potential_tolerance,
                  ExcMessage("Committed/trial source coefficients disagree with independent integration."));
      AssertThrow(delta_error <= moment_tolerance,
                  ExcMessage("The production trial-minus-committed moment does not equal the independently integrated dt*u_r source."));
      AssertThrow(combined_source_reciprocity_error <= potential_tolerance,
                  ExcMessage("The independently integrated combined source does not equal its separate volume-plus-sheet coefficients."));
      AssertThrow(omitted_volume_counterfactual > 100.0 * potential_tolerance,
                  ExcMessage("The combined source/force regression does not distinguish an omitted volume source."));
      AssertThrow(omitted_sheet_counterfactual > 100.0 * potential_tolerance,
                  ExcMessage("The combined source/force regression does not distinguish an omitted sheet source."));
      AssertThrow(wrong_sign_counterfactual > 100.0 * potential_tolerance,
                  ExcMessage("The combined source/force regression does not distinguish the wrong sheet sign."));
      AssertThrow(zero_committed_counterfactual > 100.0 * moment_tolerance,
                  ExcMessage("The regression does not distinguish zero committed internal sources."));
      if (is_timestep_zero)
        AssertThrow(reused_trial_counterfactual > 100.0 * moment_tolerance
                    && volume_delta > 100.0 * moment_tolerance
                    && sheet_delta > 100.0 * moment_tolerance,
                    ExcMessage("The first timestep-zero trial solve must include its velocity increment before history is committed."));
      else
        {
          AssertThrow(volume_delta > 100.0 * moment_tolerance,
                      ExcMessage("The nonzero-timestep volume-gradient trial increment vanished."));
          AssertThrow(sheet_delta > 100.0 * moment_tolerance,
                      ExcMessage("The nonzero-timestep internal-sheet trial increment vanished."));
          AssertThrow(reused_trial_counterfactual > 100.0 * moment_tolerance,
                      ExcMessage("The regression does not distinguish reusing trial sources as committed."));
        }
    }



    template <int dim>
    void
    replace_internal_density_jump_assembler(const SimulatorAccess<dim> &,
                                            Assemblers::Manager<dim> &assemblers)
    {
      bool replaced = false;
      for (auto &assembler : assemblers.stokes_system)
        if (dynamic_cast<Assemblers::StokesInternalDensityJumpRestoring<dim> *>(
              assembler.get()) != nullptr)
          {
            assembler = std::make_unique<InternalDensityJumpWorkIdentityAssembler<dim>>();
            replaced = true;
          }

      AssertThrow(replaced,
                  ExcMessage("The internal density jump restoring assembler was not registered."));
      std::cout << "Internal density jump work identity checker: registered"
                << std::endl;
    }



    template <int dim>
    void
    report_internal_density_jump_work_identity(const SimulatorAccess<dim> &)
    {
      std::cout << std::scientific << std::setprecision(6)
                << "Internal density jump work identity: faces="
                << InternalDensityJumpWorkIdentityAssembler<dim>::global_face_visits
                << ", q="
                << InternalDensityJumpWorkIdentityAssembler<dim>::global_quadrature_visits
                << ", matrix_error="
                << InternalDensityJumpWorkIdentityAssembler<dim>::global_matrix_error
                << ", rhs_error="
                << InternalDensityJumpWorkIdentityAssembler<dim>::global_rhs_error
                << std::endl;

      AssertThrow(InternalDensityJumpWorkIdentityAssembler<dim>::global_face_visits > 0,
                  ExcMessage("The work-identity test did not visit any internal density jump face."));
    }



    template <int dim>
    void
    signal_connector(SimulatorSignals<dim> &signals)
    {
      signals.set_assemblers.connect(
        &replace_internal_density_jump_assembler<dim>);
      signals.post_stokes_solver.connect(
        [](const SimulatorAccess<dim> &simulator_access,
           const unsigned int,
           const unsigned int,
           const SolverControl &,
           const SolverControl &)
      {
        report_internal_density_jump_work_identity(simulator_access);
        if constexpr (dim == 3)
          validate_internal_sheet_consumers(simulator_access);
      });
    }

    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(
      SelfGravityCacheStateObserver,
      "self gravity cache observer",
      "Test-only observer that validates the committed and trial full-domain "
      "self-gravity cache states against an independent internal-source oracle.")
  }
}

ASPECT_REGISTER_SIGNALS_CONNECTOR(aspect::signal_connector<2>,
                                  aspect::signal_connector<3>)
