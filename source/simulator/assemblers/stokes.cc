/*
  Copyright (C) 2016 - 2024 by the authors of the ASPECT code.

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

#include <aspect/simulator/assemblers/stokes.h>
#include <aspect/boundary_traction/potential_feedback_traction.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/simulator.h>
#include <aspect/utilities.h>

#include <deal.II/base/signaling_nan.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace aspect
{
  namespace Assemblers
  {
    namespace
    {
      template <int dim>
      const PotentialFeedback::SelfGravitation<dim> *
      active_self_gravity(
        const BoundaryTraction::Manager<dim> &traction_manager)
      {
        for (const auto &plugin : traction_manager.get_active_plugins())
          {
            if (const auto *self_gravity =
                  dynamic_cast<const PotentialFeedback::SelfGravitation<dim> *>(
                    plugin.get()))
              return self_gravity;

            if (const auto *potential_feedback =
                  dynamic_cast<const BoundaryTraction::PotentialFeedbackTraction<dim> *>(
                    plugin.get()))
              if (potential_feedback->has_self_gravity_feedback())
                return &potential_feedback->get_self_gravity();
          }

        return nullptr;
      }


      template <int dim>
      const BoundaryTraction::PotentialFeedbackTraction<dim> *
      active_potential_feedback(
        const BoundaryTraction::Manager<dim> &traction_manager)
      {
        for (const auto &plugin : traction_manager.get_active_plugins())
          if (const auto *potential_feedback =
                dynamic_cast<const BoundaryTraction::PotentialFeedbackTraction<dim> *>(
                  plugin.get()))
            return potential_feedback;

        return nullptr;
      }


      bool
      polar_wander_rhs_debug_enabled()
      {
        const char *enabled = std::getenv("ASPECT_PW_RHS_DEBUG");
        return enabled != nullptr && std::atof(enabled) != 0.0;
      }


      template <int dim, typename Accessor>
      void
      write_polar_wander_rhs_diagnostic(const Accessor &accessor,
                                        const std::string &component,
                                        const double cosine,
                                        const double sine)
      {
        if constexpr (dim == 3)
          {
            if (!polar_wander_rhs_debug_enabled())
              return;

            const unsigned int rank =
              Utilities::MPI::this_mpi_process(accessor.get_mpi_communicator());
            std::ostringstream filename;
            filename << accessor.get_parameters().output_directory
                     << "/aspect_pw_rhs_assembler_diagnostic_rank"
                     << std::setw(4) << std::setfill('0') << rank;

            static bool header_written = false;
            std::ofstream output(filename.str(),
                                 header_written ? std::ios::app : std::ios::out);
            if (!output)
              return;

            if (!header_written)
              {
                output
                    << "# ASPECT l2m1 RHS assembler diagnostic\n"
                    << "# local rank contributions; sum over ranks/files in postprocessing\n"
                    << "# columns: timestep time component cosine sine\n";
                header_written = true;
              }

            output << std::setprecision(16) << std::scientific
                   << accessor.get_timestep_number() << ' '
                   << accessor.get_time() << ' '
                   << component << ' '
                   << cosine << ' '
                   << sine << '\n';
          }
        else
          {
            (void) accessor;
            (void) component;
            (void) cosine;
            (void) sine;
          }
      }


      template <int dim>
      std::pair<double,double>
      y21_at_point(const Point<dim> &point)
      {
        if constexpr (dim == 3)
          {
            const std::array<double,dim> spherical_coordinates =
              Utilities::Coordinates::cartesian_to_spherical_coordinates(point);
            return Utilities::real_spherical_harmonic(
                     2, 1, spherical_coordinates[2], spherical_coordinates[1]);
          }
        else
          return {0.0, 0.0};
      }
    }


    template <int dim>
    void
    StokesPreconditioner<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesPreconditioner<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesPreconditioner<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesPreconditioner<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesPreconditioner<dim>&> (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      const std::shared_ptr<const MaterialModel::PrescribedPlasticDilation<dim>>
      prescribed_dilation =
        this->get_parameters().enable_prescribed_dilation
        ? scratch.material_model_outputs.template get_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>()
        : nullptr;

      // First loop over all dofs and find those that are in the Stokes system
      // save the component (pressure and dim velocities) each belongs to.
      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      // When using the Q1-Q1 equal order element, we need to compute the
      // projection of the Q1 pressure shape functions onto the constants
      // and use this projection in the computation of matrix terms.
      // Do this here by computing the integral of the shape functions
      // over the cell and then dividing by the area of the cell.
      std::vector<double> average_pressure_shape_function (this->get_parameters().use_equal_order_interpolation_for_stokes
                                                           ?
                                                           stokes_dofs_per_cell
                                                           :
                                                           0,
                                                           numbers::signaling_nan<double>());
      if (this->get_parameters().use_equal_order_interpolation_for_stokes)
        {
          // Check that we are really only using a Q1-Q1 element and
          // not a Q2-Q2 element. This is because in the latter case, the
          // projection isn't just on the piecewise constants, but onto
          // the piecewise (bi,tri)linears, and this is going to be a bit
          // more involved than just computing a single number per shape
          // function.
          Assert (this->get_parameters().stokes_velocity_degree==1,
                  ExcNotImplemented());

          double area       = 0;
          for (unsigned int q=0; q<n_q_points; ++q)
            area += scratch.finite_element_values.JxW(q);

          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  double int_over_p = 0;

                  for (unsigned int q=0; q<n_q_points; ++q)
                    {
                      int_over_p += scratch.finite_element_values[introspection.extractors.pressure].value(i,q)
                                    *
                                    scratch.finite_element_values.JxW(q);
                    }

                  average_pressure_shape_function[i_stokes] = int_over_p/area;
                  ++i_stokes;
                }
              ++i;
            }
        }

      // Loop over all quadrature points and assemble their contributions to
      // the preconditioner matrix
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  if (this->get_parameters().use_full_A_block_preconditioner == false)
                    scratch.grads_phi_u[i_stokes] =
                      scratch.finite_element_values[introspection.extractors
                                                    .velocities].symmetric_gradient(i, q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection
                                                                          .extractors.pressure].value(i, q);
                  if (this->get_parameters().use_bfbt == true)
                    {
                      scratch.grad_phi_p[i_stokes]=scratch.finite_element_values[introspection.extractors.pressure].gradient(i,q);
                      scratch.phi_u[i_stokes]=scratch.finite_element_values[introspection.extractors.velocities].value(i,q);
                    }
                  ++i_stokes;
                }
              ++i;
            }

          const double eta = scratch.material_model_outputs.viscosities[q];
          const double one_over_eta = 1. / eta;

          const double JxW = scratch.finite_element_values.JxW(q);

          if (this->get_parameters().use_full_A_block_preconditioner == false)
            {
              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                  if (scratch.dof_component_indices[i] ==
                      scratch.dof_component_indices[j])
                    {
                      data.local_matrix(i, j) += ((2.0 * eta * (scratch.grads_phi_u[i]
                                                                * scratch.grads_phi_u[j]))
                                                 )
                                                 * JxW;
                    }


            }
          if (this->get_parameters().use_bfbt == true)
            {
              const double sqrt_eta = std::sqrt(eta);
              const unsigned int pressure_component_index = this->introspection().component_indices.pressure;

              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                    {


                      // i and j are not pressures
                      if (scratch.dof_component_indices[i] != pressure_component_index && scratch.dof_component_indices[j] != pressure_component_index)
                        data.local_inverse_lumped_mass_matrix[i] += sqrt_eta*scalar_product(scratch.phi_u[i],scratch.phi_u[j])*JxW;


                      // i and j are pressures
                      if (scratch.dof_component_indices[i] == pressure_component_index && scratch.dof_component_indices[j] == pressure_component_index)
                        data.local_matrix(i, j) += (
                                                     1.0/sqrt_eta * pressure_scaling
                                                     * pressure_scaling
                                                     * (scratch.grad_phi_p[i]
                                                        * scratch.grad_phi_p[j] + 1e-6*scratch.phi_p[i]*scratch.phi_p[j] ))
                                                   * JxW;
                    }
                }
            }
          else
            {
              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                  if (scratch.dof_component_indices[i] ==
                      scratch.dof_component_indices[j])
                    {
                      data.local_matrix(i, j) += (
                                                   one_over_eta * pressure_scaling
                                                   * pressure_scaling
                                                   * (scratch.phi_p[i]
                                                      * scratch.phi_p[j]))
                                                 * JxW;
                    }
            }

          // If we are using the equal order Q1-Q1 element, then we also need
          // to add the stabilization term to the (P,P) block of the matrix.
          // Note the change in sign from the one in the assembly of the
          // system matrix, which is due to the fact that the Schur complement
          // of the matrix
          //    [ A   B ]
          //    [ B^T C ]
          // is actually
          //    S  =  B^T A^{-1} B - C
          // with the minus sign in front of C. Because C is defined in the
          // method of Dohrmann and Bochev as a *negative* definite operator,
          // we here need to add the *positive* operator.
          if (this->get_parameters().use_equal_order_interpolation_for_stokes)
            {
              for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i,j) += ( one_over_eta * pressure_scaling * pressure_scaling *
                                                (scratch.phi_p[i] - average_pressure_shape_function[i]) *
                                                (scratch.phi_p[j] - average_pressure_shape_function[j]))
                                              * JxW;
                  }
            }

          if (prescribed_dilation != nullptr)
            {
              for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  data.local_matrix(i,j) += prescribed_dilation->dilation_lhs_term[q]
                                            * pressure_scaling * pressure_scaling
                                            * scratch.phi_p[i] * scratch.phi_p[j]
                                            * JxW;
            }
        }
    }



    template <int dim>
    void
    StokesPreconditioner<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.n_evaluation_points();

      if (this->get_parameters().enable_prescribed_dilation
          && outputs.template has_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>() == false)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedPlasticDilation<dim>>(n_points));
        }

      Assert(!this->get_parameters().enable_prescribed_dilation
             ||
             outputs.template get_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>()->dilation_lhs_term.size()
             == n_points, ExcInternalError());

    }



    template <int dim>
    void
    StokesCompressiblePreconditioner<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      Assert (this->get_parameters().use_equal_order_interpolation_for_stokes == false,
              ExcNotImplemented());

      Assert (this->get_parameters().use_full_A_block_preconditioner == false,
              ExcMessage("This assembler should only be called if the simplified A block "
                         "preconditioner is used."));

      internal::Assembly::Scratch::StokesPreconditioner<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesPreconditioner<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesPreconditioner<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesPreconditioner<dim>&> (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;

      // First loop over all dofs and find those that are in the Stokes system
      // save the component (pressure and dim velocities) each belongs to.
      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      // Loop over all quadrature points and assemble their contributions to
      // the preconditioner matrix
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                  scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);

                  ++i_stokes;
                }
              ++i;
            }

          const double eta_two_thirds = scratch.material_model_outputs.viscosities[q] * 2.0 / 3.0;

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
              if (scratch.dof_component_indices[i] ==
                  scratch.dof_component_indices[j])
                data.local_matrix(i, j) += (- eta_two_thirds * (scratch.div_phi_u[i] * scratch.div_phi_u[j])
                                           )
                                           * JxW;
        }
    }



    template <int dim>
    void
    StokesIncompressibleTerms<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      const std::shared_ptr<const MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> force
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>();

      const std::shared_ptr<const MaterialModel::ElasticOutputs<dim>> elastic_outputs
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::ElasticOutputs<dim>>();

      const std::shared_ptr<const MaterialModel::PrescribedPlasticDilation<dim>> prescribed_dilation
        = (this->get_parameters().enable_prescribed_dilation)
          ? scratch.material_model_outputs.template get_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>()
          : nullptr;

      const bool use_mechanical_mass_conservation =
        this->get_parameters().density_source_law
        == Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation;
      const PotentialFeedback::SelfGravitation<dim> *self_gravity =
        (use_mechanical_mass_conservation
         ? active_self_gravity(this->get_boundary_traction_manager())
         : nullptr);
      const BoundaryTraction::PotentialFeedbackTraction<dim> *potential_feedback =
        (use_mechanical_mass_conservation
         ? active_potential_feedback(this->get_boundary_traction_manager())
         : nullptr);
      const unsigned int radial_displacement_history_index =
        use_mechanical_mass_conservation
        ? introspection.compositional_index_for_name("ve_radial_displacement")
        : numbers::invalid_unsigned_int;

      // When using the Q1-Q1 equal order element, we need to compute the
      // projection of the Q1 pressure shape functions onto the constants
      // and use this projection in the computation of matrix terms.
      // Do this here by computing the integral of the shape functions
      // over the cell and then dividing by the area of the cell.
      std::vector<double> average_pressure_shape_function (this->get_parameters().use_equal_order_interpolation_for_stokes
                                                           ?
                                                           stokes_dofs_per_cell
                                                           :
                                                           0,
                                                           numbers::signaling_nan<double>());
      if (this->get_parameters().use_equal_order_interpolation_for_stokes)
        {
          // Check that we are really only using a Q1-Q1 element and
          // not a Q2-Q2 element. This is because in the latter case, the
          // projection isn't just on the piecewise constants, but onto
          // the piecewise (bi,tri)linears, and this is going to be a bit
          // more involved than just computing a single number per shape
          // function.
          Assert (this->get_parameters().stokes_velocity_degree==1,
                  ExcNotImplemented());

          double area       = 0;
          for (unsigned int q=0; q<n_q_points; ++q)
            area += scratch.finite_element_values.JxW(q);

          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  double int_over_p = 0;

                  for (unsigned int q=0; q<n_q_points; ++q)
                    {
                      int_over_p += scratch.finite_element_values[introspection.extractors.pressure].value(i,q)
                                    *
                                    scratch.finite_element_values.JxW(q);
                    }

                  average_pressure_shape_function[i_stokes] = int_over_p/area;
                  ++i_stokes;
                }
              ++i;
            }
        }

      double pw_volume_mass_conservation_rhs_cosine = 0.0;
      double pw_volume_mass_conservation_rhs_sine = 0.0;

      // Next, do the integration of matrix and right hand side terms.
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  if (scratch.rebuild_stokes_matrix)
                    {
                      scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                      scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);
                    }
                  else if (this->get_parameters().enable_elasticity)
                    {
                      scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                      if (use_mechanical_mass_conservation)
                        scratch.div_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].divergence(i,q);
                    }
                  ++i_stokes;
                }
              ++i;
            }


          // Viscosity scalar
          const double eta = ((scratch.rebuild_stokes_matrix || prescribed_dilation)
                              ?
                              scratch.material_model_outputs.viscosities[q]
                              :
                              numbers::signaling_nan<double>());
          const double one_over_eta = (scratch.rebuild_stokes_matrix
                                       &&
                                       this->get_parameters().use_equal_order_interpolation_for_stokes
                                       ?
                                       1./eta
                                       :
                                       numbers::signaling_nan<double>());

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double density =
            this->get_density_source_manager().stokes_source_density(
              scratch.material_model_inputs,
              scratch.material_model_outputs,
              q);
          const double JxW = scratch.finite_element_values.JxW(q);

          double bulk_modulus = numbers::signaling_nan<double>();
          double reference_density = numbers::signaling_nan<double>();
          double current_radial_restoring_coefficient = 0.0;
          double history_radial_restoring_coefficient = 0.0;
          double old_radial_displacement = 0.0;
          Tensor<1,dim> radial_unit;
          if (use_mechanical_mass_conservation)
            {
              AssertThrow(elastic_outputs != nullptr, ExcInternalError());
              bulk_modulus =
                this->get_density_source_manager().elastic_bulk_modulus(
                  scratch.material_model_outputs,
                  q);
              const double mechanical_time_step =
                this->get_density_source_manager().effective_mechanical_time_step();
              const Point<dim> position =
                scratch.finite_element_values.quadrature_point(q);
              const double radius = position.norm();
              AssertThrow(radius > 0.0,
                          ExcMessage("Mechanical mass conservation is undefined at radius zero."));
              radial_unit = position / radius;

              reference_density =
                this->get_density_source_manager().reference_density(position);
              const double gravity_magnitude =
                this->get_density_source_manager().mechanical_gravity_magnitude(
                  position,
                  gravity.norm());
              current_radial_restoring_coefficient =
                reference_density * gravity_magnitude * mechanical_time_step;
              history_radial_restoring_coefficient =
                reference_density * gravity_magnitude;
              old_radial_displacement =
                scratch.material_model_inputs.composition[q][radial_displacement_history_index];
            }

          const double full_domain_potential =
            (potential_feedback != nullptr
             ? potential_feedback->full_domain_potential(
               scratch.finite_element_values.quadrature_point(q))
             : (self_gravity != nullptr
                && self_gravity->has_full_domain_potential()
                ? self_gravity->full_domain_potential(
                  scratch.finite_element_values.quadrature_point(q))
                : 0.0));

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            {
              data.local_rhs(i) += (density * gravity * scratch.phi_u[i])
                                   * JxW;

              if (force != nullptr && this->get_parameters().enable_additional_stokes_rhs)
                data.local_rhs(i) += (force->rhs_u[q] * scratch.phi_u[i]
                                      + pressure_scaling * force->rhs_p[q] * scratch.phi_p[i])
                                     * JxW;

              if (elastic_outputs != nullptr && this->get_parameters().enable_elasticity)
                data.local_rhs(i) += (elastic_outputs->elastic_force[q] * scratch.grads_phi_u[i])
                                     * JxW;

              if (prescribed_dilation != nullptr)
                data.local_rhs(i) += (
                                       - pressure_scaling
                                       * prescribed_dilation->dilation_rhs_term[q]
                                       * scratch.phi_p[i]
                                     ) * JxW;

              if (use_mechanical_mass_conservation)
                {
                  data.local_rhs(i) +=
                    history_radial_restoring_coefficient
                    * scratch.div_phi_u[i]
                    * old_radial_displacement
                    * JxW;

                  if (full_domain_potential != 0.0)
                    {
                      data.local_rhs(i) -=
                        reference_density
                        * full_domain_potential
                        * scratch.div_phi_u[i]
                        * JxW;
                    }
                }

              if (scratch.rebuild_stokes_matrix)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i,j) += ( (eta * 2.0 * (scratch.grads_phi_u[i] * scratch.grads_phi_u[j]))
                                                // assemble \nabla p as -(p, div v):
                                                - (pressure_scaling *
                                                   scratch.div_phi_u[i] * scratch.phi_p[j])
                                                // assemble the term -div(u) as -(div u, q).
                                                // Note the negative sign to make this
                                                // operator adjoint to the grad p term:
                                                - (pressure_scaling *
                                                   scratch.phi_p[i] * scratch.div_phi_u[j])
                                                // assemble -\bar\alpha\alpha pq / eta^{ve}
                                                // if plastic dilation is enabled
                                                - (prescribed_dilation == nullptr ? 0.0 :
                                                   pressure_scaling * pressure_scaling *
                                                   prescribed_dilation->dilation_lhs_term[q] *
                                                   scratch.phi_p[i] * scratch.phi_p[j])
                                                + (use_mechanical_mass_conservation
                                                   ? pressure_scaling
                                                   * history_radial_restoring_coefficient
                                                   / bulk_modulus
                                                   * (scratch.phi_u[i] * radial_unit)
                                                   * scratch.phi_p[j]
                                                   : 0.0)
                                                - (use_mechanical_mass_conservation
                                                   ? current_radial_restoring_coefficient
                                                   * scratch.div_phi_u[i]
                                                   * (scratch.phi_u[j] * radial_unit)
                                                   : 0.0)
                                              )
                                              * JxW;
                  }
            }

          // If we are using the equal order Q1-Q1 element, then we also need
          // to put the stabilization term into the (P,P) block of the matrix:
          if (scratch.rebuild_stokes_matrix
              &&
              this->get_parameters().use_equal_order_interpolation_for_stokes)
            {
              for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i,j) += ( - (one_over_eta * pressure_scaling * pressure_scaling *
                                                   (scratch.phi_p[i] - average_pressure_shape_function[i]) *
                                                   (scratch.phi_p[j] - average_pressure_shape_function[j])))
                                              * JxW;
                  }
            }

          if (use_mechanical_mass_conservation
              && full_domain_potential != 0.0
              && polar_wander_rhs_debug_enabled())
            {
              const Point<dim> position =
                scratch.finite_element_values.quadrature_point(q);
              const double radius = position.norm();
              if (radius > 0.0)
                {
                  const std::pair<double,double> y21 =
                    y21_at_point<dim>(position);
                  const double radial_test_divergence = 2.0 / radius;
                  const double contribution =
                    -reference_density
                    * full_domain_potential
                    * radial_test_divergence
                    * JxW;
                  pw_volume_mass_conservation_rhs_cosine +=
                    contribution * y21.first;
                  pw_volume_mass_conservation_rhs_sine +=
                    contribution * y21.second;
                }
            }
        }

      if (pw_volume_mass_conservation_rhs_cosine != 0.0
          || pw_volume_mass_conservation_rhs_sine != 0.0)
        write_polar_wander_rhs_diagnostic<dim>(
          *this,
          "volume_mechanical_mass_conservation_full_potential",
          pw_volume_mass_conservation_rhs_cosine,
          pw_volume_mass_conservation_rhs_sine);
    }



    template <int dim>
    void
    StokesIncompressibleTerms<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.n_evaluation_points();

      // Stokes RHS:
      if (this->get_parameters().enable_additional_stokes_rhs
          && outputs.template has_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>() == false)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> (n_points));
        }

      Assert(!this->get_parameters().enable_additional_stokes_rhs
             ||
             outputs.template get_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>()->rhs_u.size()
             == n_points, ExcInternalError());

      // prescribed dilation:
      if (this->get_parameters().enable_prescribed_dilation
          && outputs.template has_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>() == false)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedPlasticDilation<dim>> (n_points));
        }

      Assert(!this->get_parameters().enable_prescribed_dilation
             ||
             (outputs.template get_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>()->dilation_lhs_term.size() == n_points &&
              outputs.template get_additional_output_object<MaterialModel::PrescribedPlasticDilation<dim>>()->dilation_rhs_term.size() == n_points),
             ExcInternalError());

      // Elasticity:
      if ((this->get_parameters().enable_elasticity) &&
          outputs.template has_additional_output_object<MaterialModel::ElasticOutputs<dim>>() == false)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::ElasticOutputs<dim>> (n_points));
        }

      Assert(!this->get_parameters().enable_elasticity
             ||
             outputs.template get_additional_output_object<MaterialModel::ElasticOutputs<dim>>()->elastic_force.size()
             == n_points, ExcInternalError());

      this->get_density_source_manager().create_additional_material_model_outputs(outputs);
    }



    template <int dim>
    void
    StokesCompressibleStrainRateViscosityTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      if (!scratch.rebuild_stokes_matrix)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);

                  ++i_stokes;
                }
              ++i;
            }

          // Viscosity scalar
          const double eta_two_thirds = scratch.material_model_outputs.viscosities[q] * 2.0 / 3.0;

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
              {
                data.local_matrix(i,j) += (- (eta_two_thirds * (scratch.div_phi_u[i] * scratch.div_phi_u[j])
                                             ))
                                          * JxW;
              }
        }
    }



    template <int dim>
    void
    StokesReferenceDensityCompressibilityTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      // assemble RHS of:
      //  - div u = 1/rho * drho/dz g/||g||* u
      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::reference_density_profile,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
          const double drho_dz_u = scratch.reference_densities_depth_derivative[q]
                                   * (gravity * scratch.velocity_values[q]) / gravity.norm();
          const double one_over_rho = 1.0/scratch.reference_densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (pressure_scaling *
                                  one_over_rho * drho_dz_u * scratch.phi_p[i])
                                 * JxW;
        }
    }



    template <int dim>
    void
    StokesImplicitReferenceDensityCompressibilityTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      // assemble compressibility term of:
      //  - div u - 1/rho * drho/dz g/||g||* u = 0
      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::implicit_reference_density_profile,
             ExcInternalError());

      if (!scratch.rebuild_stokes_matrix)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i,q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
          const Tensor<1,dim> drho_dz = scratch.reference_densities_depth_derivative[q]
                                        * gravity / gravity.norm();
          const double one_over_rho = 1.0/scratch.reference_densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
              data.local_matrix(i,j) += -(pressure_scaling *
                                          one_over_rho * drho_dz * scratch.phi_u[j] * scratch.phi_p[i])
                                        * JxW;
        }
    }



    template <int dim>
    void
    StokesIsentropicCompressionTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      // assemble RHS of:
      //  - div \mathbf{u} = \frac{1}{\rho} \frac{\partial rho}{\partial p} \rho \mathbf{g} \cdot \mathbf{u}

      // Compared to the manual, this term seems to have the wrong sign, but
      // this is because we negate the entire equation to make sure we get
      // -div(u) as the adjoint operator of grad(p)

      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::isentropic_compression,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double compressibility
            = scratch.material_model_outputs.compressibilities[q];

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (
                                   (pressure_scaling *
                                    compressibility * density *
                                    (scratch.velocity_values[q] * gravity) *
                                    scratch.phi_p[i])
                                 )
                                 * JxW;
        }
    }



    template <int dim>
    void
    StokesElasticPressureEvolutionTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch =
        dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data =
        dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::elastic_pressure_evolution,
             ExcInternalError());

      const std::shared_ptr<const MaterialModel::ElasticOutputs<dim>> elastic_outputs =
        scratch.material_model_outputs.template get_additional_output_object<MaterialModel::ElasticOutputs<dim>>();
      AssertThrow(elastic_outputs != nullptr,
                  ExcMessage("Elastic pressure evolution requires elastic bulk-modulus material outputs."));

      double effective_time_step = this->get_timestep();
      if (this->get_timestep_number() == 0)
        effective_time_step = this->get_material_model().initial_elastic_time_step();
      AssertThrow(effective_time_step > 0.0,
                  ExcMessage("Elastic pressure evolution requires a positive effective time step."));

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      std::vector<double> old_pressure(n_q_points, 0.0);
      if (this->get_timestep_number() > 0)
        scratch.finite_element_values[introspection.extractors.pressure].get_function_values(
          this->get_old_solution(), old_pressure);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0;
               i_stokes < stokes_dofs_per_cell;
               ++i)
            if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
              {
                scratch.phi_p[i_stokes] =
                  scratch.finite_element_values[introspection.extractors.pressure].value(i, q);
                ++i_stokes;
              }

          const double bulk_modulus = elastic_outputs->elastic_bulk_moduli[q];
          AssertThrow(std::isfinite(bulk_modulus) && bulk_modulus > 0.0,
                      ExcMessage("Elastic pressure evolution requires a finite positive elastic bulk modulus."));

          const double inverse_bulk_time = 1.0 / (bulk_modulus * effective_time_step);
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            {
              data.local_rhs(i) -= pressure_scaling * inverse_bulk_time
                                   * old_pressure[q] * scratch.phi_p[i] * JxW;

              if (scratch.rebuild_stokes_matrix)
                for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                  data.local_matrix(i,j) -= pressure_scaling * pressure_scaling
                                            * inverse_bulk_time
                                            * scratch.phi_p[i] * scratch.phi_p[j] * JxW;
            }
        }
    }



    template <int dim>
    void
    StokesProjectedDensityFieldTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      // assemble RHS of:
      // $ - \nabla \cdot \mathbf{u} = \frac{1}{\rho} \frac{\partial \rho}{\partial t} + \frac{1}{\rho} \nabla \rho \cdot \mathbf{u}$

      // Compared to the manual, this term seems to have the wrong sign, but
      // this is because we negate the entire equation to make sure we get
      // -div(u) as the adjoint operator of grad(p)

      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::projected_density_field,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();
      const unsigned int density_idx = this->introspection().find_composition_type(CompositionalFieldDescription::density);

      const double time_step = this->get_timestep();
      const double old_time_step = this->get_old_timestep();

      std::vector<Tensor<1,dim>> density_gradients(n_q_points);
      std::vector<double> density(n_q_points);
      std::vector<double> density_old(n_q_points);
      std::vector<double> density_old_old(n_q_points);

      scratch.finite_element_values[introspection.extractors.compositional_fields[density_idx]].get_function_gradients (this->get_current_linearization_point(),
          density_gradients);
      scratch.finite_element_values[introspection.extractors.compositional_fields[density_idx]].get_function_values (this->get_current_linearization_point(),
          density);
      scratch.finite_element_values[introspection.extractors.compositional_fields[density_idx]].get_function_values (this->get_old_solution(),
          density_old);
      scratch.finite_element_values[introspection.extractors.compositional_fields[density_idx]].get_function_values (this->get_old_old_solution(),
          density_old_old);

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          double drho_dt;

          if (this->get_timestep_number() > 1)
            drho_dt = (1.0/time_step) *
                      (density[q] *
                       (2*time_step + old_time_step) / (time_step + old_time_step)
                       -
                       density_old[q] *
                       (1 + time_step/old_time_step)
                       +
                       density_old_old[q] *
                       (time_step * time_step) / (old_time_step * (time_step + old_time_step)));
          else if (this->get_timestep_number() == 1)
            drho_dt =
              (density[q] - density_old[q]) / time_step;
          else
            drho_dt = 0.0;

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (
                                   (pressure_scaling *
                                    (1.0 / density[q]) *
                                    (density_gradients[q] *
                                     scratch.velocity_values[q]) *
                                    scratch.phi_p[i])
                                   + pressure_scaling * (1.0 / density[q]) * drho_dt * scratch.phi_p[i]
                                 )
                                 * JxW;
        }
    }



    template <int dim>
    void
    StokesHydrostaticCompressionTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      // assemble RHS of:
      // $ -\nabla \cdot \mathbf{u} = \left( \kappa \rho \textbf{g} - \alpha \nabla T \right) \cdot \textbf{u}$
      //
      // where $\frac{1}{\rho} \frac{\partial \rho}{\partial p} = \kappa$ is the compressibility,
      // $- \frac{1}{\rho}\frac{\partial \rho}{\partial T} = \alpha$ is the thermal expansion coefficient,
      // and both are defined in the material model.

      // Compared to the manual, this term seems to have the wrong sign, but
      // this is because we negate the entire equation to make sure we get
      // -div(u) as the adjoint operator of grad(p)

      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::hydrostatic_compression,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double compressibility
            = scratch.material_model_outputs.compressibilities[q];

          const double thermal_alpha
            = scratch.material_model_outputs.thermal_expansion_coefficients[q];

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (
                                   (pressure_scaling *
                                    (
                                      // pressure part:
                                      compressibility * density *
                                      (scratch.velocity_values[q] * gravity)
                                      // temperature part:
                                      - thermal_alpha *
                                      (scratch.velocity_values[q] * scratch.temperature_gradients[q])
                                    ) * scratch.phi_p[i])
                                 )
                                 * JxW;

        }
    }


    template <int dim>
    void
    StokesPressureRHSCompatibilityModification<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                                                              internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          const double JxW = scratch.finite_element_values.JxW(q);
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  data.local_pressure_shape_function_integrals(i_stokes) += scratch.phi_p[i_stokes] * JxW;
                  ++i_stokes;
                }
              ++i;
            }
        }
    }



    template <int dim>
    void
    StokesBoundaryTraction<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                                          internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      // see if any of the faces are traction boundaries for which
      // we need to assemble force terms for the right hand side
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

      const typename DoFHandler<dim>::face_iterator face = scratch.cell->face(scratch.face_number);

      const auto &traction_bis = this->get_boundary_traction_manager().get_prescribed_boundary_traction_indicators();
      const BoundaryTraction::PotentialFeedbackTraction<dim> *potential_feedback =
        active_potential_feedback(this->get_boundary_traction_manager());

      if (traction_bis.find(face->boundary_id()) != traction_bis.end())
        {
          double pw_boundary_feedback_rhs_cosine = 0.0;
          double pw_boundary_feedback_rhs_sine = 0.0;
          for (unsigned int q=0; q<scratch.face_finite_element_values.n_quadrature_points; ++q)
            {
              const Tensor<1,dim> traction
                = this->get_boundary_traction_manager().
                  boundary_traction (face->boundary_id(),
                                     scratch.face_finite_element_values.quadrature_point(q),
                                     scratch.face_finite_element_values.normal_vector(q));

              const double JxW = scratch.face_finite_element_values.JxW(q);

              for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                {
                  if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                    {
                      data.local_rhs(i_stokes) += scratch.face_finite_element_values[introspection.extractors.velocities].value(i,q) *
                                                  traction * JxW;
                      ++i_stokes;
                    }
                  ++i;
                }

              if (potential_feedback != nullptr
                  && polar_wander_rhs_debug_enabled())
                {
                  const Point<dim> position =
                    scratch.face_finite_element_values.quadrature_point(q);
                  const Tensor<1,dim> radial_unit = position / position.norm();
                  const std::pair<double,double> y21 =
                    y21_at_point<dim>(position);
                  const Tensor<1,dim> feedback_traction =
                    potential_feedback->boundary_traction(
                      face->boundary_id(),
                      position,
                      scratch.face_finite_element_values.normal_vector(q));
                  const double contribution =
                    (feedback_traction * radial_unit) * JxW;
                  pw_boundary_feedback_rhs_cosine += contribution * y21.first;
                  pw_boundary_feedback_rhs_sine += contribution * y21.second;
                }
            }

          if (pw_boundary_feedback_rhs_cosine != 0.0
              || pw_boundary_feedback_rhs_sine != 0.0)
            {
              const std::string component =
                "boundary_feedback_" + std::to_string(face->boundary_id());
              write_polar_wander_rhs_diagnostic<dim>(
                *this,
                component,
                pw_boundary_feedback_rhs_cosine,
                pw_boundary_feedback_rhs_sine);
            }
        }
    }

    template <int dim>
    void
    StokesCitcomStyleCMBRadialRestoring<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                                                       internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch =
        dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data =
        dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      if (!scratch.rebuild_stokes_matrix)
        return;

      const Parameters<dim> &parameters = this->get_parameters();

      const typename DoFHandler<dim>::face_iterator face =
        scratch.cell->face(scratch.face_number);

      if (face->boundary_id() != parameters.citcom_style_cmb_radial_restoring_boundary_indicator)
        return;

      double effective_time_step = this->get_timestep();

      if (this->get_timestep_number() == 0 &&
          effective_time_step == 0.0)
        {
          effective_time_step =
            parameters.initial_elastic_response_time_step;

          if (this->get_material_model()
              .use_instantaneous_elastic_response_at_timestep_zero()
              && this->get_material_model().fixed_elastic_time_step() > 0.0)
            effective_time_step =
              this->get_material_model().fixed_elastic_time_step();
        }

      if (effective_time_step == 0.0)
        return;

      const double density_contrast =
        parameters.citcom_style_cmb_radial_restoring_density_contrast;
      const double scale =
        parameters.citcom_style_cmb_radial_restoring_scale;

      if (density_contrast == 0.0 || scale == 0.0)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

      for (unsigned int q=0; q<scratch.face_finite_element_values.n_quadrature_points; ++q)
        {
          const Point<dim> point =
            scratch.face_finite_element_values.quadrature_point(q);

          Tensor<1,dim> radial_unit;
          for (unsigned int d=0; d<dim; ++d)
            radial_unit[d] = point[d];

          const double radius = radial_unit.norm();

          AssertThrow(radius > 0.0,
                      ExcMessage("Cannot construct a radial unit vector at radius zero."));

          radial_unit /= radius;

          const Tensor<1,dim> gravity =
            this->get_gravity_model().gravity_vector(point);

          const double g_magnitude = gravity.norm();

          const double coefficient =
            scale * density_contrast * g_magnitude * effective_time_step;

          const double JxW = scratch.face_finite_element_values.JxW(q);

          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /* increment below */)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  const Tensor<1,dim> phi_i =
                    scratch.face_finite_element_values[introspection.extractors.velocities].value(i,q);
                  const double phi_i_radial = phi_i * radial_unit;

                  for (unsigned int j=0, j_stokes=0; j_stokes<stokes_dofs_per_cell; /* increment below */)
                    {
                      if (introspection.is_stokes_component(fe.system_to_component_index(j).first))
                        {
                          const Tensor<1,dim> phi_j =
                            scratch.face_finite_element_values[introspection.extractors.velocities].value(j,q);
                          const double phi_j_radial = phi_j * radial_unit;

                          data.local_matrix(i_stokes,j_stokes) +=
                            coefficient * phi_i_radial * phi_j_radial * JxW;

                          ++j_stokes;
                        }
                      ++j;
                    }

                  ++i_stokes;
                }
              ++i;
            }
        }
    }



    template <int dim>
    void
    StokesInternalDensityJumpRestoring<dim>::execute (
      internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
      internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch =
        dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data =
        dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&> (data_base);

      const auto &density_sources = this->get_density_source_manager();
      if (!density_sources.has_internal_density_jumps())
        return;
      const PotentialFeedback::SelfGravitation<dim> *self_gravity =
        active_self_gravity(this->get_boundary_traction_manager());
      const BoundaryTraction::PotentialFeedbackTraction<dim> *potential_feedback =
        active_potential_feedback(this->get_boundary_traction_manager());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int displacement_index =
        introspection.compositional_index_for_name("ve_radial_displacement");
      const double mechanical_time_step =
        density_sources.effective_mechanical_time_step();

      std::vector<double> committed_displacements(
        scratch.face_finite_element_values.n_quadrature_points);

      double pw_internal_density_jump_rhs_cosine = 0.0;
      double pw_internal_density_jump_rhs_sine = 0.0;

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
            all_vertices_match =
              all_vertices_match
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

          for (unsigned int q = 0;
               q < scratch.face_finite_element_values.n_quadrature_points;
               ++q)
            {
              const Point<dim> point =
                scratch.face_finite_element_values.quadrature_point(q);
              const double radius = point.norm();
              AssertThrow(radius > 0.0,
                          ExcMessage("Internal density jump restoring is undefined at radius zero."));
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

              if (potential != 0.0 && polar_wander_rhs_debug_enabled())
                {
                  const std::pair<double,double> y21 =
                    y21_at_point<dim>(point);
                  const double contribution =
                    density_contrast * potential * JxW;
                  pw_internal_density_jump_rhs_cosine +=
                    contribution * y21.first;
                  pw_internal_density_jump_rhs_sine +=
                    contribution * y21.second;
                }

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

                    data.local_rhs(i_stokes) -=
                      density_contrast * gravity_magnitude
                      * phi_i_radial * committed_displacements[q] * JxW;
                    data.local_rhs(i_stokes) +=
                      density_contrast * potential
                      * phi_i_radial * JxW;

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
                            data.local_matrix(i_stokes, j_stokes) +=
                              density_contrast * gravity_magnitude
                              * mechanical_time_step * phi_i_radial
                              * (phi_j * radial_unit) * JxW;
                            ++j_stokes;
                          }

                    ++i_stokes;
                  }
            }
        }

      if (pw_internal_density_jump_rhs_cosine != 0.0
          || pw_internal_density_jump_rhs_sine != 0.0)
        write_polar_wander_rhs_diagnostic<dim>(
          *this,
          "internal_density_jump_full_potential",
          pw_internal_density_jump_rhs_cosine,
          pw_internal_density_jump_rhs_sine);
    }


  }
} // namespace aspect

// explicit instantiation of the functions we implement in this file
namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
  template class StokesPreconditioner<dim>; \
  template class StokesCompressiblePreconditioner<dim>; \
  template class StokesIncompressibleTerms<dim>; \
  template class StokesCompressibleStrainRateViscosityTerm<dim>; \
  template class StokesReferenceDensityCompressibilityTerm<dim>; \
  template class StokesImplicitReferenceDensityCompressibilityTerm<dim>; \
  template class StokesIsentropicCompressionTerm<dim>; \
  template class StokesElasticPressureEvolutionTerm<dim>; \
  template class StokesHydrostaticCompressionTerm<dim>; \
  template class StokesProjectedDensityFieldTerm<dim>; \
  template class StokesPressureRHSCompatibilityModification<dim>; \
  template class StokesBoundaryTraction<dim>; \
  template class StokesCitcomStyleCMBRadialRestoring<dim>; \
  template class StokesInternalDensityJumpRestoring<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
