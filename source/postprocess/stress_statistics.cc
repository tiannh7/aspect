/*
  Copyright (C) 2011 - 2022 by the authors of the ASPECT code.

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


#include <aspect/postprocess/stress_statistics.h>
#include <aspect/material_model/rheology/elasticity.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/fe/fe_values.h>

#include <iomanip>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    StressStatistics<dim>::execute (TableHandler &statistics)
    {
      const bool is_spherical_like = (this->get_geometry_model().natural_coordinate_system() != Utilities::Coordinates::cartesian);

      const std::vector<std::string> stress_component_names_2d = is_spherical_like ?
                                                                 std::vector<std::string> {"rr", "rp", "pp"} :
                                                                 std::vector<std::string> {"xx", "xy", "yy"};

      const std::vector<std::string> stress_component_names_3d = is_spherical_like ?
                                                                 std::vector<std::string> {"rr", "rt", "rp", "tt", "tp", "pp"} :
                                                                 std::vector<std::string> {"xx", "xy", "xz", "yy", "yz", "zz"};

      // Use a Gauss-Lobatto quadrature formula based on the velocity
      // degree for computing the min/max, both of which may lie on the
      // boundaries of the cell.
      const QGaussLobatto<dim> quadrature_formula(this->get_fe().base_element(this->introspection().base_elements.velocities).degree + 2);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_gradients |
                               update_quadrature_points |
                               update_JxW_values);

      std::vector<double> pressure_values(n_q_points);
      std::vector<double> temperature_values(n_q_points);
      std::vector<std::vector<double>> composition_values(n_q_points, std::vector<double>(this->n_compositional_fields()));

      // Initialize local variables for stress and shear stress statistics
      std::vector<double> local_stress_integral(SymmetricTensor<2,dim>::n_independent_components, 0.0);
      std::vector<double> local_min_stress(SymmetricTensor<2,dim>::n_independent_components, std::numeric_limits<double>::max());
      std::vector<double> local_max_stress(SymmetricTensor<2,dim>::n_independent_components, std::numeric_limits<double>::lowest());

      std::vector<double> local_shear_stress_integral(SymmetricTensor<2,dim>::n_independent_components, 0.0);
      std::vector<double> local_min_shear_stress(SymmetricTensor<2,dim>::n_independent_components, std::numeric_limits<double>::max());
      std::vector<double> local_max_shear_stress(SymmetricTensor<2,dim>::n_independent_components, std::numeric_limits<double>::lowest());

      // Material model inputs and outputs
      MaterialModel::MaterialModelInputs<dim> in(n_q_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(n_q_points, this->n_compositional_fields());

      in.requested_properties = MaterialModel::MaterialProperties::viscosity | MaterialModel::MaterialProperties::additional_outputs;

      this->get_material_model().create_additional_named_outputs(out);

      // compute the integral quantities by quadrature
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.pressure].get_function_values (this->get_solution(),
                                                                                      pressure_values);
            fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
                                                                                         temperature_values);
            for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
              fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                  composition_values[c]);

            // Set up material model inputs
            in.position = fe_values.get_quadrature_points();
            in.temperature = temperature_values;
            in.pressure = pressure_values;
            in.velocity = std::vector<Tensor<1,dim>>(n_q_points);
            in.composition = composition_values;
            in.strain_rate = std::vector<SymmetricTensor<2,dim>>(n_q_points);

            // Get velocity gradients for strain rate
            std::vector<Tensor<2,dim>> velocity_gradients(n_q_points);
            fe_values[this->introspection().extractors.velocities].get_function_gradients (this->get_solution(),
                                                                                           velocity_gradients);
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                in.strain_rate[q] = symmetrize(velocity_gradients[q]);
              }

            // Evaluate material model
            this->get_material_model().evaluate(in, out);

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                // Calculate stress tensor
                SymmetricTensor<2,dim> stress = in.pressure[q] * unit_symmetric_tensor<dim>();
                const double eta = out.viscosities[q];
                const SymmetricTensor<2, dim> strain_rate = in.strain_rate[q];
                const SymmetricTensor<2, dim> deviatoric_strain_rate = (this->get_material_model().is_compressible()
                                                                        ? strain_rate - 1. / 3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                                                                        : strain_rate);

                SymmetricTensor<2,dim> deviatoric_stress;
                if (this->get_parameters().enable_elasticity)
                  {
                    const std::shared_ptr<const MaterialModel::ElasticAdditionalOutputs<dim>> elastic_additional_out
                      = out.template get_additional_output_object<MaterialModel::ElasticAdditionalOutputs<dim>>();
                    Assert(elastic_additional_out != nullptr, ExcMessage("Elastic Additional Outputs are needed for stress statistics."));
                    deviatoric_stress = elastic_additional_out->deviatoric_stress[q];
                  }
                else
                  {
                    deviatoric_stress = 2. * eta * deviatoric_strain_rate;
                  }

                stress -= deviatoric_stress; // Note: sign convention for compressive stress

                // Calculate shear stress (deviatoric stress)
                SymmetricTensor<2,dim> shear_stress = -deviatoric_stress;

                if (is_spherical_like)
                  {
                    stress = Utilities::Coordinates::cartesian_to_spherical_tensor(stress, in.position[q]);
                    shear_stress = - Utilities::Coordinates::cartesian_to_spherical_tensor(deviatoric_stress, in.position[q]);
                  }

                // Update integrals and min/max for each component
                for (unsigned int i=0; i<SymmetricTensor<2,dim>::n_independent_components; ++i)
                  {
                    const TableIndices<2> indices = SymmetricTensor<2,dim>::unrolled_to_component_indices(i);
                    const double stress_value = stress[indices[0]][indices[1]];
                    const double shear_stress_value = shear_stress[indices[0]][indices[1]];

                    local_stress_integral[i] += stress_value * fe_values.JxW(q);
                    local_min_stress[i] = std::min(local_min_stress[i], stress_value);
                    local_max_stress[i] = std::max(local_max_stress[i], stress_value);

                    local_shear_stress_integral[i] += shear_stress_value * fe_values.JxW(q);
                    local_min_shear_stress[i] = std::min(local_min_shear_stress[i], shear_stress_value);
                    local_max_shear_stress[i] = std::max(local_max_shear_stress[i], shear_stress_value);
                  }
              }
          }

      // Global reductions
      std::vector<double> global_stress_integral(SymmetricTensor<2,dim>::n_independent_components);
      std::vector<double> global_min_stress(SymmetricTensor<2,dim>::n_independent_components);
      std::vector<double> global_max_stress(SymmetricTensor<2,dim>::n_independent_components);

      std::vector<double> global_shear_stress_integral(SymmetricTensor<2,dim>::n_independent_components);
      std::vector<double> global_min_shear_stress(SymmetricTensor<2,dim>::n_independent_components);
      std::vector<double> global_max_shear_stress(SymmetricTensor<2,dim>::n_independent_components);

      for (unsigned int i=0; i<SymmetricTensor<2,dim>::n_independent_components; ++i)
        {
          global_stress_integral[i] = Utilities::MPI::sum(local_stress_integral[i], this->get_mpi_communicator());
          global_shear_stress_integral[i] = Utilities::MPI::sum(local_shear_stress_integral[i], this->get_mpi_communicator());

          // For min/max, use the trick from pressure_statistics
          double local_vals[2] = {-local_min_stress[i], local_max_stress[i]};
          double global_vals[2];
          Utilities::MPI::max(local_vals, this->get_mpi_communicator(), global_vals);
          global_min_stress[i] = -global_vals[0];
          global_max_stress[i] = global_vals[1];

          local_vals[0] = -local_min_shear_stress[i];
          local_vals[1] = local_max_shear_stress[i];
          Utilities::MPI::max(local_vals, this->get_mpi_communicator(), global_vals);
          global_min_shear_stress[i] = -global_vals[0];
          global_max_shear_stress[i] = global_vals[1];
        }

      const double volume = this->get_volume();

      // Add statistics to table
      const std::vector<std::string> component_names = (dim == 2) ?
                                                       stress_component_names_2d :
                                                       stress_component_names_3d;

      for (unsigned int i=0; i<SymmetricTensor<2,dim>::n_independent_components; ++i)
        {
          const std::string &comp = component_names[i];
          const double avg_stress = global_stress_integral[i] / volume;
          const double avg_shear_stress = global_shear_stress_integral[i] / volume;

          statistics.add_value("Minimal stress " + comp + " (Pa)", global_min_stress[i]);
          statistics.add_value("Average stress " + comp + " (Pa)", avg_stress);
          statistics.add_value("Maximal stress " + comp + " (Pa)", global_max_stress[i]);

          statistics.add_value("Minimal shear stress " + comp + " (Pa)", global_min_shear_stress[i]);
          statistics.add_value("Average shear stress " + comp + " (Pa)", avg_shear_stress);
          statistics.add_value("Maximal shear stress " + comp + " (Pa)", global_max_shear_stress[i]);
        }

      // Set precision and scientific notation
      for (unsigned int i=0; i<SymmetricTensor<2,dim>::n_independent_components; ++i)
        {
          const std::string &comp = component_names[i];
          const std::vector<std::string> columns =
          {
            "Minimal stress " + comp + " (Pa)",
            "Average stress " + comp + " (Pa)",
            "Maximal stress " + comp + " (Pa)",
            "Minimal shear stress " + comp + " (Pa)",
            "Average shear stress " + comp + " (Pa)",
            "Maximal shear stress " + comp + " (Pa)"
          };
          for (const auto &col : columns)
            {
              statistics.set_precision(col, 8);
              statistics.set_scientific(col, true);
            }
        }

      std::ostringstream output;
      output.precision(4);
      output << std::scientific;
      output << "Shear stress min/avg/max (Pa):\n";
      for (unsigned int i=0; i<SymmetricTensor<2,dim>::n_independent_components; ++i)
        {
          const std::string &comp = component_names[i];
          output << std::setw(2) << std::left << comp << ": "
                 << std::right << std::setw(13) << global_min_shear_stress[i] << ' '
                 << std::setw(13) << global_shear_stress_integral[i] / volume << ' '
                 << std::setw(13) << global_max_shear_stress[i] << "\n";
        }
      return std::pair<std::string, std::string>("Stress statistics:", output.str());
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(StressStatistics,
                                  "stress statistics",
                                  "A postprocessor that computes some statistics about "
                                  "the stress fields.")
  }
}
