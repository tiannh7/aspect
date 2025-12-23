/*
  Copyright (C) 2011 - 2023 by the authors of the ASPECT code.

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


#include <aspect/postprocess/composition_viscosity_statistics.h>
#include <aspect/global.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    CompositionViscosityStatistics<dim>::execute (TableHandler &statistics)
    {
      const QGauss<dim> quadrature_formula (this->get_fe()
                                            .base_element(this->introspection().base_elements.velocities).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_gradients |
                               update_quadrature_points |
                               update_JxW_values);

      std::vector<Tensor<1,dim>> velocity_values(n_q_points);
      std::vector<double> compositional_values(n_q_points);
      std::vector<Tensor<2,dim>> velocity_gradients(n_q_points);

      std::vector<double> local_viscosity_integral(this->n_compositional_fields());
      std::vector<double> local_min_viscosity(this->n_compositional_fields(), std::numeric_limits<double>::max());
      std::vector<double> local_max_viscosity(this->n_compositional_fields(), std::numeric_limits<double>::lowest());
      std::vector<double> local_area_integral(this->n_compositional_fields());

      // Prepare material inputs/outputs
      MaterialModel::MaterialModelInputs<dim> in(n_q_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(n_q_points, this->n_compositional_fields());
      in.requested_properties = MaterialModel::MaterialProperties::viscosity;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);
            fe_values[this->introspection().extractors.velocities].get_function_gradients (this->get_solution(),
                                                                                           velocity_gradients);

            // Fill material inputs
            in.position = fe_values.get_quadrature_points();
            in.temperature.resize(n_q_points);
            fe_values[this->introspection().extractors.temperature].get_function_values(this->get_solution(), in.temperature);
            in.pressure.resize(n_q_points);
            fe_values[this->introspection().extractors.pressure].get_function_values(this->get_solution(), in.pressure);

            // Get composition values
            for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
              {
                std::vector<double> composition_values_c(n_q_points);
                fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(), composition_values_c);
                for (unsigned int q=0; q<n_q_points; ++q)
                  in.composition[q][c] = composition_values_c[q];
              }

            // Strain rate
            for (unsigned int q=0; q<n_q_points; ++q)
              in.strain_rate[q] = symmetrize(velocity_gradients[q]);

            this->get_material_model().evaluate(in, out);

            for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
              {
                fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(),
                    compositional_values);

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    if (compositional_values[q] >= 0.5)
                      {
                        const double eta = out.viscosities[q];
                        local_viscosity_integral[c] += eta * fe_values.JxW(q);
                        local_min_viscosity[c] = std::min(local_min_viscosity[c], eta);
                        local_max_viscosity[c] = std::max(local_max_viscosity[c], eta);
                        local_area_integral[c] += fe_values.JxW(q);
                      }
                  }
              }
          }

      std::vector<double> global_viscosity_integral(local_viscosity_integral.size());
      std::vector<double> global_min_viscosity(local_min_viscosity.size());
      std::vector<double> global_max_viscosity(local_max_viscosity.size());
      std::vector<double> global_area_integral(local_area_integral.size());

      Utilities::MPI::sum(local_viscosity_integral, this->get_mpi_communicator(), global_viscosity_integral);
      Utilities::MPI::min(local_min_viscosity, this->get_mpi_communicator(), global_min_viscosity);
      Utilities::MPI::max(local_max_viscosity, this->get_mpi_communicator(), global_max_viscosity);
      Utilities::MPI::sum(local_area_integral, this->get_mpi_communicator(), global_area_integral);

      // compute the average viscosity for each compositional field
      std::vector<double> avg_viscosity_per_composition(local_area_integral.size(), 0.0);
      for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
        {
          if (global_area_integral[c] > 0)
            avg_viscosity_per_composition[c] = global_viscosity_integral[c] / global_area_integral[c];
          else
            {
              global_min_viscosity[c] = 0.0;
              global_max_viscosity[c] = 0.0;
              avg_viscosity_per_composition[c] = 0.0;
            }
        }

      const std::string unit = "Pa s";

      // finally produce something for the statistics file
      for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
        {
          statistics.add_value("Minimal viscosity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                               global_min_viscosity[c]);
          statistics.add_value("Average viscosity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                               avg_viscosity_per_composition[c]);
          statistics.add_value("Maximal viscosity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                               global_max_viscosity[c]);
        }

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          const std::string columns[] = { "Minimal viscosity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                                          "Average viscosity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                                          "Maximal viscosity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c)
                                        };
          for (const auto &col : columns)
            {
              statistics.set_precision (col, 8);
              statistics.set_scientific (col, true);
            }
        }

      std::ostringstream output;
      output.precision(4);

      // Find the maximum length of composition names for alignment
      unsigned int max_name_length = 0;
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          const std::string name = this->introspection().name_for_compositional_index(c);
          max_name_length = std::max(max_name_length, static_cast<unsigned int>(name.length()));
        }

      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          const std::string name = this->introspection().name_for_compositional_index(c);
          output << "[" << c << " (\"" << name << "\")]" << std::string(max_name_length - name.length(), ' ') << ": "
                 << global_min_viscosity[c] << " / "
                 << global_max_viscosity[c] << " / "
                 << avg_viscosity_per_composition[c] << " " << unit << "\n";
        }

      return std::pair<std::string, std::string> ("Composition viscosity (min/max/avg):",
                                                  output.str());
    }



    template <int dim>
    void
    CompositionViscosityStatistics<dim>::declare_parameters(ParameterHandler &prm)
    {
      (void)prm;
      // No parameters needed
    }



    template <int dim>
    void
    CompositionViscosityStatistics<dim>::parse_parameters(ParameterHandler &prm)
    {
      (void)prm;
      // No parameters to parse
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CompositionViscosityStatistics,
                                  "Composition viscosity",
                                  "A postprocessor that computes min/max/average viscosity "
                                  "over the area spanned by each compositional field (i.e. where "
                                  "the field values are larger or equal to 0.5).")
  }
}
