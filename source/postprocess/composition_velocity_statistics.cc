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


#include <aspect/postprocess/composition_velocity_statistics.h>
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
    CompositionVelocityStatistics<dim>::execute (TableHandler &statistics)
    {
      const QGauss<dim> quadrature_formula (this->get_fe()
                                            .base_element(this->introspection().base_elements.velocities).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);

      std::vector<Tensor<1,dim>> velocity_values(n_q_points);
      std::vector<double> compositional_values(n_q_points);

      std::vector<double> local_velocity_square_integral(this->n_compositional_fields());
      std::vector<double> local_min_velocity(this->n_compositional_fields(), std::numeric_limits<double>::max());
      std::vector<double> local_max_velocity(this->n_compositional_fields(), std::numeric_limits<double>::lowest());
      std::vector<double> local_area_integral(this->n_compositional_fields());

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);

            for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
              {
                fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(),
                    compositional_values);

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    if (compositional_values[q] >= 0.5)
                      {
                        const double vel_norm = velocity_values[q].norm();
                        local_velocity_square_integral[c] += vel_norm * vel_norm * fe_values.JxW(q);
                        local_min_velocity[c] = std::min(local_min_velocity[c], vel_norm);
                        local_max_velocity[c] = std::max(local_max_velocity[c], vel_norm);
                        local_area_integral[c] += fe_values.JxW(q);
                      }
                  }
              }
          }

      std::vector<double> global_velocity_square_integral(local_velocity_square_integral.size());
      std::vector<double> global_min_velocity(local_min_velocity.size());
      std::vector<double> global_max_velocity(local_max_velocity.size());
      std::vector<double> global_area_integral(local_area_integral.size());
      Utilities::MPI::sum(local_velocity_square_integral, this->get_mpi_communicator(), global_velocity_square_integral);
      Utilities::MPI::min(local_min_velocity, this->get_mpi_communicator(), global_min_velocity);
      Utilities::MPI::max(local_max_velocity, this->get_mpi_communicator(), global_max_velocity);
      Utilities::MPI::sum(local_area_integral, this->get_mpi_communicator(), global_area_integral);

      // compute the RMS velocity for each compositional field and for the selected compositional fields combined
      std::vector<double> vrms_per_composition(local_area_integral.size(), 0.0);
      std::vector<double> vmin_per_composition(local_area_integral.size(), 0.0);
      std::vector<double> vmax_per_composition(local_area_integral.size(), 0.0);
      double velocity_square_integral_selected_fields = 0., area_integral_selected_fields = 0.;
      for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
        {
          if (global_area_integral[c] > 0)
            {
              vrms_per_composition[c] = std::sqrt(global_velocity_square_integral[c]) /
                                        std::sqrt(global_area_integral[c]);
              vmin_per_composition[c] = global_min_velocity[c];
              vmax_per_composition[c] = global_max_velocity[c];
            }
          else
            {
              // leave the fields at their initialization value zero
              global_min_velocity[c] = 0.0;
              global_max_velocity[c] = 0.0;
            }

          const std::vector<std::string>::iterator selected_field_it = std::find(selected_fields.begin(), selected_fields.end(), this->introspection().name_for_compositional_index(c));
          if (selected_field_it != selected_fields.end())
            {
              velocity_square_integral_selected_fields += global_velocity_square_integral[c];
              area_integral_selected_fields += global_area_integral[c];
            }
        }

      double vrms_selected_fields = 0.;
      if (area_integral_selected_fields > 0)
        vrms_selected_fields = std::sqrt(velocity_square_integral_selected_fields) / std::sqrt(area_integral_selected_fields);
      else
        {
          // leave the field at its initialization value zero
        }

      const std::string unit = (this->convert_output_to_years()) ? "m/year" : "m/s";
      const double time_scaling = (this->convert_output_to_years()) ? year_in_seconds : 1.0;

      // finally produce something for the statistics file
      for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
        {
          statistics.add_value("Minimal velocity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                               time_scaling * vmin_per_composition[c]);
          statistics.add_value("Maximal velocity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                               time_scaling * vmax_per_composition[c]);
          statistics.add_value("RMS velocity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                               time_scaling * vrms_per_composition[c]);

          // also make sure that the other columns filled by this object
          // all show up with sufficient accuracy and in scientific notation
          const std::string columns[] = {"Minimal velocity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                                         "Maximal velocity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c),
                                         "RMS velocity (" + unit + ") for composition " + this->introspection().name_for_compositional_index(c)
                                        };
          for (const auto &column : columns)
            {
              statistics.set_precision(column, 8);
              statistics.set_scientific(column, true);
            }
        }

      // Also output the selected fields vrms
      if (selected_fields.size() > 0)
        {
          statistics.add_value("RMS velocity (" + unit + ") for the selected field(s)",
                               time_scaling * vrms_selected_fields);

          const std::string column = "RMS velocity (" + unit + ") for the selected field(s)";

          statistics.set_precision(column, 8);
          statistics.set_scientific(column, true);
        }

      std::ostringstream output;
      output.precision(4);

      // Find the maximum length of composition names for alignment
      unsigned int max_name_length = 0;
      for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
        {
          const std::string name = this->introspection().name_for_compositional_index(c);
          max_name_length = std::max(max_name_length, static_cast<unsigned int>(name.length()));
        }
      // Also consider "combined selected fields" if selected_fields is not empty
      if (selected_fields.size() > 0)
        max_name_length = std::max(max_name_length, static_cast<unsigned int>(std::string("combined selected fields").length()));

      for (unsigned int c = 0; c < this->n_compositional_fields(); ++c)
        {
          const std::string name = this->introspection().name_for_compositional_index(c);
          output << "[" << c << " (\"" << name << "\")]" << std::string(max_name_length - name.length(), ' ') << ": "                 << time_scaling *vmin_per_composition[c] << " / "
                 << time_scaling *vmax_per_composition[c] << " / "                 << time_scaling *vrms_per_composition[c] << " " << unit << "\n";
        }
      if (selected_fields.size() > 0)
        {
          const std::string combined_name = "combined selected fields";
          output << "[" << combined_name << "]" << std::string(max_name_length - combined_name.length(), ' ') << ": "
                 << time_scaling *vrms_selected_fields << " " << unit << "\n";
        }

      return std::pair<std::string, std::string> ("Composition velocity (min/max/rms):",
                                                  output.str());
    }



    template <int dim>
    void
    CompositionVelocityStatistics<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Composition velocity");
        {
          prm.declare_entry("Names of selected compositional fields", "",
                            Patterns::List(Patterns::Anything()),
                            "A list of names for each of the compositional fields that "
                            "you want to compute the combined RMS velocity for.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    CompositionVelocityStatistics<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Composition velocity");
        {
          selected_fields = Utilities::split_string_list(prm.get("Names of selected compositional fields"));

          // Check that the names given as selected_fields are actually fields.
          for (const std::string &field_name: selected_fields)
            {
              AssertThrow(this->introspection().compositional_name_exists(field_name),
                          ExcMessage("The entry '" + field_name + "' in the parameter "
                                     "<Names of selected compositional fields> in the composition velocity "
                                     "statistics postprocessor is not a valid name of a compositional field "
                                     "as specified in the <Compositional fields/Names of fields> parameter."));
            }

          AssertThrow(Utilities::has_unique_entries(selected_fields),
                      ExcMessage("The list of compositional fields for the parameter "
                                 "<Names of selected compositional fields> in the composition velocity "
                                 "statistics postprocessor contains entries more than once. "
                                 "This is not allowed. Please check your parameter file."));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CompositionVelocityStatistics,
                                  "Composition velocity",
                                  "A postprocessor that computes the root mean square velocity "
                                  "over the area spanned by each compositional field (i.e. where "
                                  "the field values are larger or equal to 0.5.")
  }
}
