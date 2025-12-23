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


#include <aspect/postprocess/velocity_statistics.h>
#include <aspect/material_model/simple.h>
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    VelocityStatistics<dim>::execute (TableHandler &statistics)
    {
      const bool output_spherical = this->get_postprocess_manager().get_output_in_spherical_coordinates();

      // Component names depending on coordinate system
      const bool is_spherical_like = (this->get_geometry_model().natural_coordinate_system() != Utilities::Coordinates::cartesian);
      const std::vector<std::string> velocity_component_names = (output_spherical || is_spherical_like)
                                                                ? std::vector<std::string> {"r","phi","theta"}
                                                                :
                                                                std::vector<std::string> {"x","y","z"};

      // Use a Gauss-Lobatto quadrature rule so that we do not need to use two separate quadratures
      // for the maximum velocity (ideally we would use Trapezoidal quadrature) and the RMS velocity
      // (ideally we would use Gauss quadrature).
      const QGaussLobatto<dim> quadrature_formula(this->get_fe().base_element(this->introspection().base_elements.velocities).degree + 2);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_values   |
                              update_quadrature_points |
                              update_JxW_values);

      std::vector<Tensor<1,dim>> velocity_values(n_q_points);

      std::vector<double> local_velocity_square_integral(dim, 0.0);
      std::vector<double> local_min_velocity(dim, std::numeric_limits<double>::max());
      std::vector<double> local_max_velocity(dim, std::numeric_limits<double>::lowest());
      double local_min_velocity_magnitude = std::numeric_limits<double>::max();
      double local_max_velocity_magnitude = 0.0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                          velocity_values);
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  double vel_magnitude_sq = 0.0;
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      const double vel_comp = velocity_values[q][d];
                      local_velocity_square_integral[d] += vel_comp * vel_comp * fe_values.JxW(q);
                      local_min_velocity[d] = std::min(local_min_velocity[d], vel_comp);
                      local_max_velocity[d] = std::max(local_max_velocity[d], vel_comp);
                      vel_magnitude_sq += vel_comp * vel_comp;
                    }
                  const double vel_magnitude = std::sqrt(vel_magnitude_sq);
                  local_min_velocity_magnitude = std::min(local_min_velocity_magnitude, vel_magnitude);
                  local_max_velocity_magnitude = std::max(local_max_velocity_magnitude, vel_magnitude);
                }
            }
        }

      std::vector<double> global_velocity_square_integral(dim);
      std::vector<double> global_min_velocity(dim);
      std::vector<double> global_max_velocity(dim);
      double global_min_velocity_magnitude;
      double global_max_velocity_magnitude;

      for (unsigned int d = 0; d < dim; ++d)
        {
          global_velocity_square_integral[d] = Utilities::MPI::sum(local_velocity_square_integral[d], this->get_mpi_communicator());
          global_min_velocity[d] = Utilities::MPI::min(local_min_velocity[d], this->get_mpi_communicator());
          global_max_velocity[d] = Utilities::MPI::max(local_max_velocity[d], this->get_mpi_communicator());
        }
      global_min_velocity_magnitude = Utilities::MPI::min(local_min_velocity_magnitude, this->get_mpi_communicator());
      global_max_velocity_magnitude = Utilities::MPI::max(local_max_velocity_magnitude, this->get_mpi_communicator());

      const double volume = this->get_volume();
      std::vector<double> vrms_per_component(dim);
      for (unsigned int d = 0; d < dim; ++d)
        vrms_per_component[d] = std::sqrt(global_velocity_square_integral[d]) / std::sqrt(volume);

      double vrms_total = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
        vrms_total += vrms_per_component[d] * vrms_per_component[d];
      vrms_total = std::sqrt(vrms_total);

      const std::string units = (this->convert_output_to_years() == true) ? "m/year" : "m/s";
      const double unit_scale_factor = (this->convert_output_to_years() == true) ? year_in_seconds : 1.0;

      // Add to statistics table
      for (unsigned int d = 0; d < dim; ++d)
        {
          const std::string base = "Velocity " + velocity_component_names[d] + " (" + units + ")";

          const std::string name_min = "Min. " + base;
          const std::string name_rms = "RMS " + base;
          const std::string name_max = "Max. " + base;

          statistics.add_value(name_min, global_min_velocity[d] * unit_scale_factor);
          statistics.add_value(name_rms, vrms_per_component[d] * unit_scale_factor);
          statistics.add_value(name_max, global_max_velocity[d] * unit_scale_factor);

          statistics.set_precision(name_min, 8);
          statistics.set_scientific(name_min, true);
          statistics.set_precision(name_rms, 8);
          statistics.set_scientific(name_rms, true);
          statistics.set_precision(name_max, 8);
          statistics.set_scientific(name_max, true);
        }

      statistics.add_value("RMS velocity (" + units + ")", vrms_total * unit_scale_factor);
      statistics.add_value("Max. velocity (" + units + ")", global_max_velocity_magnitude * unit_scale_factor);

      statistics.set_precision("RMS velocity (" + units + ")", 8);
      statistics.set_scientific("RMS velocity (" + units + ")", true);
      statistics.set_precision("Max. velocity (" + units + ")", 8);
      statistics.set_scientific("Max. velocity (" + units + ")", true);

      std::ostringstream output;
      output.precision(4);

      // Find the maximum length of component names for alignment
      unsigned int max_name_length = 5; // for "total"
      for (unsigned int d = 0; d < dim; ++d)
        max_name_length = std::max(max_name_length, static_cast<unsigned int>(velocity_component_names[d].length()));

      for (unsigned int d = 0; d < dim; ++d)
        {
          output << velocity_component_names[d] << std::string(max_name_length - velocity_component_names[d].length(), ' ') << ": "
                 << global_min_velocity[d] * unit_scale_factor << " / "
                 << global_max_velocity[d] * unit_scale_factor << " / "
                 << vrms_per_component[d] * unit_scale_factor << " " << units << "\n";
        }

      output << "total" << std::string(max_name_length - 5, ' ') << ": "
             << global_min_velocity_magnitude *unit_scale_factor << " / "
             << global_max_velocity_magnitude *unit_scale_factor << " / "
             << vrms_total *unit_scale_factor << " " << units << "\n";

      return std::pair<std::string, std::string> ("Velocity statistics (min/max/rms):",
                                                  output.str());
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(VelocityStatistics,
                                  "velocity statistics",
                                  "A postprocessor that computes the root mean square and "
                                  "maximum velocity in the computational domain.")
  }
}
