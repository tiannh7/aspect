/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

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

#include <aspect/termination_criteria/rms.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace TerminationCriteria
  {
    template <int dim>
    double
    RMS<dim>::compute_rms_velocity() const
    {
      const Quadrature<dim> &quadrature_formula = this->introspection().quadratures.velocities;
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);
      std::vector<Tensor<1,dim>> velocity_values(n_q_points);

      double local_velocity_square_integral = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                local_velocity_square_integral += ((velocity_values[q] * velocity_values[q]) *
                                                   fe_values.JxW(q));
              }
          }

      const double global_velocity_square_integral
        = Utilities::MPI::sum (local_velocity_square_integral, this->get_mpi_communicator());

      // Calculate the global root mean square velocity
      const double vrms = std::sqrt(global_velocity_square_integral) / std::sqrt(this->get_volume());

      return vrms;
    }

    template <int dim>
    bool
    RMS<dim>::execute()
    {
      const double vrms_si = compute_rms_velocity();
      const double vrms = vrms_si * (this->convert_output_to_years() ? year_in_seconds : 1.0);

      // Create a point with the RMS velocity value
      Point<1> p(vrms);

      // Evaluate the function
      const double result = function.value(p);

      // Terminate if the function evaluates to a positive value
      if (result > 0)
        {
          const std::string unit = this->convert_output_to_years() ? "m/yr" : "m/s";
          this->get_pcout() << "RMS termination criterion met: vrms = " << vrms
                            << " " << unit << " satisfies the condition in expression '"
                            << function_expression << "'" << std::endl;
          return true;
        }

      return false;
    }

    template <int dim>
    void
    RMS<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("RMS velocity");
        {
          Functions::ParsedFunction<1>::declare_parameters (prm, 1);
          prm.declare_entry("Function expression", "0",
                            Patterns::Anything(),
                            "Expression for the termination criterion as a function of 'vrms' (RMS velocity). "
                            "The simulation terminates if this expression evaluates to a positive value. "
                            "Units of vrms: m/yr if the "
                            "'Use years instead of seconds' parameter is set; "
                            "m/s otherwise.");
          prm.declare_entry("Variable names", "vrms",
                            Patterns::Anything(),
                            "Name for the variable representing the current RMS velocity. "
                            "Currently only 'vrms' is supported.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    RMS<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("RMS velocity");
        {
          function_expression = prm.get("Function expression");
          try
            {
              function.parse_parameters (prm);
            }
          catch (...)
            {
              std::cerr << "ERROR: FunctionParser failed to parse\n"
                        << "\t'Termination criteria.RMS velocity'\n"
                        << "with expression\n"
                        << "\t'" << function_expression << "'"
                        << "More information about the cause of the parse error \n"
                        << "is shown below.\n";
              throw;
            }
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace TerminationCriteria
  {
    ASPECT_REGISTER_TERMINATION_CRITERION(RMS,
                                          "rms velocity",
                                          "Terminate the simulation when a user-defined function "
                                          "of the RMS velocity evaluates to a positive value.")
  }
}
