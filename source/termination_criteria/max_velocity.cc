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

#include <aspect/termination_criteria/max_velocity.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace TerminationCriteria
  {
    template <int dim>
    double
    MaxVelocity<dim>::compute_max_velocity() const
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

      double local_max_velocity = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const double velocity_magnitude = velocity_values[q].norm();
                local_max_velocity = std::max(local_max_velocity, velocity_magnitude);
              }
          }

      const double global_max_velocity
        = Utilities::MPI::max (local_max_velocity, this->get_mpi_communicator());

      return global_max_velocity;
    }

    template <int dim>
    bool
    MaxVelocity<dim>::execute()
    {
      const double vmax_si = compute_max_velocity();
      const double vmax = vmax_si * (this->convert_output_to_years() ? year_in_seconds : 1.0);

      // Create a point with the maximum velocity value
      Point<1> p(vmax);

      // Evaluate the function
      const double result = function.value(p);

      // Terminate if the function evaluates to a positive value
      if (result > 0)
        {
          const std::string unit = this->convert_output_to_years() ? "m/yr" : "m/s";
          this->get_pcout() << "Maximum velocity termination criterion met: vmax = " << vmax
                            << " " << unit << " satisfies the condition in expression '"
                            << function_expression << "'" << std::endl;
          return true;
        }

      return false;
    }

    template <int dim>
    void
    MaxVelocity<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("Maximum velocity");
        {
          Functions::ParsedFunction<1>::declare_parameters (prm, 1);
          prm.declare_entry("Function expression", "0",
                            Patterns::Anything(),
                            "Expression for the termination criterion as a function of 'vmax' (maximum velocity). "
                            "The simulation terminates if this expression evaluates to a positive value. "
                            "Units of vmax: m/yr if the "
                            "'Use years instead of seconds' parameter is set; "
                            "m/s otherwise.");
          prm.declare_entry("Variable names", "vmax",
                            Patterns::Anything(),
                            "Name for the variable representing the current maximum velocity. "
                            "Currently only 'vmax' is supported.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    MaxVelocity<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("Maximum velocity");
        {
          function_expression = prm.get("Function expression");
          try
            {
              function.parse_parameters (prm);
            }
          catch (...)
            {
              std::cerr << "ERROR: FunctionParser failed to parse\n"
                        << "\t'Termination criteria.Maximum velocity'\n"
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
    ASPECT_REGISTER_TERMINATION_CRITERION(MaxVelocity,
                                          "maximum velocity",
                                          "Terminate the simulation when a user-defined function "
                                          "of the maximum velocity evaluates to a positive value.")
  }
}
