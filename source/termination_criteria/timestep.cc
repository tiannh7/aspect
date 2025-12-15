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

#include <aspect/termination_criteria/timestep.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace TerminationCriteria
  {
    template <int dim>
    bool
    Timestep<dim>::execute()
    {
      const double time_scale = this->convert_output_to_years() ? constants::year_in_seconds : 1.0;
      const double dt = this->get_timestep() / time_scale;

      // Create a point with the timestep value
      Point<1> p(dt);

      // Evaluate the function
      const double result = function.value(p);

      // Terminate if the function evaluates to a positive value
      if (result > 0)
        {
          const std::string unit = this->convert_output_to_years() ? "yr" : "s";
          this->get_pcout() << "Timestep termination criterion met: dt = " << dt
                            << " " << unit << " satisfies the condition in expression '"
                            << function_expression << "'" << std::endl;
          return true;
        }

      return false;
    }

    template <int dim>
    void
    Timestep<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("Timestep");
        {
          Functions::ParsedFunction<1>::declare_parameters (prm, 1);
          prm.declare_entry("Function expression", "0",
                            Patterns::Anything(),
                            "Expression for the termination criterion as a function of 'dt' (timestep). "
                            "The simulation terminates if this expression evaluates to a positive value. "
                            "Units of dt: years if the "
                            "'Use years instead of seconds' parameter is set; "
                            "seconds otherwise.");
          prm.declare_entry("Variable names", "dt",
                            Patterns::Anything(),
                            "Name for the variable representing the current timestep. "
                            "Currently only 'dt' is supported.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    Timestep<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("Timestep");
        {
          function_expression = prm.get("Function expression");
          try
            {
              function.parse_parameters (prm);
            }
          catch (...)
            {
              std::cerr << "ERROR: FunctionParser failed to parse\n"
                        << "\t'Termination criteria.Timestep'\n"
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
    ASPECT_REGISTER_TERMINATION_CRITERION(Timestep,
                                          "timestep",
                                          "Terminate the simulation when a user-defined function "
                                          "of the timestep evaluates to a positive value.")
  }
}
