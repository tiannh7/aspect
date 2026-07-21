/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class ViscoelasticDisplacementOnlyRadial : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &/*statistics*/) override
        {
          AssertThrow(dim == 2, ExcNotImplemented());

          const QGauss<dim> quadrature(2);
          FEValues<dim> fe_values(this->get_mapping(),
                                  this->get_fe(),
                                  quadrature,
                                  update_values | update_quadrature_points);
          std::vector<double> radial_displacement_values(quadrature.size());
          double local_radial_displacement_error = 0.0;
          // Operator splitting commits the velocity increment from timestep n
          // at the beginning of timestep n+1. Timestep zero is initialized
          // explicitly, so timestep one must not commit that increment twice.
          const unsigned int committed_history_factor =
            std::max(1u, this->get_timestep_number());
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values[this->introspection().extractors.compositional_fields[
                            this->introspection().compositional_index_for_name("ve_radial_displacement")]]
                .get_function_values(this->get_solution(), radial_displacement_values);

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    local_radial_displacement_error =
                      std::max(local_radial_displacement_error,
                               std::abs(radial_displacement_values[q]
                                        - 0.01 * committed_history_factor
                                        * fe_values.quadrature_point(q).norm()));
                  }
              }

          const double radial_displacement_error =
            Utilities::MPI::max(local_radial_displacement_error,
                                this->get_mpi_communicator());

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << "timestep " << this->get_timestep_number()
                 << ", committed factor " << committed_history_factor
                 << ", radial history error " << radial_displacement_error;
          return {"Reference-mesh radial history:", output.str()};
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(ViscoelasticDisplacementOnlyRadial,
                                  "viscoelastic displacement-only radial",
                                  "Test reference-mesh radial displacement accumulation without compositional advection.")
  }
}
