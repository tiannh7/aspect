/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/density_source_manager.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class DensitySourceManagerLegacy : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &statistics) override
        {
          const QGauss<dim> quadrature(this->introspection().polynomial_degree.temperature + 1);
          FEValues<dim> fe_values(this->get_mapping(),
                                  this->get_fe(),
                                  quadrature,
                                  update_values |
                                  update_quadrature_points |
                                  update_gradients);

          MaterialModel::MaterialModelInputs<dim>
          inputs(quadrature.size(), this->n_compositional_fields());
          MaterialModel::MaterialModelOutputs<dim>
          outputs(quadrature.size(), this->n_compositional_fields());
          inputs.requested_properties = MaterialModel::MaterialProperties::density;

          double local_min_physical_density = std::numeric_limits<double>::max();
          double local_max_physical_density = -std::numeric_limits<double>::max();
          double local_max_stokes_error = 0.0;
          double local_max_self_gravity_error = 0.0;

          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                inputs.reinit(fe_values,
                              cell,
                              this->introspection(),
                              this->get_solution());
                this->get_material_model().evaluate(inputs, outputs);

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    const double physical_density =
                      this->get_density_source_manager().physical_density(outputs, q);
                    const double stokes_density =
                      this->get_density_source_manager().stokes_source_density(inputs,
                                                                               outputs,
                                                                               q);
                    const double self_gravity_density =
                      this->get_density_source_manager().self_gravity_source_density(inputs,
                                                                                     outputs,
                                                                                     q,
                                                                                     3300.0);

                    local_min_physical_density =
                      std::min(local_min_physical_density, physical_density);
                    local_max_physical_density =
                      std::max(local_max_physical_density, physical_density);
                    local_max_stokes_error =
                      std::max(local_max_stokes_error,
                               std::abs(stokes_density - physical_density));
                    local_max_self_gravity_error =
                      std::max(local_max_self_gravity_error,
                               std::abs(self_gravity_density
                                        - 100.0 * inputs.composition[q][0]));
                  }
              }

          const double min_physical_density =
            Utilities::MPI::min(local_min_physical_density,
                                this->get_mpi_communicator());
          const double max_physical_density =
            Utilities::MPI::max(local_max_physical_density,
                                this->get_mpi_communicator());
          const double max_stokes_error =
            Utilities::MPI::max(local_max_stokes_error,
                                this->get_mpi_communicator());
          const double max_self_gravity_error =
            Utilities::MPI::max(local_max_self_gravity_error,
                                this->get_mpi_communicator());

          statistics.add_value("Minimum physical density", min_physical_density);
          statistics.add_value("Maximum physical density", max_physical_density);
          statistics.add_value("Maximum legacy Stokes density error", max_stokes_error);
          statistics.add_value("Maximum legacy self-gravity density error", max_self_gravity_error);
          statistics.set_precision("Minimum physical density", 12);
          statistics.set_precision("Maximum physical density", 12);
          statistics.set_precision("Maximum legacy Stokes density error", 12);
          statistics.set_precision("Maximum legacy self-gravity density error", 12);

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << min_physical_density << " / "
                 << max_physical_density << "; errors "
                 << max_stokes_error << " / "
                 << max_self_gravity_error;
          return {"Density source manager legacy:", output.str()};
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(DensitySourceManagerLegacy,
                                  "density source manager legacy",
                                  "Test the legacy typed density-source accessors with a "
                                  "composition-dependent material density.")
  }
}
