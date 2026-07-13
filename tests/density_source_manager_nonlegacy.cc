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
    class DensitySourceManagerNonlegacy : public Interface<dim>,
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
          double local_min_reference_density = std::numeric_limits<double>::max();
          double local_max_reference_density = -std::numeric_limits<double>::max();
          double local_max_source_density = 0.0;
          double local_max_source_mismatch = 0.0;

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
                    const auto &manager = this->get_density_source_manager();
                    const double physical_density = manager.physical_density(outputs, q);
                    const double reference_density = manager.reference_density(inputs.position[q]);
                    const double stokes_density =
                      manager.stokes_source_density(inputs, outputs, q);
                    const double self_gravity_density =
                      manager.self_gravity_source_density(inputs, outputs, q, 1234.0);

                    local_min_physical_density =
                      std::min(local_min_physical_density, physical_density);
                    local_max_physical_density =
                      std::max(local_max_physical_density, physical_density);
                    local_min_reference_density =
                      std::min(local_min_reference_density, reference_density);
                    local_max_reference_density =
                      std::max(local_max_reference_density, reference_density);
                    local_max_source_density =
                      std::max(local_max_source_density, std::abs(stokes_density));
                    local_max_source_mismatch =
                      std::max(local_max_source_mismatch,
                               std::abs(stokes_density - self_gravity_density));
                  }
              }

          const double min_physical_density =
            Utilities::MPI::min(local_min_physical_density,
                                this->get_mpi_communicator());
          const double max_physical_density =
            Utilities::MPI::max(local_max_physical_density,
                                this->get_mpi_communicator());
          const double min_reference_density =
            Utilities::MPI::min(local_min_reference_density,
                                this->get_mpi_communicator());
          const double max_reference_density =
            Utilities::MPI::max(local_max_reference_density,
                                this->get_mpi_communicator());
          const double max_source_density =
            Utilities::MPI::max(local_max_source_density,
                                this->get_mpi_communicator());
          const double max_source_mismatch =
            Utilities::MPI::max(local_max_source_mismatch,
                                this->get_mpi_communicator());

          statistics.add_value("Minimum physical density", min_physical_density);
          statistics.add_value("Maximum physical density", max_physical_density);
          statistics.add_value("Minimum central reference density", min_reference_density);
          statistics.add_value("Maximum central reference density", max_reference_density);
          statistics.add_value("Maximum volume source density", max_source_density);
          statistics.add_value("Maximum Stokes self-gravity source mismatch", max_source_mismatch);

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << "physical " << min_physical_density << " / " << max_physical_density
                 << ", reference " << min_reference_density << " / " << max_reference_density
                 << ", max source " << max_source_density
                 << ", mismatch " << max_source_mismatch
                 << ", initializations "
                 << this->get_density_source_manager().get_initialization_count();

          if (this->get_density_source_manager().has_initialized_reference_density())
            {
              const auto &diagnostics =
                this->get_density_source_manager().get_initial_diagnostics();
              output << ", initial mass " << diagnostics.integrated_mass
                     << ", lateral residual "
                     << diagnostics.max_lateral_average_residual;
            }

          return {"Density source manager nonlegacy:", output.str()};
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(DensitySourceManagerNonlegacy,
                                  "density source manager nonlegacy",
                                  "Test non-legacy reference-density models and volume-source laws.")
  }
}
