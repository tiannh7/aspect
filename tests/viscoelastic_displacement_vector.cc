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
#include <array>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class ViscoelasticDisplacementVector : public Interface<dim>,
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
          std::array<std::vector<double>,dim> displacement_values;
          for (auto &component_values : displacement_values)
            component_values.resize(quadrature.size());

          double local_component_error = 0.0;
          double local_radial_projection_error = 0.0;
          const unsigned int committed_history_factor =
            std::max(1u, this->get_timestep_number());

          const unsigned int radial_index =
            this->introspection().compositional_index_for_name(
              "ve_radial_displacement");
          const std::array<std::string,2> displacement_names =
          {
            "ve_displacement_x",
            "ve_displacement_y"
          };

          for (const auto &cell :
               this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values[
                  this->introspection().extractors.compositional_fields[
                    radial_index]]
                .get_function_values(this->get_solution(),
                                     radial_displacement_values);

                for (unsigned int d = 0; d < dim; ++d)
                  {
                    const unsigned int displacement_index =
                      this->introspection().compositional_index_for_name(
                        displacement_names[d]);
                    fe_values[
                      this->introspection().extractors.compositional_fields[
                        displacement_index]]
                    .get_function_values(this->get_solution(),
                                         displacement_values[d]);
                  }

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    Tensor<1,dim> displacement;
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        displacement[d] = displacement_values[d][q];
                        const double expected_component =
                          0.01 * committed_history_factor
                          * fe_values.quadrature_point(q)[d];
                        local_component_error =
                          std::max(local_component_error,
                                   std::abs(displacement[d]
                                            - expected_component));
                      }

                    const Point<dim> position =
                      fe_values.quadrature_point(q);
                    const Tensor<1,dim> radial_unit =
                      position / position.norm();
                    local_radial_projection_error =
                      std::max(local_radial_projection_error,
                               std::abs(displacement * radial_unit
                                        - radial_displacement_values[q]));
                  }
              }

          const double component_error =
            Utilities::MPI::max(local_component_error,
                                this->get_mpi_communicator());
          const double radial_projection_error =
            Utilities::MPI::max(local_radial_projection_error,
                                this->get_mpi_communicator());

          AssertThrow(component_error < 1e-4,
                      ExcMessage("Vector material-displacement components "
                                 "do not match the prescribed displacement."));
          AssertThrow(radial_projection_error < 1e-5,
                      ExcMessage("The vector material-displacement radial "
                                 "projection does not match the scalar radial "
                                 "history."));

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << "timestep " << this->get_timestep_number()
                 << ", committed factor " << committed_history_factor
                 << ", component error " << component_error
                 << ", radial projection error "
                 << radial_projection_error;
          return {"Vector material displacement:", output.str()};
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(
      ViscoelasticDisplacementVector,
      "viscoelastic displacement vector",
      "Test Cartesian vector material-displacement accumulation and its "
      "radial projection against the existing scalar radial history.")
  }
}
