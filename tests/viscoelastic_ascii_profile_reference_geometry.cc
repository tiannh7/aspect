/*
  Copyright (C) 2026 by the authors of the ASPECT code.

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

#include <aspect/geometry_model/interface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace MeshDeformation
  {
    template <int dim>
    class ReferenceProfileDeformation : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim>
        compute_initial_deformation_on_boundary(
          const types::boundary_id,
          const Point<dim> &position) const override
        {
          Tensor<1,dim> displacement;
          displacement[dim-1] =
            2.0e4 * std::cos(2.0 * numbers::PI * position[0] / 1.0e5);
          return displacement;
        }
    };
  }



  namespace Postprocess
  {
    template <int dim>
    class AsciiProfileReferenceGeometry : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &) override
        {
          const QGauss<dim> quadrature(3);
          FEValues<dim> current_fe_values(this->get_mapping(),
                                          this->get_fe(),
                                          quadrature,
                                          update_values |
                                          update_gradients |
                                          update_quadrature_points);
          const MappingQ1<dim> reference_mapping;
          FEValues<dim> reference_fe_values(reference_mapping,
                                            this->get_fe(),
                                            quadrature,
                                            update_quadrature_points);

          MaterialModel::MaterialModelInputs<dim>
          inputs(quadrature.size(), this->n_compositional_fields());
          MaterialModel::MaterialModelOutputs<dim>
          outputs(quadrature.size(), this->n_compositional_fields());
          inputs.requested_properties =
            MaterialModel::MaterialProperties::density;

          double local_maximum_density_error = 0.0;
          for (const auto &cell :
               this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                current_fe_values.reinit(cell);
                reference_fe_values.reinit(cell);
                inputs.reinit(current_fe_values,
                              cell,
                              this->introspection(),
                              this->get_solution());
                this->get_material_model().fill_additional_material_model_inputs(
                  inputs,
                  this->get_solution(),
                  current_fe_values,
                  this->introspection());
                this->get_material_model().evaluate(inputs, outputs);

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    const double reference_depth =
                      this->get_geometry_model().depth(
                        reference_fe_values.quadrature_point(q));
                    const double expected_density =
                      2800.0 + 500.0 * reference_depth / 1.0e5;
                    local_maximum_density_error =
                      std::max(local_maximum_density_error,
                               std::abs(outputs.densities[q]
                                        - expected_density));
                  }
              }

          const double maximum_density_error =
            Utilities::MPI::max(local_maximum_density_error,
                                this->get_mpi_communicator());

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << maximum_density_error;
          return {"Maximum reference-profile density error:",
                  output.str()
                 };
        }
    };
  }
}



namespace aspect
{
  namespace MeshDeformation
  {
    ASPECT_REGISTER_MESH_DEFORMATION_MODEL(
      ReferenceProfileDeformation,
      "prescribed reference-profile deformation",
      "Prescribe a laterally varying initial deformation for the reference "
      "geometry ASCII profile test.")
  }

  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(
      AsciiProfileReferenceGeometry,
      "ascii profile reference geometry",
      "Compare an ASCII profile evaluated on a deformed mesh against the "
      "profile depth of the corresponding reference-mesh points.")
  }
}
