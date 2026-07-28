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

#include <aspect/mesh_deformation/interface.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class LagrangianMeshVelocity : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &) override
        {
          if (this->get_timestep_number() == 0)
            return {"Maximum Lagrangian velocity mismatch:", "not evaluated"};

          const QGauss<dim> quadrature(this->get_fe().degree + 1);
          FEValues<dim> fe_values(this->get_mapping(),
                                  this->get_fe(),
                                  quadrature,
                                  update_values);
          std::vector<Tensor<1,dim>> material_velocity(quadrature.size());
          std::vector<Tensor<1,dim>> mesh_velocity(quadrature.size());

          double local_maximum_mismatch = 0.0;
          for (const auto &cell :
               this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values[this->introspection().extractors.velocities]
                .get_function_values(this->get_solution(), material_velocity);
                fe_values[this->introspection().extractors.velocities]
                .get_function_values(this->get_mesh_velocity(), mesh_velocity);

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  local_maximum_mismatch =
                    std::max(local_maximum_mismatch,
                             (material_velocity[q] - mesh_velocity[q]).norm());
              }

          const double maximum_mismatch =
            Utilities::MPI::max(local_maximum_mismatch,
                                this->get_mpi_communicator());

          double local_maximum_displacement = 0.0;
          const LinearAlgebra::Vector &mesh_displacements =
            this->get_mesh_deformation_handler().get_mesh_displacements();
          for (const auto index : mesh_displacements.locally_owned_elements())
            local_maximum_displacement =
              std::max(local_maximum_displacement,
                       std::abs(mesh_displacements[index]));
          const double maximum_displacement =
            Utilities::MPI::max(local_maximum_displacement,
                                this->get_mpi_communicator());

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << maximum_mismatch << ", "
                 << maximum_displacement;
          return {"Maximum Lagrangian velocity mismatch and mesh displacement:",
                  output.str()
                 };
        }
    };
  }
}



namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(
      LagrangianMeshVelocity,
      "lagrangian mesh velocity",
      "Verify that the full-Lagrangian mesh velocity matches a uniform "
      "material velocity and produces the expected mesh displacement.")
  }
}
