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


#ifndef _aspect_mesh_deformation_free_surface_h
#define _aspect_mesh_deformation_free_surface_h

#include <aspect/mesh_deformation/interface.h>

#include <aspect/simulator_access.h>
#include <aspect/simulator/assemblers/interface.h>


namespace aspect
{
  namespace MeshDeformation
  {
    /**
     * A plugin that computes the deformation of surface
     * vertices according to the solution of the flow problem.
     * In particular this means if the surface of the domain is
     * left open to flow, this flow will carry the mesh with it.
     *
     * @ingroup MeshDeformation
     */
    template <int dim>
    class FreeSurface : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Initialize function, which connects the set_assemblers function
         * to the appropriate Simulator signal.
         */
        void initialize() override;

        /**
         * Called by Simulator::set_assemblers() to allow the FreeSurface plugin
         * to register its assembler.
         */
        void set_assemblers(const SimulatorAccess<dim> &simulator_access,
                            aspect::Assemblers::Manager<dim> &assemblers) const;


        /**
         * A function that creates constraints for the velocity of certain mesh
         * vertices (e.g. the surface vertices) for a specific boundary.
         * The calling class will respect
         * these constraints when computing the new vertex positions.
         */
        void
        compute_velocity_constraints_on_boundary(const DoFHandler<dim> &mesh_deformation_dof_handler,
                                                 AffineConstraints<double> &mesh_velocity_constraints,
                                                 const std::set<types::boundary_id> &boundary_ids) const override;

        /**
         * Project the current Stokes velocity onto all free-surface boundaries
         * in the mesh-deformation finite element space. This is public so that
         * other boundary operators can use exactly the same discrete velocity
         * that will later advect the ALE mesh.
         */
        void project_velocity_onto_boundary (const DoFHandler<dim> &free_surface_dof_handler,
                                             const IndexSet &mesh_locally_owned,
                                             const IndexSet &mesh_locally_relevant,
                                             LinearAlgebra::Vector &output) const;

        /**
         * Apply the mesh-space boundary mass solve used by
         * project_velocity_onto_boundary() to an arbitrary vector field. This
         * exposes the exact production projection operator for constructing
         * its discrete adjoint without duplicating its quadrature, mapping, or
         * normal/vertical projection choices.
         */
        void project_boundary_field_onto_boundary (
          const DoFHandler<dim> &free_surface_dof_handler,
          const IndexSet &mesh_locally_owned,
          const IndexSet &mesh_locally_relevant,
          const std::function<Tensor<1,dim>(const Point<dim> &,
                                            const Tensor<1,dim> &)> &field,
          const bool project_along_surface_direction,
          LinearAlgebra::Vector &output) const;

        /** Return the normalized direction used by the production projector. */
        Tensor<1,dim> projection_direction (const Point<dim> &position,
                                            const Tensor<1,dim> &normal) const;

        /**
         * Returns whether or not the plugin requires surface stabilization
         */
        bool needs_surface_stabilization () const override;

        /**
         * Declare parameters for the free surface handling.
         */
        static
        void declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters for the free surface handling.
         */
        void parse_parameters (ParameterHandler &prm) override;

      private:
        /**
         * A struct for holding information about how to advect the free surface.
         */
        struct SurfaceAdvection
        {
          enum Direction { normal, vertical };
        };

        /**
         * Stores whether to advect the free surface in the normal direction
         * or the direction of the local vertical.
         */
        typename SurfaceAdvection::Direction advection_direction;

        /**
         * Linear algebra objects for the boundary mass projection. Their
         * sparsity and vector maps only depend on the mesh topology and
         * partitioning, so keep them across repeated post-Stokes projections
         * and only reassemble their values on the current ALE geometry.
         */
        mutable LinearAlgebra::SparseMatrix boundary_projection_mass_matrix;
        mutable LinearAlgebra::Vector boundary_projection_rhs;
        mutable LinearAlgebra::Vector boundary_projection_solution;
        mutable IndexSet boundary_projection_locally_owned;
    };
  }
}


#endif
