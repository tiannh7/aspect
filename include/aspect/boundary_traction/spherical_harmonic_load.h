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

#ifndef _aspect_boundary_traction_spherical_harmonic_load_h
#define _aspect_boundary_traction_spherical_harmonic_load_h

#include <aspect/boundary_traction/interface.h>
#include <aspect/simulator_access.h>

#include <string>

namespace aspect
{
  namespace BoundaryTraction
  {
    /**
     * A boundary traction plugin that prescribes a single real spherical
     * harmonic load. This is intended for benchmark input files where the
     * physical load is naturally described by degree and order rather than by a
     * hand-written Cartesian vector expression.
     *
     * @ingroup BoundaryTractions
     */
    template <int dim>
    class SphericalHarmonicLoad : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        /**
         * Return the boundary traction as a function of position. The
         * (outward) normal vector to the domain is also provided as a second
         * argument.
         */
        Tensor<1,dim>
        boundary_traction (const types::boundary_id boundary_indicator,
                           const Point<dim> &position,
                           const Tensor<1,dim> &normal_vector) const override;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm) override;

      private:
        /**
         * Evaluate the configured scalar spherical harmonic load at position.
         */
        double
        scalar_load (const Point<dim> &position) const;

        /**
         * Evaluate the unnormalized Legendre polynomial P_l(x).
         */
        double
        legendre_p (const unsigned int degree,
                    const double x) const;

        unsigned int harmonic_degree;
        unsigned int harmonic_order;
        double load_magnitude;
        bool use_load_height;
        double load_height;
        double load_density;
        std::string positive_load_direction;
        std::string coefficient_type;
        std::string normalization;
    };
  }
}

#endif
