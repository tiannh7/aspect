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


#ifndef _aspect_density_source_manager_h
#define _aspect_density_source_manager_h

#include <aspect/simulator_access.h>
#include <aspect/material_model/interface.h>

namespace aspect
{
  /**
   * Centralize density selection for perturbation calculations without
   * changing the physical-density contract of material model outputs.
   *
   * Typed accessors preserve the different historical meanings of Stokes and
   * internal self-gravity density sources. This initial implementation only
   * centralizes legacy behavior.
   */
  template <int dim>
  class DensitySourceManager : public SimulatorAccess<dim>
  {
    public:
      /**
       * Return the physical density produced by the material model.
       */
      double
      physical_density(
        const MaterialModel::MaterialModelOutputs<dim> &outputs,
        const unsigned int q) const;

      /**
       * Return the selected reference density. Legacy mode has no central
       * reference-density model and therefore returns zero.
       */
      double
      reference_density(const Point<dim> &position) const;

      /**
       * Return the density used by the Stokes momentum body force. Legacy
       * behavior uses the full physical material density.
       */
      double
      stokes_source_density(
        const MaterialModel::MaterialModelInputs<dim> &inputs,
        const MaterialModel::MaterialModelOutputs<dim> &outputs,
        const unsigned int q) const;

      /**
       * Return the density used by internal self-gravity volume and degree-1
       * mass-dipole integrals. Legacy behavior subtracts the scalar reference
       * density owned by the existing self-gravity settings.
       */
      double
      self_gravity_source_density(
        const MaterialModel::MaterialModelInputs<dim> &inputs,
        const MaterialModel::MaterialModelOutputs<dim> &outputs,
        const unsigned int q,
        const double legacy_reference_density) const;
  };
}

#endif
