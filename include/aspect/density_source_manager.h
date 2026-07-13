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
   * internal self-gravity density sources while allowing non-legacy consumers
   * to share one explicit source law.
   */
  template <int dim>
  class DensitySourceManager : public SimulatorAccess<dim>
  {
    public:
      struct Diagnostics
      {
        double integrated_mass = 0.0;
        double l2_norm = 0.0;
        double max_abs = 0.0;
        double max_lateral_average_residual = 0.0;
      };

      /**
       * Compute and store a frozen initial lateral-average reference-density
       * profile when that reference model is selected.
       */
      void
      initialize_reference_density();

      /**
       * Return the physical density produced by the material model.
       */
      double
      physical_density(
        const MaterialModel::MaterialModelOutputs<dim> &outputs,
        const unsigned int q) const;

      /**
       * Return the selected central reference density. The default `none'
       * reference-density model returns zero.
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

      /** Return whether a frozen reference profile has been initialized. */
      bool
      has_initialized_reference_density() const;

      /** Return stored frozen-profile depth coordinates. */
      const std::vector<double> &
      get_depth_samples() const;

      /** Return stored frozen reference-density values. */
      const std::vector<double> &
      get_reference_density_values() const;

      /** Return diagnostics evaluated immediately after initialization. */
      const Diagnostics &
      get_initial_diagnostics() const;

      /** Return how many times the frozen profile was initialized. */
      unsigned int
      get_initialization_count() const;

      /** Return a scale used for source auto-detection tolerances. */
      double
      get_reference_density_scale() const;

    private:
      /** Compute diagnostics for the stored frozen profile. */
      Diagnostics
      compute_initial_diagnostics() const;

      /** Return the depth-bin index that contains @p depth. */
      unsigned int
      depth_bin_index(const double depth) const;

      std::vector<double> depth_bounds;
      std::vector<double> depth_samples;
      std::vector<double> reference_density_values;
      Diagnostics initial_diagnostics;
      bool initialized = false;
      unsigned int initialization_count = 0;
      double reference_density_scale = 0.0;
  };
}

#endif
