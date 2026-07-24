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

#include <deal.II/base/symmetric_tensor.h>

#include <functional>
#include <string>

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
       * Low-order moments of mantle volume-density perturbations and
       * displaced internal density interfaces. Surface and CMB sheets are
       * deliberately excluded because they are owned by the corresponding
       * boundary-feedback plugins.
       */
      struct InternalMassMoments
      {
        Tensor<1,dim> mass_dipole;
        SymmetricTensor<2,dim> inertia_tensor;
      };

      /**
       * Compute and store a frozen initial lateral-average reference-density
       * profile when that reference model is selected.
       */
      void
      initialize_reference_density();

      /** Add material outputs needed by the selected density-source law. */
      void
      create_additional_material_model_outputs(
        MaterialModel::MaterialModelOutputs<dim> &outputs) const;

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

      /** Return the Cartesian gradient of the selected radial reference density. */
      Tensor<1,dim>
      reference_density_gradient(const Point<dim> &position) const;

      /** Return the finite elastic bulk modulus supplied by the material model. */
      double
      elastic_bulk_modulus(
        const MaterialModel::MaterialModelOutputs<dim> &outputs,
        const unsigned int q) const;

      /** Return the time interval represented by the current elastic solve. */
      double
      effective_mechanical_time_step() const;

      /**
       * Return the gravity magnitude used by the local mechanical volume
       * couplings. The default empty table returns @p gravity_model_magnitude.
       */
      double
      mechanical_gravity_magnitude(
        const Point<dim> &position,
        const double gravity_model_magnitude) const;

      /**
       * Return whether explicit or piecewise-constant tabulated internal
       * density jumps are active.
       */
      bool
      has_internal_density_jumps() const;

      /**
       * Return a radial representative point for @p cell whose radius is the
       * arithmetic mean of its vertex radii. Unlike `cell->center().norm()',
       * this radius is not biased inward by averaging Cartesian vertices on
       * a curved spherical cell.
       */
      template <typename CellIterator>
      Point<dim>
      radial_cell_representative_point(const CellIterator &cell) const
      {
        double radius = 0.0;
        for (unsigned int vertex = 0;
             vertex < GeometryInfo<dim>::vertices_per_cell;
             ++vertex)
          radius += cell->vertex(vertex).norm();
        radius /= GeometryInfo<dim>::vertices_per_cell;

        const Point<dim> center = cell->center();
        AssertThrow(radius > 0.0 && center.norm() > 0.0,
                    ExcMessage("A radial cell representative point is "
                               "undefined at radius zero."));
        return radius * center / center.norm();
      }

      /**
       * Return the configured density contrast at @p radius, or zero when no
       * interface matches within the configured face tolerance. The contrast
       * is density below minus density above the interface.
       */
      double
      internal_density_jump(const double radius) const;

      /**
       * Return the density contrast across an internal face. For a
       * piecewise-constant tabulated reference state, infer the contrast from
       * radial representative points for the adjacent inner and outer cells
       * so that material interfaces remain identified after mesh deformation.
       * Explicit jumps retain the radius-based query at @p face_radius.
       */
      double
      internal_density_jump_across_face(
        const Point<dim> &inner_cell_center,
        const Point<dim> &outer_cell_center,
        const double face_radius) const;

      /** Reset timestep-zero prediction bookkeeping before the first solve. */
      void
      begin_initial_mechanical_solve();

      /** Record that timestep-zero radial history includes the solved increment. */
      void
      mark_initial_mechanical_history_initialized();

      /** Return committed radial displacement plus the current trial increment. */
      double
      mechanical_radial_displacement(
        const MaterialModel::MaterialModelInputs<dim> &inputs,
        const unsigned int q) const;

      /** Return the selected non-legacy volume-density perturbation. */
      double
      density_perturbation(
        const MaterialModel::MaterialModelInputs<dim> &inputs,
        const MaterialModel::MaterialModelOutputs<dim> &outputs,
        const unsigned int q) const;

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

      /**
       * Resolve the `true|false|auto' internal-density selection shared by
       * self-gravity, center-of-mass, and rotational-feedback calculations.
       */
      bool
      internal_density_anomalies_are_enabled(
        const std::string &selection) const;

      /**
       * Visit each locally owned discrete internal mass source used by the
       * full-domain self-gravity calculation. The callback receives the
       * source mass and its Cartesian position. Surface and CMB sheets are
       * excluded; displaced internal density interfaces are included.
       *
       * Keeping this traversal in one place guarantees that low-order mass
       * moments and the radial Green-function potential use the same volume
       * projection selected by @p volume_source_discretization.
       */
      void
      for_each_internal_mass_source(
        const double legacy_reference_density,
        const std::string &volume_source_discretization,
        const std::function<void(const double,
                                 const Point<dim> &)> &consumer) const;

      /**
       * Integrate the mass dipole and inertia tensor of the selected internal
       * density source. For legacy density selection, @p legacy_reference_density
       * is subtracted from the material density. Non-legacy laws ignore it.
       * The volume-source discretization must match the full-domain potential
       * discretization when these moments feed a coupled reference-frame or
       * rotational-feedback calculation.
       */
      InternalMassMoments
      compute_internal_mass_moments(
        const double legacy_reference_density,
        const std::string &volume_source_discretization =
          "quadrature point") const;

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

      /** Return the lower interval index for a clamped radial table lookup. */
      unsigned int
      radial_table_interval(const double radius) const;

      /** Return whether a table interval encodes an explicit density jump. */
      bool
      radial_table_interval_contains_internal_density_jump(
        const unsigned int lower_interval_index) const;

      std::vector<double> depth_bounds;
      std::vector<double> depth_samples;
      std::vector<double> reference_density_values;
      Diagnostics initial_diagnostics;
      bool initialized = false;
      bool initial_mechanical_history_includes_current_solution = false;
      unsigned int initialization_count = 0;
      double reference_density_scale = 0.0;
  };
}

#endif
