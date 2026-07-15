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


#include <aspect/density_source_manager.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/lateral_averaging.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>

namespace aspect
{
  template <int dim>
  void
  DensitySourceManager<dim>::initialize_reference_density()
  {
    if (this->get_parameters().reference_density_model
        != Parameters<dim>::Formulation::ReferenceDensityModel::frozen_initial_lateral_average)
      return;

    AssertThrow(!initialized && initialization_count == 0,
                ExcMessage("The frozen initial reference-density profile was requested to be "
                           "initialized more than once. It must remain fixed after its initial "
                           "construction."));

    const auto density_dependence =
      this->get_material_model().get_model_dependence().density;
    if ((static_cast<int>(density_dependence)
         & static_cast<int>(MaterialModel::NonlinearDependence::pressure)) != 0)
      AssertThrow(false,
                  ExcMessage("The frozen initial lateral-average reference-density model does "
                             "not yet support pressure-dependent material density. A future "
                             "implementation needs an explicit thermodynamic pressure policy; "
                             "this implementation does not use raw Stokes pressure in the "
                             "reference state."));

    const unsigned int n_slices =
      this->get_parameters().frozen_reference_density_profile_slices;
    const double maximal_depth = this->get_geometry_model().maximal_depth();

    AssertThrow(n_slices > 0,
                ExcMessage("The frozen reference-density profile needs at least one depth slice."));
    AssertThrow(maximal_depth > 0.0,
                ExcMessage("The frozen reference-density profile requires a positive maximal depth."));

    depth_bounds.assign(n_slices + 1, 0.0);
    for (unsigned int i = 1; i < depth_bounds.size(); ++i)
      depth_bounds[i] = maximal_depth * static_cast<double>(i) / static_cast<double>(n_slices);

    depth_samples.assign(n_slices, 0.0);
    for (unsigned int i = 0; i < n_slices; ++i)
      depth_samples[i] = 0.5 * (depth_bounds[i] + depth_bounds[i+1]);

    const std::vector<std::vector<double>> averages =
      this->get_lateral_averaging().compute_lateral_averages(depth_bounds,
    {"density"});
    AssertThrow(averages.size() == 1 && averages[0].size() == n_slices,
                ExcInternalError());

    reference_density_values = averages[0];
    reference_density_scale = 0.0;
    for (const double value : reference_density_values)
      {
        AssertThrow(std::isfinite(value),
                    ExcMessage("The frozen initial reference-density profile contains a non-finite "
                               "value. Reduce the number of depth slices or inspect the material "
                               "model density evaluation."));
        reference_density_scale = std::max(reference_density_scale, std::abs(value));
      }

    initialized = true;
    ++initialization_count;
    initial_diagnostics = compute_initial_diagnostics();

    this->get_pcout() << "Frozen initial reference-density profile: "
                      << n_slices << " geometric-depth bins, "
                      << "integral(rho-rho_ref)dV="
                      << initial_diagnostics.integrated_mass << ", "
                      << "||rho-rho_ref||_L2="
                      << initial_diagnostics.l2_norm << ", "
                      << "max|rho-rho_ref|="
                      << initial_diagnostics.max_abs << ", "
                      << "max depth-bin lateral mean residual="
                      << initial_diagnostics.max_lateral_average_residual
                      << std::endl;
  }



  template <int dim>
  void
  DensitySourceManager<dim>::create_additional_material_model_outputs(
    MaterialModel::MaterialModelOutputs<dim> &outputs) const
  {
    if (this->get_parameters().density_source_law
        == Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation
        && outputs.template has_additional_output_object<MaterialModel::ElasticOutputs<dim>>() == false)
      outputs.additional_outputs.push_back(
        std::make_unique<MaterialModel::ElasticOutputs<dim>>(
          outputs.n_evaluation_points()));
  }



  template <int dim>
  double
  DensitySourceManager<dim>::physical_density(
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q) const
  {
    AssertIndexRange(q, outputs.densities.size());
    return outputs.densities[q];
  }



  template <int dim>
  double
  DensitySourceManager<dim>::reference_density(const Point<dim> &position) const
  {
    const auto model = this->get_parameters().reference_density_model;

    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::none)
      return 0.0;

    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::constant)
      return this->get_parameters().constant_reference_density;

    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::tabulated_radial)
      {
        AssertThrow(this->get_geometry_model().natural_coordinate_system()
                    == Utilities::Coordinates::CoordinateSystem::spherical,
                    ExcMessage("The tabulated radial reference-density model requires a spherical geometry."));

        const auto &radii = this->get_parameters().tabulated_reference_radii;
        const auto &densities = this->get_parameters().tabulated_reference_densities;
        const double radius = position.norm();

        if (radius <= radii.front())
          return densities.front();
        if (radius >= radii.back())
          return densities.back();

        if (this->get_parameters().tabulated_reference_density_interpolation
            == Parameters<dim>::Formulation::TabulatedReferenceDensityInterpolation::piecewise_constant)
          return densities[radial_table_interval(radius)];

        const auto upper = std::upper_bound(radii.begin(), radii.end(), radius);
        const unsigned int lower_index =
          static_cast<unsigned int>(std::distance(radii.begin(), upper)) - 1;
        AssertIndexRange(lower_index, radii.size() - 1);

        const double weight =
          (radius - radii[lower_index])
          / (radii[lower_index+1] - radii[lower_index]);
        return (1.0 - weight) * densities[lower_index]
               + weight * densities[lower_index+1];
      }

    AssertThrow(model == Parameters<dim>::Formulation::ReferenceDensityModel::frozen_initial_lateral_average,
                ExcNotImplemented());
    AssertThrow(initialized,
                ExcMessage("The frozen initial reference-density profile was queried before initialization."));
    AssertThrow(reference_density_values.size() == depth_samples.size()
                && !reference_density_values.empty(),
                ExcInternalError());

    if (reference_density_values.size() == 1)
      return reference_density_values.front();

    const double depth = this->get_geometry_model().depth(position);
    if (depth <= depth_samples.front())
      return reference_density_values.front();
    if (depth >= depth_samples.back())
      return reference_density_values.back();

    const auto upper = std::upper_bound(depth_samples.begin(),
                                        depth_samples.end(),
                                        depth);
    const unsigned int upper_index =
      static_cast<unsigned int>(std::distance(depth_samples.begin(), upper));
    const unsigned int lower_index = upper_index - 1;
    const double denominator =
      depth_samples[upper_index] - depth_samples[lower_index];
    Assert(denominator > 0.0, ExcInternalError());

    const double weight =
      (depth - depth_samples[lower_index]) / denominator;
    return (1.0 - weight) * reference_density_values[lower_index]
           + weight * reference_density_values[upper_index];
  }



  template <int dim>
  Tensor<1,dim>
  DensitySourceManager<dim>::reference_density_gradient(const Point<dim> &position) const
  {
    const auto model = this->get_parameters().reference_density_model;
    Tensor<1,dim> gradient;

    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::none
        || model == Parameters<dim>::Formulation::ReferenceDensityModel::constant)
      return gradient;

    AssertThrow(model == Parameters<dim>::Formulation::ReferenceDensityModel::tabulated_radial,
                ExcMessage("Mechanical reference-density gradients currently require a constant or tabulated radial model."));
    AssertThrow(this->get_geometry_model().natural_coordinate_system()
                == Utilities::Coordinates::CoordinateSystem::spherical,
                ExcMessage("The tabulated radial reference-density gradient requires spherical geometry."));

    const double radius = position.norm();
    AssertThrow(radius > 0.0,
                ExcMessage("A radial reference-density gradient is undefined at radius zero."));

    const auto &radii = this->get_parameters().tabulated_reference_radii;
    const auto &densities = this->get_parameters().tabulated_reference_densities;
    double radial_derivative = 0.0;
    if (this->get_parameters().tabulated_reference_density_interpolation
        == Parameters<dim>::Formulation::TabulatedReferenceDensityInterpolation::linear
        && radius >= radii.front() && radius <= radii.back())
      {
        const unsigned int lower = radial_table_interval(radius);
        if (!radial_table_interval_contains_internal_density_jump(lower))
          radial_derivative = (densities[lower+1] - densities[lower])
                              / (radii[lower+1] - radii[lower]);
      }

    gradient = radial_derivative * position / radius;
    return gradient;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::elastic_bulk_modulus(
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q) const
  {
    const std::shared_ptr<const MaterialModel::ElasticOutputs<dim>> elastic_outputs =
      outputs.template get_additional_output_object<MaterialModel::ElasticOutputs<dim>>();
    AssertThrow(elastic_outputs != nullptr,
                ExcMessage("Mechanical mass conservation requires elastic bulk-modulus material outputs."));
    AssertIndexRange(q, elastic_outputs->elastic_bulk_moduli.size());
    AssertThrow(std::isfinite(elastic_outputs->elastic_bulk_moduli[q])
                && elastic_outputs->elastic_bulk_moduli[q] > 0.0,
                ExcMessage("The elastic bulk modulus must be finite and positive."));
    return elastic_outputs->elastic_bulk_moduli[q];
  }



  template <int dim>
  double
  DensitySourceManager<dim>::effective_mechanical_time_step() const
  {
    double time_step = this->get_timestep();
    if (this->get_timestep_number() == 0 && time_step == 0.0)
      {
        time_step = this->get_parameters().initial_elastic_response_time_step;
        if (this->get_material_model().initial_elastic_time_step() > 0.0)
          time_step = this->get_material_model().initial_elastic_time_step();
      }
    AssertThrow(std::isfinite(time_step) && time_step > 0.0,
                ExcMessage("Mechanical mass conservation requires a finite positive elastic time step."));
    return time_step;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::mechanical_gravity_magnitude(
    const Point<dim> &position,
    const double gravity_model_magnitude) const
  {
    AssertThrow(std::isfinite(gravity_model_magnitude)
                && gravity_model_magnitude >= 0.0,
                ExcMessage("The gravity-model magnitude must be finite and non-negative."));

    const auto &magnitudes =
      this->get_parameters().tabulated_mechanical_gravity_magnitudes;
    if (magnitudes.empty())
      return gravity_model_magnitude;

    const double radius = position.norm();
    AssertThrow(std::isfinite(radius) && radius > 0.0,
                ExcMessage("Tabulated mechanical gravity is undefined at radius zero."));
    AssertThrow(magnitudes.size() + 1
                == this->get_parameters().tabulated_reference_radii.size(),
                ExcInternalError());

    return magnitudes[radial_table_interval(radius)];
  }



  template <int dim>
  bool
  DensitySourceManager<dim>::has_internal_density_jumps() const
  {
    const auto &parameters = this->get_parameters();
    if (!parameters.internal_density_jump_radii.empty())
      return true;

    if (parameters.reference_density_model
        == Parameters<dim>::Formulation::ReferenceDensityModel::tabulated_radial
        && parameters.tabulated_reference_density_interpolation
        == Parameters<dim>::Formulation::TabulatedReferenceDensityInterpolation::piecewise_constant)
      for (unsigned int i = 1;
           i + 1 < parameters.tabulated_reference_densities.size();
           ++i)
        if (parameters.tabulated_reference_densities[i-1]
            != parameters.tabulated_reference_densities[i])
          return true;

    return false;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::internal_density_jump(const double radius) const
  {
    AssertThrow(std::isfinite(radius) && radius > 0.0,
                ExcMessage("Internal density jump matching requires a finite positive radius."));

    if (this->get_parameters().tabulated_reference_density_interpolation
        == Parameters<dim>::Formulation::TabulatedReferenceDensityInterpolation::piecewise_constant)
      {
        const auto &radii = this->get_parameters().tabulated_reference_radii;
        const auto &densities = this->get_parameters().tabulated_reference_densities;
        const double tolerance =
          this->get_parameters().internal_density_jump_face_tolerance;

        AssertThrow(radii.size() == densities.size(), ExcInternalError());
        for (unsigned int i = 1; i + 1 < radii.size(); ++i)
          if (std::abs(radius - radii[i]) <= tolerance)
            return densities[i-1] - densities[i];

        return 0.0;
      }

    const auto &radii = this->get_parameters().internal_density_jump_radii;
    const auto &contrasts =
      this->get_parameters().internal_density_jump_density_contrasts;
    const double tolerance =
      this->get_parameters().internal_density_jump_face_tolerance;

    AssertThrow(radii.size() == contrasts.size(), ExcInternalError());

    for (unsigned int i = 0; i < radii.size(); ++i)
      if (std::abs(radius - radii[i]) <= tolerance)
        return contrasts[i];

    return 0.0;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::internal_density_jump_across_face(
    const Point<dim> &inner_cell_center,
    const Point<dim> &outer_cell_center,
    const double face_radius) const
  {
    if (this->get_parameters().tabulated_reference_density_interpolation
        == Parameters<dim>::Formulation::TabulatedReferenceDensityInterpolation::piecewise_constant)
      {
        AssertThrow(inner_cell_center.norm() > 0.0
                    && outer_cell_center.norm() > inner_cell_center.norm(),
                    ExcMessage("Internal density jump faces require radially ordered cell centers."));
        return reference_density(inner_cell_center)
               - reference_density(outer_cell_center);
      }

    return internal_density_jump(face_radius);
  }



  template <int dim>
  void
  DensitySourceManager<dim>::begin_initial_mechanical_solve()
  {
    initial_mechanical_history_includes_current_solution = false;
  }



  template <int dim>
  void
  DensitySourceManager<dim>::mark_initial_mechanical_history_initialized()
  {
    initial_mechanical_history_includes_current_solution = true;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::mechanical_radial_displacement(
    const MaterialModel::MaterialModelInputs<dim> &inputs,
    const unsigned int q) const
  {
    AssertIndexRange(q, inputs.position.size());
    AssertIndexRange(q, inputs.velocity.size());
    AssertIndexRange(q, inputs.composition.size());

    const unsigned int radial_displacement_index =
      this->introspection().compositional_index_for_name("ve_radial_displacement");
    AssertIndexRange(radial_displacement_index, inputs.composition[q].size());

    const double radius = inputs.position[q].norm();
    AssertThrow(radius > 0.0,
                ExcMessage("The radial material-displacement history is undefined at radius zero."));
    const Tensor<1,dim> radial_unit = inputs.position[q] / radius;
    const double committed_displacement =
      inputs.composition[q][radial_displacement_index];

    if (this->get_timestep_number() == 0
        && initial_mechanical_history_includes_current_solution)
      return committed_displacement;

    return committed_displacement
           + effective_mechanical_time_step()
           * (inputs.velocity[q] * radial_unit);
  }



  template <int dim>
  double
  DensitySourceManager<dim>::density_perturbation(
    const MaterialModel::MaterialModelInputs<dim> &inputs,
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q) const
  {
    const auto law = this->get_parameters().density_source_law;
    AssertIndexRange(q, inputs.position.size());

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::material_density)
      return physical_density(outputs, q);
    if (law == Parameters<dim>::Formulation::DensitySourceLaw::material_minus_reference)
      return physical_density(outputs, q) - reference_density(inputs.position[q]);
    if (law == Parameters<dim>::Formulation::DensitySourceLaw::zero_volume_perturbation)
      return 0.0;

    AssertThrow(law == Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation,
                ExcNotImplemented());
    AssertIndexRange(q, inputs.pressure.size());

    const double radius = inputs.position[q].norm();
    AssertThrow(radius > 0.0,
                ExcMessage("The radial material-displacement history is undefined at radius zero."));
    const Tensor<1,dim> radial_unit = inputs.position[q] / radius;
    const double radial_density_gradient =
      reference_density_gradient(inputs.position[q]) * radial_unit;

    return
      reference_density(inputs.position[q])
      * inputs.pressure[q]
      / elastic_bulk_modulus(outputs, q)
      - mechanical_radial_displacement(inputs, q)
      * radial_density_gradient;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::stokes_source_density(
    const MaterialModel::MaterialModelInputs<dim> &inputs,
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q) const
  {
    const auto law = this->get_parameters().density_source_law;

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::legacy
        || law == Parameters<dim>::Formulation::DensitySourceLaw::material_density)
      return physical_density(outputs, q);

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::zero_volume_perturbation)
      return 0.0;

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation)
      return 0.0;

    return density_perturbation(inputs, outputs, q);
  }



  template <int dim>
  double
  DensitySourceManager<dim>::self_gravity_source_density(
    const MaterialModel::MaterialModelInputs<dim> &inputs,
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q,
    const double legacy_reference_density) const
  {
    const auto law = this->get_parameters().density_source_law;

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::legacy)
      return physical_density(outputs, q) - legacy_reference_density;

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::material_density)
      return physical_density(outputs, q);

    if (law == Parameters<dim>::Formulation::DensitySourceLaw::zero_volume_perturbation)
      return 0.0;

    return density_perturbation(inputs, outputs, q);
  }



  template <int dim>
  bool
  DensitySourceManager<dim>::has_initialized_reference_density() const
  {
    return initialized;
  }



  template <int dim>
  const std::vector<double> &
  DensitySourceManager<dim>::get_depth_samples() const
  {
    return depth_samples;
  }



  template <int dim>
  const std::vector<double> &
  DensitySourceManager<dim>::get_reference_density_values() const
  {
    return reference_density_values;
  }



  template <int dim>
  const typename DensitySourceManager<dim>::Diagnostics &
  DensitySourceManager<dim>::get_initial_diagnostics() const
  {
    return initial_diagnostics;
  }



  template <int dim>
  unsigned int
  DensitySourceManager<dim>::get_initialization_count() const
  {
    return initialization_count;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::get_reference_density_scale() const
  {
    const auto model = this->get_parameters().reference_density_model;
    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::none)
      return 0.0;
    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::constant)
      return std::abs(this->get_parameters().constant_reference_density);
    if (model == Parameters<dim>::Formulation::ReferenceDensityModel::tabulated_radial)
      {
        double scale = 0.0;
        for (const double density : this->get_parameters().tabulated_reference_densities)
          scale = std::max(scale, std::abs(density));
        return scale;
      }

    AssertThrow(initialized,
                ExcMessage("The frozen reference-density scale was queried before initialization."));
    return reference_density_scale;
  }



  template <int dim>
  typename DensitySourceManager<dim>::Diagnostics
  DensitySourceManager<dim>::compute_initial_diagnostics() const
  {
    AssertThrow(initialized, ExcInternalError());

    Diagnostics local_diagnostics;
    std::vector<double> local_bin_mass(depth_bounds.size() - 1, 0.0);
    std::vector<double> local_bin_volume(depth_bounds.size() - 1, 0.0);

    const unsigned int quadrature_degree =
      std::max(1u, this->introspection().polynomial_degree.temperature);
    const QGauss<dim> quadrature_formula(quadrature_degree);
    FEValues<dim> fe_values(this->get_mapping(),
                            this->get_fe(),
                            quadrature_formula,
                            update_values |
                            update_quadrature_points |
                            update_JxW_values |
                            update_gradients);

    MaterialModel::MaterialModelInputs<dim>
    inputs(fe_values.n_quadrature_points, this->n_compositional_fields());
    MaterialModel::MaterialModelOutputs<dim>
    outputs(fe_values.n_quadrature_points, this->n_compositional_fields());
    inputs.requested_properties = MaterialModel::MaterialProperties::density;

    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          inputs.reinit(fe_values,
                        cell,
                        this->introspection(),
                        this->get_solution());
          this->get_material_model().evaluate(inputs, outputs);

          for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
            {
              const double anomaly =
                physical_density(outputs, q) - reference_density(inputs.position[q]);
              const double JxW = fe_values.JxW(q);

              local_diagnostics.integrated_mass += anomaly * JxW;
              local_diagnostics.l2_norm += anomaly * anomaly * JxW;
              local_diagnostics.max_abs =
                std::max(local_diagnostics.max_abs, std::abs(anomaly));

              const unsigned int bin =
                depth_bin_index(this->get_geometry_model().depth(inputs.position[q]));
              local_bin_mass[bin] += anomaly * JxW;
              local_bin_volume[bin] += JxW;
            }
        }

    Diagnostics global_diagnostics;
    global_diagnostics.integrated_mass =
      Utilities::MPI::sum(local_diagnostics.integrated_mass,
                          this->get_mpi_communicator());
    global_diagnostics.l2_norm =
      std::sqrt(Utilities::MPI::sum(local_diagnostics.l2_norm,
                                    this->get_mpi_communicator()));
    global_diagnostics.max_abs =
      Utilities::MPI::max(local_diagnostics.max_abs,
                          this->get_mpi_communicator());

    std::vector<double> global_bin_mass(local_bin_mass.size());
    std::vector<double> global_bin_volume(local_bin_volume.size());
    Utilities::MPI::sum(local_bin_mass,
                        this->get_mpi_communicator(),
                        global_bin_mass);
    Utilities::MPI::sum(local_bin_volume,
                        this->get_mpi_communicator(),
                        global_bin_volume);

    for (unsigned int i = 0; i < global_bin_mass.size(); ++i)
      if (global_bin_volume[i] > 0.0)
        global_diagnostics.max_lateral_average_residual =
          std::max(global_diagnostics.max_lateral_average_residual,
                   std::abs(global_bin_mass[i] / global_bin_volume[i]));

    return global_diagnostics;
  }



  template <int dim>
  unsigned int
  DensitySourceManager<dim>::depth_bin_index(const double depth) const
  {
    AssertThrow(depth_bounds.size() > 1, ExcInternalError());

    if (depth <= depth_bounds.front())
      return 0;
    if (depth >= depth_bounds.back())
      return static_cast<unsigned int>(depth_bounds.size() - 2);

    unsigned int layer_index =
      static_cast<unsigned int>(std::distance(depth_bounds.begin(),
                                              std::lower_bound(depth_bounds.begin(),
                                                               depth_bounds.end(),
                                                               depth)));
    if (layer_index > 0)
      --layer_index;

    AssertIndexRange(layer_index, depth_bounds.size() - 1);
    return layer_index;
  }



  template <int dim>
  unsigned int
  DensitySourceManager<dim>::radial_table_interval(const double radius) const
  {
    const auto &radii = this->get_parameters().tabulated_reference_radii;
    AssertThrow(radii.size() >= 2, ExcInternalError());

    if (radius <= radii.front())
      return 0;
    if (radius >= radii.back())
      return static_cast<unsigned int>(radii.size() - 2);

    const auto upper = std::upper_bound(radii.begin(), radii.end(), radius);
    return static_cast<unsigned int>(std::distance(radii.begin(), upper)) - 1;
  }



  template <int dim>
  bool
  DensitySourceManager<dim>::radial_table_interval_contains_internal_density_jump(
    const unsigned int lower_interval_index) const
  {
    const auto &table_radii = this->get_parameters().tabulated_reference_radii;
    AssertIndexRange(lower_interval_index, table_radii.size() - 1);

    const double lower_radius = table_radii[lower_interval_index];
    const double upper_radius = table_radii[lower_interval_index+1];
    for (const double jump_radius :
         this->get_parameters().internal_density_jump_radii)
      if (jump_radius > lower_radius && jump_radius < upper_radius)
        return true;

    return false;
  }
}


namespace aspect
{
#define INSTANTIATE(dim) \
  template class DensitySourceManager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
