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

namespace aspect
{
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
  DensitySourceManager<dim>::reference_density(const Point<dim> &/*position*/) const
  {
    return 0.0;
  }



  template <int dim>
  double
  DensitySourceManager<dim>::stokes_source_density(
    const MaterialModel::MaterialModelInputs<dim> &/*inputs*/,
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q) const
  {
    return physical_density(outputs, q);
  }



  template <int dim>
  double
  DensitySourceManager<dim>::self_gravity_source_density(
    const MaterialModel::MaterialModelInputs<dim> &/*inputs*/,
    const MaterialModel::MaterialModelOutputs<dim> &outputs,
    const unsigned int q,
    const double legacy_reference_density) const
  {
    return physical_density(outputs, q) - legacy_reference_density;
  }
}


namespace aspect
{
#define INSTANTIATE(dim) \
  template class DensitySourceManager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
