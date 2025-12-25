/*
  Copyright (C) 2016 - 2024 by the authors of the ASPECT code.

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

#include <aspect/simulator.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/postprocess/visualization/surface_gravity_anomaly.h>



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      SurfaceGravityAnomaly<dim>::
      SurfaceGravityAnomaly ()
        :
        DataPostprocessorScalar<dim> ("surface_gravity_anomaly",
                                      update_quadrature_points),
        Interface<dim>("m/s/s")
      {}



      template <int dim>
      void
      SurfaceGravityAnomaly<dim>::
      initialize()
      {
        CitationInfo::add("geoid");
      }



      template <int dim>
      void
      SurfaceGravityAnomaly<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        AssertThrow (Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model()),
                     ExcMessage("The gravity anomaly postprocessor is currently only implemented for "
                                "the spherical shell geometry model."));

        for (auto &quantity : computed_quantities)
          quantity(0) = 0;

        const Postprocess::Geoid<dim> &geoid =
          this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::Geoid<dim>>();

        auto cell = input_data.template get_cell<dim>();

        bool cell_at_top_boundary = false;
        for (const unsigned int f : cell->face_indices())
          if (cell->at_boundary(f) &&
              this->get_geometry_model().translate_id_to_symbol_name (cell->face(f)->boundary_id()) == "top")
            cell_at_top_boundary = true;

        if (cell_at_top_boundary)
          for (unsigned int q=0; q<input_data.evaluation_points.size(); ++q)
            // Convert from m/s² to mGal (1 mGal = 10^-5 m/s²)
            computed_quantities[q](0) = geoid.evaluate_gravity_anomaly(input_data.evaluation_points[q]);
      }

      template <int dim>
      std::list<std::string>
      SurfaceGravityAnomaly<dim>::required_other_postprocessors() const
      {
        return {"geoid"};
      }

    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(SurfaceGravityAnomaly,
                                                  "surface gravity anomaly",
                                                  "Visualization for the surface gravity anomaly. The gravity anomaly "
                                                  "is the deviation of the gravity field from a reference gravity field. "
                                                  "This postprocessor computes the free-air gravity anomaly at the surface "
                                                  "using spherical harmonic expansion. The gravity anomaly is physically "
                                                  "meaningful primarily at the surface."
                                                  "\n\n"
                                                  "Physical units: $\\text{mGal}$.")
    }
  }
}
