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

#include "common.h"

#include <aspect/potential_feedback/self_gravitation.h>


TEST_CASE("Cumulative radial Green moments match direct summation",
          "[potential_feedback][self_gravitation]")
{
  const std::vector<double> evaluation_radii = {3.48e6, 4.9e6, 6.37e6};
  const std::vector<double> source_radii =
  {3.0e6, 3.48e6, 4.1e6, 5.6e6, 6.37e6, 6.5e6};
  const unsigned int minimum_degree = 1;
  const unsigned int maximum_degree = 32;
  const double reference_radius = 6.37e6;

  aspect::PotentialFeedback::internal::RadialGreenMomentAccumulator
  accumulator(evaluation_radii,
              minimum_degree,
              maximum_degree,
              reference_radius);

  std::vector<unsigned int> coefficient_degrees;
  for (unsigned int degree = minimum_degree;
       degree <= maximum_degree;
       ++degree)
    for (unsigned int order = 0; order <= degree; ++order)
      coefficient_degrees.push_back(degree);

  std::vector<std::vector<double>> source_cos_coefficients;
  std::vector<std::vector<double>> source_sin_coefficients;
  for (unsigned int source_index = 0;
       source_index < source_radii.size();
       ++source_index)
    {
      std::vector<double> cos_coefficients(coefficient_degrees.size());
      std::vector<double> sin_coefficients(coefficient_degrees.size());
      for (unsigned int coefficient_index = 0;
           coefficient_index < coefficient_degrees.size();
           ++coefficient_index)
        {
          cos_coefficients[coefficient_index] =
            0.1 * (source_index + 1) * (coefficient_index + 1);
          sin_coefficients[coefficient_index] =
            -0.07 * (source_index + 2) * (coefficient_index + 1);
        }
      source_cos_coefficients.push_back(cos_coefficients);
      source_sin_coefficients.push_back(sin_coefficients);
      accumulator.add_source(source_radii[source_index],
                             cos_coefficients,
                             sin_coefficients);
    }

  accumulator.mpi_sum(MPI_COMM_SELF);
  const std::pair<std::vector<double>, std::vector<double>> values =
    accumulator.evaluate();

  for (unsigned int radius_index = 0;
       radius_index < evaluation_radii.size();
       ++radius_index)
    for (unsigned int coefficient_index = 0;
         coefficient_index < coefficient_degrees.size();
         ++coefficient_index)
      {
        const double evaluation_radius = evaluation_radii[radius_index];
        const unsigned int degree = coefficient_degrees[coefficient_index];
        double expected_cos = 0.0;
        double expected_sin = 0.0;
        for (unsigned int source_index = 0;
             source_index < source_radii.size();
             ++source_index)
          {
            const double source_radius = source_radii[source_index];
            const double radial_kernel =
              (source_radius <= evaluation_radius
               ? (1.0 / source_radius)
               * std::pow(source_radius / evaluation_radius, degree + 1)
               : (1.0 / source_radius)
               * std::pow(evaluation_radius / source_radius, degree));
            expected_cos +=
              source_cos_coefficients[source_index][coefficient_index]
              * radial_kernel;
            expected_sin +=
              source_sin_coefficients[source_index][coefficient_index]
              * radial_kernel;
          }

        const unsigned int value_index =
          radius_index * coefficient_degrees.size() + coefficient_index;
        REQUIRE(values.first[value_index]
                == Approx(expected_cos).epsilon(1e-12));
        REQUIRE(values.second[value_index]
                == Approx(expected_sin).epsilon(1e-12));
      }
}
