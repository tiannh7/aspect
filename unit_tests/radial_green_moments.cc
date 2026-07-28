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



namespace
{
  struct RadialQuadraturePoints
  {
    std::vector<double> radii;
    std::vector<double> weights;
  };

  RadialQuadraturePoints make_radial_quadrature(const std::vector<double> &boundaries,
                                                const unsigned int order)
  {
    const dealii::QGauss<1> quadrature(order);
    RadialQuadraturePoints points;
    for (unsigned int cell = 0; cell + 1 < boundaries.size(); ++cell)
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          const double left = boundaries[cell];
          const double right = boundaries[cell + 1];
          const double radius = left + (right-left) * quadrature.point(q)[0];
          points.radii.push_back(radius);
          points.weights.push_back(quadrature.weight(q) * (right-left) * radius*radius);
        }
    return points;
  }

  double kernel(const unsigned int l, const double r, const double a)
  {
    return std::pow(std::min(r,a),l) / std::pow(std::max(r,a),l+1);
  }

  std::vector<double> default_cache_supports(
    const std::vector<double> &boundaries)
  {
    std::vector<double> uniform_supports;
    for (unsigned int i = 0; i <= 32; ++i)
      uniform_supports.push_back(
        boundaries.front()
        + (boundaries.back() - boundaries.front()) * i / 32.0);
    return aspect::PotentialFeedback::internal::merge_radial_cache_supports(
    {boundaries, uniform_supports}, 1000);
  }

  std::vector<std::vector<double>> make_operator(
    const std::vector<double> &targets,
    const std::vector<double> &sources,
    const std::vector<double> &supports,
    const unsigned int degree,
    const bool exact_lookup)
  {
    std::vector<std::vector<double>> matrix(
      targets.size(), std::vector<double>(sources.size()));
    for (unsigned int source_index = 0;
         source_index < sources.size();
         ++source_index)
      {
        std::vector<std::vector<double>> values(
          supports.size(), std::vector<double>(1));
        for (unsigned int support_index = 0;
             support_index < supports.size();
             ++support_index)
          values[support_index][0] =
            kernel(degree, supports[support_index], sources[source_index]);
        for (unsigned int target_index = 0;
             target_index < targets.size();
             ++target_index)
          matrix[target_index][source_index] =
            (exact_lookup
             ? aspect::PotentialFeedback::internal::lookup_radial_cache_exact(
               supports, values, 0, targets[target_index])
             : aspect::PotentialFeedback::internal::interpolate_radial_cache(
               supports, values, 0, targets[target_index]));
      }
    return matrix;
  }

  std::vector<std::vector<double>> make_symmetric_support_operator(
    const std::vector<double> &targets,
    const std::vector<double> &sources,
    const std::vector<double> &supports,
    const unsigned int degree)
  {
    std::vector<std::vector<double>> matrix(
      targets.size(), std::vector<double>(sources.size(), 0.0));
    for (unsigned int target = 0; target < targets.size(); ++target)
      {
        const auto target_stencil =
          aspect::PotentialFeedback::internal::radial_interpolation_stencil(
            supports, targets[target]);
        for (unsigned int source = 0; source < sources.size(); ++source)
          {
            const auto source_stencil =
              aspect::PotentialFeedback::internal::radial_interpolation_stencil(
                supports, sources[source]);
            for (unsigned int i = 0; i < 2; ++i)
              for (unsigned int j = 0; j < 2; ++j)
                matrix[target][source] +=
                  target_stencil.weights[i] * source_stencil.weights[j]
                  * kernel(degree,
                           supports[target_stencil.indices[i]],
                           supports[source_stencil.indices[j]]);
          }
      }
    return matrix;
  }

  std::vector<std::vector<double>> make_accumulator_support_operator(
    const std::vector<double> &targets,
    const std::vector<double> &sources,
    const std::vector<double> &supports,
    const unsigned int degree)
  {
    std::vector<std::vector<double>> matrix(
      targets.size(), std::vector<double>(sources.size(), 0.0));
    const unsigned int n_coefficients = degree + 1;
    for (unsigned int source = 0; source < sources.size(); ++source)
      {
        aspect::PotentialFeedback::internal::RadialGreenMomentAccumulator
        accumulator(supports, degree, degree, supports.back());
        std::vector<double> coefficients(n_coefficients, 0.0);
        coefficients[0] = 1.0;
        accumulator.add_interpolated_source(sources[source],
                                            coefficients,
                                            coefficients);
        accumulator.mpi_sum(MPI_COMM_SELF);
        const auto support_values = accumulator.evaluate().first;
        std::vector<std::vector<double>> values(
          supports.size(), std::vector<double>(1));
        for (unsigned int support = 0; support < supports.size(); ++support)
          values[support][0] =
            support_values[support * n_coefficients];
        for (unsigned int target = 0; target < targets.size(); ++target)
          matrix[target][source] =
            aspect::PotentialFeedback::internal::interpolate_radial_cache(
              supports, values, 0, targets[target]);
      }
    return matrix;
  }

  double operator_residual(const std::vector<std::vector<double>> &forward,
                           const std::vector<std::vector<double>> &reverse,
                           const std::vector<double> &wt,
                           const std::vector<double> &ws)
  {
    double residual=0.0, scale=0.0;
    for (unsigned int i=0; i<forward.size(); ++i)
      for (unsigned int j=0; j<forward[i].size(); ++j)
        {
          const double f=wt[i]*forward[i][j]*ws[j];
          const double r=wt[i]*reverse[j][i]*ws[j];
          residual += (f-r)*(f-r);
          scale += f*f+r*r;
        }
    return std::sqrt(residual/scale);
  }

  double vector_residual(const std::vector<std::vector<double>> &forward,
                         const std::vector<std::vector<double>> &reverse,
                         const std::vector<double> &wt,
                         const std::vector<double> &ws,
                         const unsigned int variant)
  {
    double f=0.0, r=0.0;
    for (unsigned int i=0; i<forward.size(); ++i)
      for (unsigned int j=0; j<forward[i].size(); ++j)
        {
          const double x = variant==0 ? 1.0+0.07*i : std::sin(0.31*(i+1));
          const double y = variant==0 ? 0.8-0.03*j : std::cos(0.23*(j+1));
          f += wt[i]*x*forward[i][j]*ws[j]*y;
          r += ws[j]*y*reverse[j][i]*wt[i]*x;
        }
    return std::abs(f-r)/std::max(std::abs(f),std::abs(r));
  }
}


TEST_CASE("Target-enriched radial cache restores weighted reciprocity",
          "[potential_feedback][self_gravitation]")
{
  const std::vector<double> boundaries =
  {
    3485497.648, 5201000., 5701000., 5961000., 6116000.,
    6271000., 6311000., 6341000., 6371000.
  };
  const auto sources = make_radial_quadrature(boundaries, 2);
  const auto targets = make_radial_quadrature(boundaries, 3);
  const std::vector<double> default_supports =
    default_cache_supports(boundaries);
  const std::vector<double> enriched_supports =
    aspect::PotentialFeedback::internal::merge_radial_cache_supports(
  {default_supports, sources.radii, targets.radii}, 1000);

  for (const unsigned int degree :
  {
    2u, 4u, 6u, 8u
  })
  {
    const auto default_forward =
      make_operator(targets.radii,
                    sources.radii,
                    default_supports,
                    degree,
                    false);
    const auto default_reverse =
      make_operator(sources.radii,
                    targets.radii,
                    default_supports,
                    degree,
                    false);
    const auto enriched_forward =
      make_operator(targets.radii,
                    sources.radii,
                    enriched_supports,
                    degree,
                    true);
    const auto enriched_reverse =
      make_operator(sources.radii,
                    targets.radii,
                    enriched_supports,
                    degree,
                    true);
    const auto symmetric_forward =
      make_symmetric_support_operator(targets.radii,
                                      sources.radii,
                                      default_supports,
                                      degree);
    const auto symmetric_reverse =
      make_symmetric_support_operator(sources.radii,
                                      targets.radii,
                                      default_supports,
                                      degree);
    const auto accumulator_forward =
      make_accumulator_support_operator(targets.radii,
                                        sources.radii,
                                        default_supports,
                                        degree);

    INFO("degree=" << degree);
    const double default_residual =
      operator_residual(default_forward,
                        default_reverse,
                        targets.weights,
                        sources.weights);
    INFO("default linear-cache weighted residual=" << default_residual);
    CHECK(default_residual > 1e-8);

    CHECK(operator_residual(enriched_forward,
                            enriched_reverse,
                            targets.weights,
                            sources.weights) < 5e-14);
    CHECK(vector_residual(enriched_forward,
                          enriched_reverse,
                          targets.weights,
                          sources.weights,
                          0) < 5e-14);
    CHECK(vector_residual(enriched_forward,
                          enriched_reverse,
                          targets.weights,
                          sources.weights,
                          1) < 5e-14);

    CHECK(operator_residual(symmetric_forward,
                            symmetric_reverse,
                            targets.weights,
                            sources.weights) < 5e-14);
    CHECK(vector_residual(symmetric_forward,
                          symmetric_reverse,
                          targets.weights,
                          sources.weights,
                          0) < 5e-14);
    CHECK(vector_residual(symmetric_forward,
                          symmetric_reverse,
                          targets.weights,
                          sources.weights,
                          1) < 5e-14);
    for (unsigned int target = 0; target < targets.radii.size(); ++target)
      for (unsigned int source = 0; source < sources.radii.size(); ++source)
        CHECK(accumulator_forward[target][source]
              == Approx(symmetric_forward[target][source]).epsilon(2e-13));
  }
}


TEST_CASE("Radial interpolation stencils conserve moments and clamp endpoints",
          "[potential_feedback][self_gravitation]")
{
  using aspect::PotentialFeedback::internal::radial_interpolation_stencil;
  const std::vector<double> supports = {1., 2., 4., 7.};

  for (const double radius :
  {
    -1., 1., 1.25, 2., 3., 7., 9.
  })
  {
    const auto stencil = radial_interpolation_stencil(supports, radius);
    CHECK(stencil.weights[0] + stencil.weights[1] == Approx(1.0));
    CHECK(stencil.weights[0] >= 0.0);
    CHECK(stencil.weights[1] >= 0.0);
    const double projected_radius =
      stencil.weights[0] * supports[stencil.indices[0]]
      + stencil.weights[1] * supports[stencil.indices[1]];
    CHECK(projected_radius
          == Approx(std::min(supports.back(),
                             std::max(supports.front(), radius))));
  }

  const auto lower = radial_interpolation_stencil(supports, -1.);
  CHECK(lower.indices[0] == 0);
  CHECK(lower.weights[0] == 1.0);
  const auto upper = radial_interpolation_stencil(supports, 9.);
  CHECK(upper.indices[0] == supports.size()-1);
  CHECK(upper.weights[0] == 1.0);
  CHECK_THROWS(radial_interpolation_stencil({}, 1.));
  CHECK_THROWS(radial_interpolation_stencil({1., 1.}, 1.));
  CHECK_THROWS(radial_interpolation_stencil({2., 1.}, 1.));
}


TEST_CASE("Symmetric radial transfer converges under support refinement",
          "[potential_feedback][self_gravitation]")
{
  const std::vector<double> targets = {1.17, 1.91, 2.63, 3.77};
  const std::vector<double> sources = {1.31, 2.22, 3.49};
  double coarse_error = 0.0;
  double refined_error = 0.0;
  for (const unsigned int intervals :
  {
    8u, 256u
  })
  {
    std::vector<double> supports;
    for (unsigned int i = 0; i <= intervals; ++i)
      supports.push_back(1. + 3. * i / intervals);
    const auto approximate =
      make_symmetric_support_operator(targets, sources, supports, 6);
    double error = 0.0;
    for (unsigned int i = 0; i < targets.size(); ++i)
      for (unsigned int j = 0; j < sources.size(); ++j)
        error = std::max(error,
                         std::abs(approximate[i][j]
                                  - kernel(6, targets[i], sources[j])));
    if (intervals == 8)
      coarse_error = error;
    else
      refined_error = error;
  }
  CHECK(refined_error < coarse_error / 100.0);
  CHECK(refined_error < 4e-4);
}


TEST_CASE("Symmetric support deposition is MPI partition independent",
          "[potential_feedback][self_gravitation]")
{
  const std::vector<double> supports = {1., 2., 3., 4.};
  const unsigned int rank =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int n_ranks =
    dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const double source_radius = 1.25 + 2.5 * (rank + 0.5) / n_ranks;
  aspect::PotentialFeedback::internal::RadialGreenMomentAccumulator
  accumulator(supports, 2, 2, supports.back());
  std::vector<double> coefficients(3, 0.0);
  coefficients[0] = 1.0 + rank;
  accumulator.add_interpolated_source(source_radius,
                                      coefficients,
                                      coefficients);
  accumulator.mpi_sum(MPI_COMM_WORLD);
  const auto values = accumulator.evaluate().first;

  for (unsigned int support = 0; support < supports.size(); ++support)
    {
      double expected = 0.0;
      for (unsigned int source_rank = 0; source_rank < n_ranks; ++source_rank)
        {
          const double radius =
            1.25 + 2.5 * (source_rank + 0.5) / n_ranks;
          const auto stencil =
            aspect::PotentialFeedback::internal::radial_interpolation_stencil(
              supports, radius);
          for (unsigned int entry = 0; entry < 2; ++entry)
            expected += (1.0 + source_rank) * stencil.weights[entry]
                        * kernel(2,
                                 supports[support],
                                 supports[stencil.indices[entry]]);
        }
      CHECK(values[support * coefficients.size()]
            == Approx(expected).epsilon(2e-13));
    }
}


TEST_CASE("Radial cache support helpers are deterministic and capped",
          "[potential_feedback][self_gravitation]")
{
  using aspect::PotentialFeedback::internal::collect_global_radial_cache_supports;
  using aspect::PotentialFeedback::internal::lookup_radial_cache_exact;
  using aspect::PotentialFeedback::internal::merge_radial_cache_supports;

  const std::vector<double> supports =
  merge_radial_cache_supports({{4., 2., 2.}, {3., 1., 4.}}, 4);
  CHECK(supports == std::vector<double>({1., 2., 3., 4.}));
  CHECK(collect_global_radial_cache_supports({3., 1., 3., 2.},
                                             4,
                                             MPI_COMM_WORLD)
        == std::vector<double>({1., 2., 3.}));

  const std::vector<std::vector<double>> values =
  {{10.}, {20.}, {30.}, {40.}};
  CHECK(lookup_radial_cache_exact(supports, values, 0, 3.) == 30.);
  CHECK_THROWS(lookup_radial_cache_exact(supports, values, 0, 2.5));
  CHECK_THROWS(merge_radial_cache_supports({{1., 2.}, {3.}}, 2));
  CHECK_THROWS(collect_global_radial_cache_supports({1., 2., 3.},
                                                    2,
                                                    MPI_COMM_WORLD));
}
