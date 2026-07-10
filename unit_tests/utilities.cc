/*
  Copyright (C) 2018 - 2024 by the authors of the ASPECT code.

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
#include <aspect/utilities.h>

#include <deal.II/base/mpi.h>

#include <cmath>

namespace
{
  std::vector<std::vector<double>>
  make_spherical_harmonic_test_fields(const std::vector<double> &theta,
                                      const std::vector<double> &phi)
  {
    std::vector<std::vector<double>> fields(3,
                                            std::vector<double>(theta.size()));

    for (unsigned int i = 0; i < theta.size(); ++i)
      {
        fields[0][i] = std::sin(theta[i]) + 0.25 * std::cos(phi[i]);
        fields[1][i] = std::cos(2.0 * theta[i]) * std::sin(phi[i])
                       + 0.1 * static_cast<double>(i + 1);
        fields[2][i] = 1.0 + 0.1 * theta[i] - 0.05 * phi[i]
                       + std::sin(3.0 * phi[i]) * std::cos(theta[i]);
      }

    return fields;
  }



  void
  compare_spherical_harmonic_coefficients(
    const std::pair<std::vector<double>, std::vector<double>> &computed,
    const std::pair<std::vector<double>, std::vector<double>> &expected)
  {
    REQUIRE(computed.first.size() == expected.first.size());
    REQUIRE(computed.second.size() == expected.second.size());

    for (unsigned int i = 0; i < expected.first.size(); ++i)
      {
        INFO("cosine coefficient index i=" << i);
        REQUIRE(computed.first[i] == Approx(expected.first[i]).margin(1e-12));
      }
    for (unsigned int i = 0; i < expected.second.size(); ++i)
      {
        INFO("sine coefficient index i=" << i);
        REQUIRE(computed.second[i] == Approx(expected.second[i]).margin(1e-12));
      }
  }



  std::pair<std::vector<double>, std::vector<double>>
  reference_spherical_harmonic_analysis(
    const aspect::Utilities::SphericalHarmonicTransform &transform,
    const unsigned int max_degree,
    const std::vector<double> &theta,
    const std::vector<double> &phi,
    const std::vector<double> &weights,
    const std::vector<double> &values,
    const MPI_Comm mpi_comm)
  {
    const unsigned int n_coefficients = transform.n_coefficients();
    std::vector<double> local_cos_coefficients(n_coefficients, 0.0);
    std::vector<double> local_sin_coefficients(n_coefficients, 0.0);

    for (unsigned int point_index = 0; point_index < theta.size(); ++point_index)
      {
        const double value_times_weight =
          values[point_index] * weights[point_index];
        for (unsigned int degree = 0; degree <= max_degree; ++degree)
          for (unsigned int order = 0; order <= degree; ++order)
            {
              const std::pair<double, double> ylm =
                aspect::Utilities::real_spherical_harmonic(degree,
                                                           order,
                                                           theta[point_index],
                                                           phi[point_index]);
              const unsigned int coefficient_index =
                transform.index(degree, order);
              local_cos_coefficients[coefficient_index] +=
                value_times_weight * ylm.first;
              local_sin_coefficients[coefficient_index] +=
                value_times_weight * ylm.second;
            }
      }

    std::vector<double> cos_coefficients(n_coefficients);
    std::vector<double> sin_coefficients(n_coefficients);
    dealii::Utilities::MPI::sum(local_cos_coefficients,
                                mpi_comm,
                                cos_coefficients);
    dealii::Utilities::MPI::sum(local_sin_coefficients,
                                mpi_comm,
                                sin_coefficients);

    return std::make_pair(cos_coefficients, sin_coefficients);
  }



  void
  check_spherical_harmonic_analyze_multiple(
    const unsigned int max_degree,
    const MPI_Comm mpi_comm,
    const bool leave_some_ranks_empty)
  {
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

    const std::vector<double> global_theta =
    {
      0.17, 0.41, 0.73, 1.02, 1.37, 1.61, 1.94,
      2.18, 2.46, 2.73, 0.89, 1.79, 2.91
    };
    const std::vector<double> global_phi =
    {
      0.11, 0.59, 1.24, 1.83, 2.51, 3.06, 3.77,
      4.18, 4.86, 5.47, 2.22, 3.39, 5.92
    };
    const std::vector<double> global_weights =
    {
      0.08, 0.11, 0.09, 0.13, 0.10, 0.12, 0.07,
      0.14, 0.06, 0.15, 0.05, 0.16, 0.04
    };
    const std::vector<std::vector<double>> global_fields =
      make_spherical_harmonic_test_fields(global_theta, global_phi);

    std::vector<double> theta;
    std::vector<double> phi;
    std::vector<double> weights;
    std::vector<std::vector<double>> fields(global_fields.size());

    for (unsigned int i = 0; i < global_theta.size(); ++i)
      {
        const bool keep_point =
          leave_some_ranks_empty ? rank == 0 : (i % n_ranks == rank);
        if (keep_point)
          {
            theta.push_back(global_theta[i]);
            phi.push_back(global_phi[i]);
            weights.push_back(global_weights[i]);
            for (unsigned int field_index = 0;
                 field_index < global_fields.size();
                 ++field_index)
              fields[field_index].push_back(global_fields[field_index][i]);
          }
      }

    const aspect::Utilities::SphericalHarmonicTransform transform(max_degree,
                                                                  0);
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>
    multi_coefficients =
      transform.analyze_multiple(theta, phi, weights, fields, mpi_comm);

    REQUIRE(multi_coefficients.size() == fields.size());
    for (unsigned int field_index = 0; field_index < fields.size(); ++field_index)
      {
        INFO("max_degree=" << max_degree
             << " field_index=" << field_index
             << " leave_some_ranks_empty=" << leave_some_ranks_empty);
        const std::pair<std::vector<double>, std::vector<double>>
        single_coefficients =
          transform.analyze(theta,
                            phi,
                            weights,
                            fields[field_index],
                            mpi_comm);
        compare_spherical_harmonic_coefficients(multi_coefficients[field_index],
                                                single_coefficients);

        const std::pair<std::vector<double>, std::vector<double>>
        reference_coefficients =
          reference_spherical_harmonic_analysis(transform,
                                                max_degree,
                                                theta,
                                                phi,
                                                weights,
                                                fields[field_index],
                                                mpi_comm);
        compare_spherical_harmonic_coefficients(multi_coefficients[field_index],
                                                reference_coefficients);
      }
  }
}

TEST_CASE("Utilities::weighted_p_norm_average")
{
  std::vector<double> weights = {1,1,2,2,3,3};
  std::vector<double> values = {6,5,4,3,2,1};
  std::vector<double> p_norms = {-1000,-2.5,-2,-1,0,1,2,2.5,3,4,1000};
  std::vector<double> expected = {1., 1.59237, 1.6974 , 1.98895, 2.38899, 2.83333, 3.24037, 3.41824, 3.57872, 3.85347, 6. };

  for (unsigned int i = 0; i < p_norms.size(); i++)
    {
      INFO("check i=" << i << ": ");
      REQUIRE(aspect::Utilities::weighted_p_norm_average(weights,values,p_norms[i]) == Approx(expected[i]));
    }

}

TEST_CASE("Utilities::AsciiDataLookup")
{
  using namespace dealii;

  //TODO: add support for setting data directly instead of relying on a file to load:
  aspect::Utilities::StructuredDataLookup<1> lookup(2 /*n_components*/, 1.0 /*scaling*/);
  const std::string data_filename = aspect::Utilities::expand_ASPECT_SOURCE_DIR("$ASPECT_SOURCE_DIR/data/boundary-velocity/ascii-data/test/box_2d_left.0.txt");
  lookup.load_file(data_filename, MPI_COMM_WORLD);

  INFO(lookup.get_data(Point<1>(330000./2.0),0));
  INFO(lookup.get_data(Point<1>(330000./2.0),1));
  INFO(lookup.get_gradients(Point<1>(330000./2.0),0));
  INFO(lookup.get_gradients(Point<1>(330000./2.0),1));

  REQUIRE(lookup.get_data(Point<1>(330000./2.0),0) == Approx(0.5));
  REQUIRE(lookup.get_data(Point<1>(330000./2.0),1) == Approx(0.0));
  REQUIRE(lookup.get_gradients(Point<1>(330000./2.0),0)[0] == Approx(-1.0/330000.));
  REQUIRE(lookup.get_gradients(Point<1>(330000./2.0),1)[0] == Approx(0.0));
}


TEST_CASE("Utilities::AsciiDataLookup manual dim=1")
{
  using namespace dealii;

  aspect::Utilities::StructuredDataLookup<1> lookup(2 /*n_components*/, 1.0 /*scaling*/);

  std::vector<std::string> column_names = {"a", "b"};
  Table<1,double> table(2);
  std::vector<Table<1,double>> raw_data(2, table);

  std::vector<std::vector<double>> coordinate_values(1, std::vector<double>({1.0, 2.0}));
  // c1:
  raw_data[0](0) = 0.0;
  raw_data[0](1) = 1.0;
  // c2:
  raw_data[1](0) = 5.0;
  raw_data[1](1) = 3.0;

  lookup.reinit(column_names, std::move(coordinate_values), std::move(raw_data),
                MPI_COMM_SELF, numbers::invalid_unsigned_int);

  INFO(lookup.get_data(Point<1>(1.5), 0));
  INFO(lookup.get_data(Point<1>(1.5), 1));

  REQUIRE(lookup.get_data(Point<1>(1.5),0) == Approx(0.5));
  REQUIRE(lookup.get_data(Point<1>(1.5),1) == Approx(4.0));

  REQUIRE(lookup.get_gradients(Point<1>(1.5),0)[0] == Approx(1.0));
  REQUIRE(lookup.get_gradients(Point<1>(1.5),1)[0] == Approx(-2.0));
}

TEST_CASE("Utilities::AsciiDataLookup manual dim=2")
{
  using namespace dealii;

  aspect::Utilities::StructuredDataLookup<2> lookup(1 /*n_components*/, 1.0 /*scaling*/);

  std::vector<std::string> column_names = {"topography"};
  std::vector<Table<2,double>> raw_data(1, Table<2,double>(3,3));
  std::vector<std::vector<double>> coordinate_values(2, std::vector<double>(3, 0.));

  // x:
  coordinate_values[0] = {0., 1., 3.};
  // y:
  coordinate_values[1] = {5., 6., 7.};
  // c1:
  raw_data[0](0,0) = 1.0;
  raw_data[0](1,0) = 2.0;
  raw_data[0](2,0) = 3.0;
  raw_data[0](0,1) = 4.0;
  raw_data[0](1,1) = 5.0;
  raw_data[0](2,1) = 6.0;
  raw_data[0](0,2) = 0.0;
  raw_data[0](1,2) = 0.0;
  raw_data[0](2,2) = 0.0;

  lookup.reinit(column_names, std::move(coordinate_values), std::move(raw_data),
                MPI_COMM_SELF, numbers::invalid_unsigned_int);

  REQUIRE(lookup.get_data(Point<2>(1.0,6.0),0) == Approx(5.0));
  REQUIRE(lookup.get_data(Point<2>(2.0,6.0),0) == Approx(5.5));
}

TEST_CASE("Utilities::AsciiDataLookup manual dim=2 equid")
{
  using namespace dealii;

  aspect::Utilities::StructuredDataLookup<2> lookup(1 /*n_components*/, 1.0 /*scaling*/);

  std::vector<std::string> column_names = {"topography"};
  std::vector<Table<2,double>> raw_data(1, Table<2,double>(3,3));
  std::vector<std::vector<double>> coordinate_values(2, std::vector<double>(3, 0.));

  // x:
  coordinate_values[0] = {0., 1., 2.};
  // y:
  coordinate_values[1] = {5., 6., 7.};
  // c1:
  raw_data[0](0,0) = 1.0;
  raw_data[0](1,0) = 2.0;
  raw_data[0](2,0) = 3.0;
  raw_data[0](0,1) = 4.0;
  raw_data[0](1,1) = 5.0;
  raw_data[0](2,1) = 6.0;
  raw_data[0](0,2) = 0.0;
  raw_data[0](1,2) = 0.0;
  raw_data[0](2,2) = 0.0;

  lookup.reinit(column_names, std::move(coordinate_values), std::move(raw_data),
                MPI_COMM_SELF, numbers::invalid_unsigned_int);

  REQUIRE(lookup.get_data(Point<2>(1.0,6.0),0) == Approx(5.0));
  REQUIRE(lookup.get_data(Point<2>(1.5,6.0),0) == Approx(5.5));
}

TEST_CASE("Utilities::SphericalHarmonicTransform analyze_multiple MPI_SELF")
{
  for (const unsigned int max_degree : {2u, 8u, 16u, 32u})
    check_spherical_harmonic_analyze_multiple(max_degree,
                                              MPI_COMM_SELF,
                                              false);
}



TEST_CASE("Utilities::SphericalHarmonicTransform analyze_multiple MPI_WORLD")
{
  for (const unsigned int max_degree : {2u, 8u, 16u, 32u})
    check_spherical_harmonic_analyze_multiple(max_degree,
                                              MPI_COMM_WORLD,
                                              false);
}



TEST_CASE("Utilities::SphericalHarmonicTransform analyze_multiple empty local points")
{
  for (const unsigned int max_degree : {2u, 8u, 16u, 32u})
    check_spherical_harmonic_analyze_multiple(max_degree,
                                              MPI_COMM_WORLD,
                                              true);
}

TEST_CASE("Random draw volume weighted average rotation matrix")
{
  std::vector<double> unsorted_volume_fractions = {2.,5.,1.,3.,6.,4.};
  std::vector<double> sorted_volume_fractions_ref = {1.,2.,3.,4.,5.,6.};
  const std::vector<std::size_t> permutation = aspect::Utilities::compute_sorting_permutation<double>(unsorted_volume_fractions);
  const std::vector<double> sorted_volume_fractions = aspect::Utilities::apply_permutation<double>(unsorted_volume_fractions,permutation);

  for (unsigned int i = 0; i < sorted_volume_fractions.size(); i++)
    {
      REQUIRE(sorted_volume_fractions[i] == Approx(sorted_volume_fractions_ref[i]));
    }

  const std::vector<dealii::Tensor<2,3>> unsorted_rotation_matrices =
  {
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{5.,5.,5.},{5.,5.,5.},{5.,5.,5.}}),
    dealii::Tensor<2,3>({{1.,1.,1.},{1.,1.,1.},{1.,1.,1.}}),
    dealii::Tensor<2,3>({{3.,3.,3.},{3.,3.,3.},{3.,3.,3.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{4.,4.,4.},{4.,4.,4.},{4.,4.,4.}})
  };

  const std::vector<dealii::Tensor<2,3>> sorted_rotation_matrices_ref =
  {
    dealii::Tensor<2,3>({{1.,1.,1.},{1.,1.,1.},{1.,1.,1.}}),
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{3.,3.,3.},{3.,3.,3.},{3.,3.,3.}}),
    dealii::Tensor<2,3>({{4.,4.,4.},{4.,4.,4.},{4.,4.,4.}}),
    dealii::Tensor<2,3>({{5.,5.,5.},{5.,5.,5.},{5.,5.,5.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}})
  };
  const std::vector<dealii::Tensor<2,3>> sorted_rotation_matrices = aspect::Utilities::apply_permutation<dealii::Tensor<2,3>>(unsorted_rotation_matrices,permutation);
  for (unsigned int i = 0; i < sorted_rotation_matrices.size(); i++)
    {
      REQUIRE(sorted_rotation_matrices[i][0][0] == Approx(sorted_rotation_matrices_ref[i][0][0]));
    }

  std::mt19937 random_number_generator;
  random_number_generator.seed(5);
  const std::vector<dealii::Tensor<2,3>> result = aspect::Utilities::rotation_matrices_random_draw_volume_weighting(unsorted_volume_fractions,
                                                   unsorted_rotation_matrices,
                                                   25,
                                                   random_number_generator);

  const std::vector<dealii::Tensor<2,3>> result_ref =
  {
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{4.,4.,4.},{4.,4.,4.},{4.,4.,4.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{4.,4.,4.},{4.,4.,4.},{4.,4.,4.}}),
    dealii::Tensor<2,3>({{4.,4.,4.},{4.,4.,4.},{4.,4.,4.}}),
    dealii::Tensor<2,3>({{5.,5.,5.},{5.,5.,5.},{5.,5.,5.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{5.,5.,5.},{5.,5.,5.},{5.,5.,5.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{3.,3.,3.},{3.,3.,3.},{3.,3.,3.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{3.,3.,3.},{3.,3.,3.},{3.,3.,3.}}),
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{5.,5.,5.},{5.,5.,5.},{5.,5.,5.}}),
    dealii::Tensor<2,3>({{5.,5.,5.},{5.,5.,5.},{5.,5.,5.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{1.,1.,1.},{1.,1.,1.},{1.,1.,1.}}),
    dealii::Tensor<2,3>({{2.,2.,2.},{2.,2.,2.},{2.,2.,2.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
    dealii::Tensor<2,3>({{6.,6.,6.},{6.,6.,6.},{6.,6.,6.}}),
  };
  for (unsigned int i = 0; i < result.size(); i++)
    {
      REQUIRE(result[i][0][0] == Approx(result_ref[i][0][0]));
    }
}

TEST_CASE("wrap angle")
{
  REQUIRE(aspect::Utilities::wrap_angle(-720.) == 0.);
  REQUIRE(aspect::Utilities::wrap_angle(-540.) == 180.);
  REQUIRE(aspect::Utilities::wrap_angle(-361.) == 359.);
  REQUIRE(aspect::Utilities::wrap_angle(-360.) == 0.);
  REQUIRE(aspect::Utilities::wrap_angle(-359.) == 1.);
  REQUIRE(aspect::Utilities::wrap_angle(-270.) == 90.);
  REQUIRE(aspect::Utilities::wrap_angle(-180.) == 180.);
  REQUIRE(aspect::Utilities::wrap_angle(-90.) == 270.);
  REQUIRE(aspect::Utilities::wrap_angle(0.) == 0.);
  REQUIRE(aspect::Utilities::wrap_angle(90.) == 90.);
  REQUIRE(aspect::Utilities::wrap_angle(180.) == 180.);
  REQUIRE(aspect::Utilities::wrap_angle(270.) == 270.);
  REQUIRE(aspect::Utilities::wrap_angle(359.) == 359.);
  REQUIRE(aspect::Utilities::wrap_angle(360.) == 0.);
  REQUIRE(aspect::Utilities::wrap_angle(361.) == 1.);
  REQUIRE(aspect::Utilities::wrap_angle(540.) == 180.);
  REQUIRE(aspect::Utilities::wrap_angle(720.) == 0.);
}

TEST_CASE("CPO elastic tensor transform functions")
{
  dealii::SymmetricTensor<2,6> reference_elastic_tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21});

// first test whether the functions are invertable
  {
    dealii::SymmetricTensor<2,6> result_up_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor));

    for (size_t i = 0; i < 6; i++)
      {
        for (size_t j = 0; j < 6; j++)
          {
            REQUIRE(reference_elastic_tensor[i][j] == Approx(result_up_down[i][j]));
          }
      }
  }
  {
    dealii::SymmetricTensor<2,6> result_down_up = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(aspect::Utilities::Tensors::to_voigt_stiffness_vector(reference_elastic_tensor));

    for (size_t i = 0; i < 6; i++)
      {
        for (size_t j = 0; j < 6; j++)
          {
            REQUIRE(reference_elastic_tensor[i][j] == Approx(result_down_up[i][j]));
          }
      }
  }
  {
    dealii::SymmetricTensor<2,6> result_up_2down_up = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(aspect::Utilities::Tensors::to_voigt_stiffness_vector(aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor)));

    for (size_t i = 0; i < 6; i++)
      {
        for (size_t j = 0; j < 6; j++)
          {
            REQUIRE(reference_elastic_tensor[i][j] == Approx(result_up_2down_up[i][j]));
          }
      }
  }

  // test rotations
  // rotation matrix
  dealii::Tensor<2,3> rotation_tensor;

  {

    // fill the rotation matrix with a rotations in all directions
    {
      double radians = 0;
      double alpha = radians;
      double beta = radians;
      double gamma = radians;
      rotation_tensor[0][0] = std::cos(alpha) * std::cos(beta);
      rotation_tensor[0][1] = std::sin(alpha) * std::cos(beta);
      rotation_tensor[0][2] = -std::sin(beta);
      rotation_tensor[1][0] = std::cos(alpha) * std::sin(beta) * std::sin(gamma) - std::sin(alpha)*std::cos(gamma);
      rotation_tensor[1][1] = std::sin(alpha) * std::sin(beta) * std::sin(gamma) + std::cos(alpha)*std::cos(gamma);
      rotation_tensor[1][2] = std::cos(beta) * std::sin(gamma);
      rotation_tensor[2][0] = std::cos(alpha) * std::sin(beta) * std::cos(gamma) + std::sin(alpha)*std::sin(gamma);
      rotation_tensor[2][1] = std::sin(alpha) * std::sin(beta) * std::cos(gamma) - std::cos(alpha)*std::sin(gamma);
      rotation_tensor[2][2] = std::cos(beta) * std::cos(gamma);
    }

    {
      dealii::SymmetricTensor<4,3> result_full_stiffness_tensor = aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor);
      dealii::SymmetricTensor<4,3> result_full_stiffness_tensor_rotate_zero = aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,result_full_stiffness_tensor);

      // first check that one the tensors didn't change with zero rotation
      for (size_t i = 0; i < 3; i++)
        {
          for (size_t j = 0; j < 3; j++)
            {
              for (size_t k = 0; k < 3; k++)
                {
                  for (size_t l = 0; l < 3; l++)
                    {
                      REQUIRE(result_full_stiffness_tensor[i][j][k][l] == Approx(result_full_stiffness_tensor_rotate_zero[i][j][k][l]));
                    }
                }
            }
        }
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(
                                                               aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,
                                                                   aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor)));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor,reference_elastic_tensor);

      // first check that one the tensors didn't change with zero rotation
      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_up_1_rotate_down[i][j] == Approx(reference_elastic_tensor[i][j]));
              REQUIRE(result_1_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }
    }

    // fill the rotation matrix with a rotations in all directions
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = radians;
      double beta = radians;
      double gamma = radians;
      rotation_tensor[0][0] = std::cos(alpha) * std::cos(beta);
      rotation_tensor[0][1] = std::sin(alpha) * std::cos(beta);
      rotation_tensor[0][2] = -std::sin(beta);
      rotation_tensor[1][0] = std::cos(alpha) * std::sin(beta) * std::sin(gamma) - std::sin(alpha)*std::cos(gamma);
      rotation_tensor[1][1] = std::sin(alpha) * std::sin(beta) * std::sin(gamma) + std::cos(alpha)*std::cos(gamma);
      rotation_tensor[1][2] = std::cos(beta) * std::sin(gamma);
      rotation_tensor[2][0] = std::cos(alpha) * std::sin(beta) * std::cos(gamma) + std::sin(alpha)*std::sin(gamma);
      rotation_tensor[2][1] = std::sin(alpha) * std::sin(beta) * std::cos(gamma) - std::cos(alpha)*std::sin(gamma);
      rotation_tensor[2][2] = std::cos(beta) * std::cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(
                                                               aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,
                                                                   aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor)));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor,reference_elastic_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::SymmetricTensor<4,3> result_up_10_rotate = aspect::Utilities::Tensors::to_full_stiffness_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,result_up_10_rotate);
          result_5_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor, result_5_rotate);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              // This test doesn't work when rotating in all rotations at the same time.
              //REQUIRE(result_1_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }
  {
    // fill the rotation matrix with a rotations in the alpha direction
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = radians;
      double beta = 0;
      double gamma = 0;
      rotation_tensor[0][0] = std::cos(alpha) * std::cos(beta);
      rotation_tensor[0][1] = std::sin(alpha) * std::cos(beta);
      rotation_tensor[0][2] = -std::sin(beta);
      rotation_tensor[1][0] = std::cos(alpha) * std::sin(beta) * std::sin(gamma) - std::sin(alpha)*std::cos(gamma);
      rotation_tensor[1][1] = std::sin(alpha) * std::sin(beta) * std::sin(gamma) + std::cos(alpha)*std::cos(gamma);
      rotation_tensor[1][2] = std::cos(beta) * std::sin(gamma);
      rotation_tensor[2][0] = std::cos(alpha) * std::sin(beta) * std::cos(gamma) + std::sin(alpha)*std::sin(gamma);
      rotation_tensor[2][1] = std::sin(alpha) * std::sin(beta) * std::cos(gamma) - std::cos(alpha)*std::sin(gamma);
      rotation_tensor[2][2] = std::cos(beta) * std::cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(
                                                               aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,
                                                                   aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor)));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor,reference_elastic_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::SymmetricTensor<4,3> result_up_10_rotate = aspect::Utilities::Tensors::to_full_stiffness_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor, result_up_10_rotate);
          result_5_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor, result_5_rotate);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              REQUIRE(result_5_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }
  {
    // fill the rotation matrix with a rotations in the beta direction
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = 0;
      double beta = radians;
      double gamma = 0;
      rotation_tensor[0][0] = std::cos(alpha) * std::cos(beta);
      rotation_tensor[0][1] = std::sin(alpha) * std::cos(beta);
      rotation_tensor[0][2] = -std::sin(beta);
      rotation_tensor[1][0] = std::cos(alpha) * std::sin(beta) * std::sin(gamma) - std::sin(alpha)*std::cos(gamma);
      rotation_tensor[1][1] = std::sin(alpha) * std::sin(beta) * std::sin(gamma) + std::cos(alpha)*std::cos(gamma);
      rotation_tensor[1][2] = std::cos(beta) * std::sin(gamma);
      rotation_tensor[2][0] = std::cos(alpha) * std::sin(beta) * std::cos(gamma) + std::sin(alpha)*std::sin(gamma);
      rotation_tensor[2][1] = std::sin(alpha) * std::sin(beta) * std::cos(gamma) - std::cos(alpha)*std::sin(gamma);
      rotation_tensor[2][2] = std::cos(beta) * std::cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(
                                                               aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,
                                                                   aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor)));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor, reference_elastic_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::SymmetricTensor<4,3> result_up_10_rotate = aspect::Utilities::Tensors::to_full_stiffness_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor, result_up_10_rotate);
          result_5_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor, result_5_rotate);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              REQUIRE(result_5_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }

  {
    // fill the rotation matrix with a rotations in the gamma direction
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = 0;
      double beta = 0;
      double gamma = radians;
      rotation_tensor[0][0] = std::cos(alpha) * std::cos(beta);
      rotation_tensor[0][1] = std::sin(alpha) * std::cos(beta);
      rotation_tensor[0][2] = -std::sin(beta);
      rotation_tensor[1][0] = std::cos(alpha) * std::sin(beta) * std::sin(gamma) - std::sin(alpha)*std::cos(gamma);
      rotation_tensor[1][1] = std::sin(alpha) * std::sin(beta) * std::sin(gamma) + std::cos(alpha)*std::cos(gamma);
      rotation_tensor[1][2] = std::cos(beta) * std::sin(gamma);
      rotation_tensor[2][0] = std::cos(alpha) * std::sin(beta) * std::cos(gamma) + std::sin(alpha)*std::sin(gamma);
      rotation_tensor[2][1] = std::sin(alpha) * std::sin(beta) * std::cos(gamma) - std::cos(alpha)*std::sin(gamma);
      rotation_tensor[2][2] = std::cos(beta) * std::cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(
                                                               aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor,
                                                                   aspect::Utilities::Tensors::to_full_stiffness_tensor(reference_elastic_tensor)));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor, reference_elastic_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::SymmetricTensor<4,3> result_up_10_rotate = aspect::Utilities::Tensors::to_full_stiffness_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Utilities::Tensors::rotate_full_stiffness_tensor(rotation_tensor, result_up_10_rotate);
          result_5_rotate = aspect::Utilities::Tensors::rotate_voigt_stiffness_matrix(rotation_tensor, result_5_rotate);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Utilities::Tensors::to_voigt_stiffness_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              REQUIRE(result_5_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }

  /**
   * test Levi-Cevita tensor function
   */
  {

    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][0][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][0][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][0][2] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][1][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][1][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][1][2] == Approx(1.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][2][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][2][1] == Approx(-1.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[0][2][2] == Approx(0.0));

    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][0][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][0][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][0][2] == Approx(-1.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][1][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][1][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][1][2] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][2][0] == Approx(1.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][2][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[1][2][2] == Approx(0.0));

    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][0][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][0][1] == Approx(1.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][0][2] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][1][0] == Approx(-1.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][1][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][1][2] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][2][0] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][2][1] == Approx(0.0));
    REQUIRE(aspect::Utilities::Tensors::levi_civita<3>()[2][2][2] == Approx(0.0));
  }
}

TEST_CASE("Utilities::string_to_unsigned_int")
{
  CHECK(aspect::Utilities::string_to_unsigned_int("1234") == 1234);

  CHECK(aspect::Utilities::string_to_unsigned_int(std::vector<std::string>({"234","0","1"}))
        == std::vector<unsigned int>({234,0,1}));

  CHECK(aspect::Utilities::string_to_unsigned_int(std::vector<std::string>({"42"}))
        == std::vector<unsigned int>({42}));

  CHECK(aspect::Utilities::string_to_unsigned_int(std::vector<std::string>({}))
        == std::vector<unsigned int>());
}
