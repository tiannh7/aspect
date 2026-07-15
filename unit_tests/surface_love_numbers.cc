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

#include <aspect/postprocess/surface_love_numbers.h>


TEST_CASE("Surface load Love number h uses radial displacement",
          "[postprocess][surface_love_numbers]")
{
  const std::pair<double,double> radial_displacement(-0.9441318, 0.125);
  const std::pair<double,double> love_number =
    aspect::Postprocess::SurfaceLoveNumberUtilities::
    load_love_number_from_displacement(radial_displacement, 1.0, 1.0);

  REQUIRE(love_number.first == Approx(-0.9441318));
  REQUIRE(love_number.second == Approx(0.125));

  const double surface_density_jump = 3381.0;
  const double load_density = 4400.0;
  const double density_scaled_value =
    radial_displacement.first * surface_density_jump / load_density;
  REQUIRE(density_scaled_value == Approx(-0.7254794581363636));
  REQUIRE(love_number.first != Approx(density_scaled_value));
}
