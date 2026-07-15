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

#include <aspect/potential_feedback/glacial_isostatic_adjustment.h>
#include <aspect/potential_feedback/interface.h>
#include <aspect/potential_feedback/surface_history.h>
#include <aspect/potential_feedback/tidal_potential.h>


TEST_CASE("Potential feedback GIA is disabled by default",
          "[potential_feedback][gia]")
{
  dealii::ParameterHandler prm;
  aspect::PotentialFeedback::Settings::declare_parameters(prm);

  aspect::PotentialFeedback::Settings settings;
  settings.parse_parameters(prm);

  REQUIRE(settings.feedback_mechanisms.empty());
  REQUIRE_FALSE(settings.has_active_mechanisms());

  prm.enter_subsection("Potential feedback");
  prm.enter_subsection("Potential iteration");
  prm.set("Freeze feedback after timestep zero", "true");
  prm.leave_subsection();
  prm.leave_subsection();

  settings.parse_parameters(prm);
  REQUIRE(settings.freeze_feedback_after_timestep_zero);
}



TEST_CASE("Potential feedback GIA requires self gravity",
          "[potential_feedback][gia]")
{
  dealii::ParameterHandler prm;
  aspect::PotentialFeedback::Settings::declare_parameters(prm);
  prm.enter_subsection("Potential feedback");
  prm.set("List of feedback mechanisms", "glacial isostatic adjustment");
  prm.leave_subsection();

  aspect::PotentialFeedback::Settings settings;
  REQUIRE_THROWS_WITH(settings.parse_parameters(prm),
                      Contains("requires `self gravity'"));
}



TEST_CASE("Tidal potential uses external solid-harmonic radial scaling",
          "[potential_feedback][tidal]")
{
  aspect::PotentialFeedback::Settings settings;
  settings.tidal_model_name = "spherical harmonic potential";
  settings.tidal_harmonic_degree = 2;
  settings.tidal_harmonic_order = 1;
  settings.tidal_coefficient_type = "sine";
  settings.tidal_potential_height_amplitude = 4.0;

  aspect::PotentialFeedback::TidalPotential tidal_potential;
  tidal_potential.configure_from_settings(settings, 1, 4, 3);

  const double outer_radius = 10.0;
  const double radius = 5.0;
  const double theta = dealii::numbers::PI / 4.0;
  const double phi = dealii::numbers::PI / 2.0;
  const dealii::Point<3> position(radius * std::sin(theta) * std::cos(phi),
                                  radius * std::sin(theta) * std::sin(phi),
                                  radius * std::cos(theta));
  const auto harmonic =
    aspect::Utilities::real_spherical_harmonic(2, 1, theta, phi);

  REQUIRE(tidal_potential.is_enabled());
  REQUIRE(tidal_potential.full_domain_potential_height(position,
                                                       outer_radius,
                                                       0.0)
          == Approx(4.0 * 0.25 * harmonic.second));
}



TEST_CASE("Surface history parses CitcomSVE stage ages",
          "[potential_feedback][gia]")
{
  using aspect::PotentialFeedback::SurfaceHistoryUtilities::parse_schedule;

  const std::string negative_age_schedule =
    "2 1\n"
    "-26.0 10\n"
    "-25.5 5\n"
    "0.0 0\n";
  const auto negative_age_stages =
    parse_schedule(negative_age_schedule,
                   "citcomsve stage ages",
                   7,
                   false);

  REQUIRE(negative_age_stages.size() == 3);
  REQUIRE(negative_age_stages[0].time == Approx(0.0));
  REQUIRE(negative_age_stages[1].time
          == Approx(500.0 * aspect::year_in_seconds));
  REQUIRE(negative_age_stages[2].time
          == Approx(26000.0 * aspect::year_in_seconds));
  REQUIRE(negative_age_stages[0].file_number == 7);
  REQUIRE(negative_age_stages[2].file_number == 9);

  const std::string positive_age_schedule =
    "2 1\n"
    "26.0 10\n"
    "25.5 5\n"
    "0.0 0\n";
  const auto positive_age_stages =
    parse_schedule(positive_age_schedule,
                   "citcomsve stage ages",
                   2,
                   false);
  REQUIRE(positive_age_stages[1].time
          == Approx(negative_age_stages[1].time));
  REQUIRE(positive_age_stages[2].time
          == Approx(negative_age_stages[2].time));

  const std::string nonmonotone_schedule =
    "2 1\n"
    "-26.0 10\n"
    "-25.0 5\n"
    "-25.5 5\n";
  REQUIRE_THROWS_WITH(
    parse_schedule(nonmonotone_schedule,
                   "citcomsve stage ages",
                   0,
                   false),
    Contains("strictly increasing"));
}



TEST_CASE("Surface history parses elapsed nonuniform stages",
          "[potential_feedback][gia]")
{
  const auto stages =
    aspect::PotentialFeedback::SurfaceHistoryUtilities::parse_schedule(
      "# elapsed years and file number\n0 4\n2.5 9\n10 12\n",
      "elapsed time and file number",
      0,
      true);

  REQUIRE(stages.size() == 3);
  REQUIRE(stages[0].file_number == 4);
  REQUIRE(stages[1].file_number == 9);
  REQUIRE(stages[1].time
          == Approx(2.5 * aspect::year_in_seconds));
  REQUIRE(stages[2].time
          == Approx(10.0 * aspect::year_in_seconds));
}



TEST_CASE("Surface history parses canonical CitcomSVE regular grids",
          "[potential_feedback][gia]")
{
  const std::string contents =
    "4 2\n"
    "45 45 1\n"
    "135 45 2\n"
    "225 45 3\n"
    "315 45 4\n"
    "45 -45 5\n"
    "135 -45 6\n"
    "225 -45 7\n"
    "315 -45 8\n";

  auto grid =
    aspect::PotentialFeedback::SurfaceHistoryUtilities::
    parse_citcomsve_regular_grid(contents, 2.0);

  REQUIRE(grid.coordinate_values.size() == 2);
  REQUIRE(grid.coordinate_values[0].size() == 6);
  REQUIRE(grid.coordinate_values[1].size() == 4);

  aspect::Utilities::StructuredDataLookup<2> lookup(1, 1.0);
  lookup.reinit({"surface value"},
                std::move(grid.coordinate_values),
                std::move(grid.data_tables));

  REQUIRE(lookup.get_data(dealii::Point<2>(dealii::numbers::PI / 4.0,
                                           dealii::numbers::PI / 4.0), 0)
          == Approx(2.0));
  REQUIRE(lookup.get_data(dealii::Point<2>(0.0,
                                           dealii::numbers::PI / 4.0), 0)
          == Approx(5.0));
  REQUIRE(lookup.get_data(dealii::Point<2>(2.0 * dealii::numbers::PI,
                                           dealii::numbers::PI / 4.0), 0)
          == Approx(5.0));
  REQUIRE(lookup.get_data(dealii::Point<2>(dealii::numbers::PI,
                                           0.0), 0)
          == Approx(5.0));
  REQUIRE(lookup.get_data(dealii::Point<2>(dealii::numbers::PI,
                                           dealii::numbers::PI), 0)
          == Approx(13.0));
}



TEST_CASE("Canonical CitcomSVE GIA sea level equation conserves water mass",
          "[potential_feedback][gia]")
{
  using aspect::PotentialFeedback::GIA::barystatic_sea_level;
  using aspect::PotentialFeedback::GIA::sea_level_change;

  const double ocean_area = 100.0;
  const double ice_mass_change = -2000.0;
  const double relative_sea_level_volume = 3.0;
  const double density_water = 1000.0;

  const double barystatic =
    barystatic_sea_level(ocean_area,
                         ice_mass_change,
                         relative_sea_level_volume,
                         density_water);
  REQUIRE(barystatic == Approx(-0.01));
  REQUIRE(density_water
          * (relative_sea_level_volume
             + barystatic * ocean_area)
          == Approx(-ice_mass_change));

  REQUIRE(sea_level_change(1.0,
                           0.5,
                           0.25)
          == Approx(0.75));
  REQUIRE(sea_level_change(0.0,
                           0.5,
                           0.25)
          == Approx(0.0));
  REQUIRE(sea_level_change(0.5,
                           0.5,
                           0.25)
          == Approx(0.375));
}
