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



TEST_CASE("Surface history parses CitcomSVE stage ages",
          "[potential_feedback][gia]")
{
  using aspect::PotentialFeedback::SurfaceHistoryUtilities::parse_schedule;

  const std::string negative_age_schedule =
    "3 1\n"
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
    "3 1\n"
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
    "3 1\n"
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
