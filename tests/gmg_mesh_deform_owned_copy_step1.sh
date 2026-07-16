#!/bin/sh

# Keep only stable evidence that the physical step-1 GMG mesh solve completed
# and that execution continued past the owned-only copy and ALE interpolation.
awk '
  /^\*\*\* Timestep 1:/ {
    in_step_one = 1
    print "*** Timestep 1"
    next
  }

  in_step_one && /^   Solving mesh displacement system\.\.\. [0-9]+ iterations\.$/ {
    mesh_solve_completed = 1
    print "GMG mesh displacement solve at physical step 1 completed"
    next
  }

  in_step_one && mesh_solve_completed && /^   Solving temperature system\.\.\./ {
    if (!post_mesh_stage_reached)
      print "Post-mesh owned-only copy and ALE interpolation completed"
    post_mesh_stage_reached = 1
    next
  }

  /Termination requested by criterion:/ {
    if (!terminated)
      print "Termination requested after physical step 1"
    terminated = 1
  }

  END {
    if (!post_mesh_stage_reached || !terminated)
      exit 1
  }
'
