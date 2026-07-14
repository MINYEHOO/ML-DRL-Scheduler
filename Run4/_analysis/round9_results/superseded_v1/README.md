v1 outputs superseded in audit round 10b: the v1 census hooked
SlotAllocationPlanner.close_rbg class-wide and therefore counted every unit
TWICE (scheduler-side planner + env-side planner; Gate 2 makes them
bit-identical, so all RATES were correct but raw counts were exactly 2x).
See ../r10_tail_census_v2.out for the corrected env-side-only run.
