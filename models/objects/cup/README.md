# Table Cup

- Original archive: `/home/sparx3d/Documents/cup.zip`.
- `source/` preserves its original `coffcup.stp`, `coffcup.stl`, and `cup1.jpg`.
- `cup_82x98mm.stl` is the MuJoCo visual asset, in metres, upright along +Z.
- Cup body outer diameter: 0.082 m, excluding the handle.
- Cup height: 0.098 m. The original handle and open bowl are retained.
- Origin: body-axis centre at the bottom. Handle points along +X.

The source STL has an offset origin, a +Y height axis, body diameter
9.018693923950195 source units, and height 10 source units. Horizontal
coordinates are scaled by `0.082 / 9.018693923950195`; height is scaled by
`0.098 / 10`. This deliberately adjusts the aspect ratio to meet both requested
dimensions rather than assuming the source CAD already has the required size.
Source STEP/STL files are unchanged; only the derived STL is resized/oriented.
Zero-area triangles are discarded and normals recalculated in the derived mesh.

Scene: `models/scenes/table_and_cup.xml`, included by
`models/scenario_g1_dex5_stationary.xml`. Table top is 0.75 m along Y and
0.62 m along X, at 0.762 m height (30 inches). The cup bottom sits on that
surface at X=0.65, Y=-0.16, in front of the robot on its right-hand side.

The current retargeting demo uses kinematic qpos updates, not a grasp simulation.
The cup is a fixed visual target with contacts disabled, not a dynamic graspable
object. This avoids MuJoCo's convex collision hull filling the bowl/handle holes.
No arm targets, tracking/calibration, robot joints, or walking behavior are changed.
