// Standalone robot ankle bearing carrier part (v5.3: OPEN carrier -
// two 608 bearing housings + top bridge RAISED to z 23..31 so the 42 mm
// AS5047P board (top edge z=+21) passes 2 mm under it; bridge end walls
// weld the raised bridge onto the housings. Flange z0 = 31 (+9 vs v5.2)
// -> the lower body grew 9 mm; robot_l1/sim L1 updated to match, the
// printed lower spine is reused unchanged.
// The AS5047P rides on the L-bracket (ankle_pcb_bracket_part) screwed
// to the underside of the top bridge.
// Edit shared dimensions in frame_v5.scad so the assembly stays linked.

part_mode = "ankle_carrier";
part_cutaway = false;     // true cuts the bearing pockets open for inspection
print_export = true;

include <frame_v5.scad>
