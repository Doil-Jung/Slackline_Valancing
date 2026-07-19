// DEPRECATED in v5.2 - the closed carrier + bottom lid design was replaced
// by the OPEN carrier: the AS5047P now mounts on a T-shaped bracket screwed
// to the underside of the carrier top bridge.
//   -> use ankle_pcb_bracket_part.scad instead.
// This file is kept only so old links don't break; it renders nothing
// (frame_v5.scad prints a deprecation echo for this part_mode).

part_mode = "ankle_pcb_lid";
print_export = true;

include <frame_v5.scad>
