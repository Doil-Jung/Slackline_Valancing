// Standalone robot ankle bearing carrier part (v5: 26x26 sensor cavity,
// flat bottom pad + lid window, boss 44). Pair with ankle_pcb_lid_part.
// Edit shared dimensions in frame_v5.scad so the assembly stays linked.

part_mode = "ankle_carrier";
part_cutaway = false;     // true cuts the pockets/cavity open for inspection
print_export = true;

include <frame_v5.scad>
