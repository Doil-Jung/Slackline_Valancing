// Standalone robot ankle bearing carrier part (v5.2: OPEN carrier -
// two 608 bearing housings + top bridge, center removed so the encoder
// PCB / magnet / inner lock collars are assembled from the open middle).
// The AS5047P rides on the T-bracket (ankle_pcb_bracket_part) screwed
// to the underside of the top bridge.
// Edit shared dimensions in frame_v5.scad so the assembly stays linked.

part_mode = "ankle_carrier";
part_cutaway = false;     // true cuts the bearing pockets open for inspection
print_export = true;

include <frame_v5.scad>
