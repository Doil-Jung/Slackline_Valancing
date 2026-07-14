// Standalone robot ankle bearing carrier part.
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.

part_mode = "ankle_carrier";
part_cutaway = false;     // true cuts the two 608 pockets open for inspection
print_export = true;      // use high quality $fn for STL export

include <frame_v4.scad>
