// Standalone front bearing mount part.
// Edit shared dimensions in frame_v3.scad so the assembly and this part stay linked.

part_mode = "bearing_mount";
part_cutaway = false;     // true shows the two 608 pockets and center land
print_export = true;      // use high quality $fn for STL export

include <frame_v3.scad>
