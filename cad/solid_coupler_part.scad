// Standalone solid coupler part.
// Edit shared dimensions in frame_v3.scad so the assembly and this part stay linked.

part_mode = "solid_coupler";
part_cutaway = false;      // true shows the stepped blind bores and center stop
print_export = true;       // use high quality $fn for STL export

include <frame_v3.scad>
