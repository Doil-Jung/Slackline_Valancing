// Standalone lock collar part.
// Edit shared dimensions in frame_v3.scad so the assembly and this part stay linked.

part_mode = "lock_collar";
collar_dir = 1;            // flip to -1 only for mirrored preview
print_export = true;       // use high quality $fn for STL export

include <frame_v3.scad>
