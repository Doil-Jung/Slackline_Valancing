// Standalone robot upper body A piece (S101 cap -> splice, taper to 80).
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.
// Print UPRIGHT as placed (~161 tall).

part_mode = "upper_body_a";
part_cutaway = false;     // true removes a corner quarter to inspect the box interior
print_export = true;      // use high quality $fn for STL export

include <frame_v4.scad>
