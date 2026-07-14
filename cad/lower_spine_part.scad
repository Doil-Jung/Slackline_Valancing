// Standalone robot lower spine part (foot 80 wide -> H101 waist cap).
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.
// Print UPRIGHT as placed (~198 tall).

part_mode = "lower_spine";
part_cutaway = false;     // true removes a corner quarter to inspect the box interior
print_export = true;      // use high quality $fn for STL export

include <frame_v4.scad>
