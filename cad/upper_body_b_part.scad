// Standalone robot upper body B piece = v4 equipment bay
// (splice tongues / central OpenCR bulkhead / battery shelf / door opening).
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.
// Print UPRIGHT as placed, tongues down (~190 tall); the battery shelf
// bridges ~30 mm - allow bridging or a little support via the door opening.

part_mode = "upper_body_b";
part_cutaway = false;     // true removes a corner quarter: bulkhead, OpenCR
                          // bosses, guide grooves and battery shelf visible.
                          // Set back to false before exporting the STL!
print_export = true;      // use high quality $fn for STL export

include <frame_v4.scad>
