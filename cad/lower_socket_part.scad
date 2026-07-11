// Standalone lower socket part.
// Edit shared dimensions in frame_v3.scad so the assembly and this part stay linked.

part_mode = "lower_socket";
socket_side = "front";     // "front" or "rear"
part_cutaway = false;      // false for STL export, true to inspect pipe stops and wire passage
print_export = true;       // use high quality $fn for STL export

include <frame_v3.scad>
