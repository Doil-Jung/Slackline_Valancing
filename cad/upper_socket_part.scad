// Standalone upper socket part.
// Edit shared dimensions in frame_v3.scad so the assembly and this part stay linked.

part_mode = "upper_socket";
socket_side = "front";     // "front" or "rear"
print_export = true;       // use high quality $fn for STL export

include <frame_v3.scad>
