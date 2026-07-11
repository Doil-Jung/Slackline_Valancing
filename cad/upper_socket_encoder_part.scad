// Standalone upper socket part for the encoder side.
// Edit shared dimensions in frame_v3.scad so the assembly and this part stay linked.

part_mode = "upper_socket";
socket_side = "rear";      // encoder side, with the longer rear shaft extension
print_export = true;       // use high quality $fn for STL export

include <frame_v3.scad>
