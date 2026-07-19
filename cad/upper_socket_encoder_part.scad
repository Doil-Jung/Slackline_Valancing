// Standalone upper socket, REAR/encoder side.
// v5: shaft SHORTENED (rear_shaft_out 48 -> 13, no coupler) and the tip
// gets a 6.15 x 2.2 magnet pocket (diametral 6x2.5 magnet, 0.3 proud).
// Edit shared dimensions in frame_v5.scad so the assembly stays linked.

part_mode = "upper_socket";
socket_side = "rear";
print_export = true;

include <frame_v5.scad>
