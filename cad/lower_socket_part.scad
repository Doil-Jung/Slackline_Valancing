// Standalone lower socket part - v5: INTEGRATED ANKLE AXLE version.
// The bar cup is retired; the socket prints with a 7.8 mm axle
// (hub stop -> through one carrier 608 -> magnet-pocket tip).
// front = encoder side in the assembly, but both sides print the pocket
// (mirror parts double as spares).
// Edit shared dimensions in frame_v5.scad so the assembly stays linked.

part_mode = "lower_socket";
socket_side = "front";     // "front" or "rear"
part_cutaway = false;
print_export = true;

include <frame_v5.scad>
