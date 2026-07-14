// Standalone upper rope socket (front or rear - set socket_side below).
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.
// v4 2026-07-13: shaft print dia 7.8 on BOTH sockets (final, easier slip fit).
// Front (bearing side, no encoder) has a SHORT shaft (~9.2mm past the outer
// bearing face, just for the lock collar) - this is by design, not a bug;
// only the rear/encoder side needs the long rear_shaft_out=48 shaft for the
// coupler+encoder engagement. For the rear/broken one, use
// upper_socket_encoder_part.scad instead (dedicated file, socket_side="rear").

part_mode = "upper_socket";
socket_side = "front";     // "front" or "rear" - this file = FRONT (short shaft, no encoder)
print_export = true;       // use high quality $fn for STL export

include <frame_v4.scad>
