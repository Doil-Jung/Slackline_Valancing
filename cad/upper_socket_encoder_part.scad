// Standalone upper socket part for the encoder side (rear).
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.
// v4 2026-07-13 FIX: this file was still pointing at frame_v3.scad, which has
// the OLD short rear_shaft_out=24 and shaft_print_d=7.85 - that is why the
// shaft rendered shorter here than in the assembly. Switched to frame_v4.scad
// (rear_shaft_out=48, shaft_print_d=7.8) so it matches frame_v4's assembly
// rendering. Only the include target changed; part_mode/socket_side unchanged.

part_mode = "upper_socket";
socket_side = "rear";      // encoder side, with the longer rear shaft extension
print_export = true;       // use high quality $fn for STL export

include <frame_v4.scad>
