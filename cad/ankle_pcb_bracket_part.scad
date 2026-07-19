// Standalone v5.2 ankle PCB T-bracket (print orientation: flipped so the
// bridge-contact face of the bar is on the build plate).
// T shape: horizontal bar screws up into the carrier top-bridge underside
// - M3 CLEARANCE (3.4) through-holes in this bar, self-tap PILOTS (2.7)
// are in the bridge; the screw row sits BEHIND the vertical plate so the
// driver has a straight shot from below. The vertical plate carries the
// AS5047P module facing the front-axle magnet (2 x M2 self-tap, pilot 1.7)
// with a ribbon slot for the 6-wire SPI cable.
// Assembly: screw the PCB to the plate FIRST, then bolt the bar to the
// bridge from the open center of the carrier.
// Edit shared dimensions in frame_v5.scad so everything stays linked.

part_mode = "ankle_pcb_bracket";
print_export = true;

include <frame_v5.scad>
