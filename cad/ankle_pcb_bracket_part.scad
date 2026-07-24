// Standalone v5.3 ankle PCB L-bracket for the 42 mm AS5047P module
// (Coupang 하이제니스, magnet included, side SMT 6-pin SPI socket).
// Vertical plate (46 wide) backs the full board and takes FOUR M3
// self-taps on the 37 mm corner grid (pilot 2.7, (*) measure on
// arrival). The top bar starts at the plate front plane and runs aft
// under the carrier bridge - M3 CLEARANCE (3.4) through-holes in the
// bar, self-tap PILOTS (2.7) in the bridge, screw row behind the plate
// so the driver has a straight shot from below. The board's SMT
// connector exits -X sideways through the open carrier center.
// Print orientation (set in frame_v5): plate flat on the bed, bar as a
// vertical wall - no supports.
// Assembly: screw the PCB to the plate FIRST, then bolt the bar to the
// bridge from the open center of the carrier.
// Edit shared dimensions in frame_v5.scad so everything stays linked.

part_mode = "ankle_pcb_bracket";
print_export = true;

include <frame_v5.scad>
