// Standalone rear pivot bearing + encoder bracket.
// v5: SHORTER (plate follows the 13 mm shaft + AS5047P chain).
// v5.3: sized for the 42 mm AS5047P module - FOUR M3 self-tap pilots on
// the 37 mm corner grid ((*) measure on arrival), no wire slot. The
// frame is ASYMMETRIC: bottom face stays on the profile top, the top
// bar is raised so the board (+-21) and its SMT connector (mounted
// pointing UP, tip ~axis+27) clear the window. The AS5047P PCB screws
// to the plate INNER face, chip facing the magnet; the SPI cable exits
// upward, then loops down the rear diagonal path.

part_mode = "encoder_bracket";
print_export = true;

include <frame_v5.scad>
