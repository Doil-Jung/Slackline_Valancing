// VIEWING FILE - open this in OpenSCAD and press F5.
// Assembled ankle cutaway: socket axles in the 608s, hub stops, magnet,
// air gap, AS5047P on the lid post, cavity and window - all in one section.
// Mouse drag = rotate, wheel = zoom. NOT for printing.
//
// part_cutaway = false  ->  same assembly WITHOUT the cut (whole ankle).

part_mode = "ankle_section";
part_cutaway = true;       // true = half cut away, false = intact
print_export = false;      // fast preview quality

include <frame_v5.scad>
