// Standalone X- service door for the upper body B piece.
// Edit shared dimensions in frame_v4.scad so the assembly and this part stay linked.
// Prints flat as placed. (*) cut USB / power-switch openings after
// measuring the real OpenCR position behind the door.

part_mode = "upper_body_door";
print_export = true;      // use high quality $fn for STL export

include <frame_v4.scad>
