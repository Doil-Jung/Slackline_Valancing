// Standalone robot lower spine part.
// v5: reprint needed - the carrier boss grew 40 -> 44, so the flange
// height (lsp_z0) moved up 2 mm with it.
// Edit shared dimensions in frame_v5.scad so the assembly stays linked.

part_mode = "lower_spine";
part_cutaway = false;
print_export = true;

include <frame_v5.scad>
