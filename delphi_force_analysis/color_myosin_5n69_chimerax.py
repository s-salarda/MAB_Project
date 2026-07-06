# color_myosin_5n69_chimerax.py
# Converted from Chimera to ChimeraX
# Run with: runscript /path/to/color_myosin_5n69_chimerax.py

from chimerax.core.commands import run

# Show as ribbon only
run(session, "hide #1 atoms")
run(session, "show #1 cartoons")

# Color each region
run(session, "color #1:1-112,124-169,176-214 #ffffd9 cartoons")
run(session, "color #1:113-123,666-671,170-175,454-460,244-265 #a1dab4 cartoons")
run(session, "color #1:215-231,266-453,604-621 #41b6c4 cartoons")
run(session, "color #1:645-665,472-566,577-590 #2c7fb8 cartoons")
run(session, "color #1:672-765 #253494 cartoons")
run(session, "color #1:766-810 #5420A1 cartoons")
run(session, "color #1:814-1007 #104547 cartoons")
run(session, "color #1:231-243,461-471,176-181,621-646,590-603,567-577 #AF3800 cartoons")