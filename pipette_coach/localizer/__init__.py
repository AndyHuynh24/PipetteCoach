"""Plate localizer: frame -> homography H mapping canonical plate coordinates to image pixels.

Canonical coordinates are defined in pipette_coach.protocol.wells (A1 = (0, 0), H12 = (11, 7)).
"""

from pipette_coach.localizer.localizer import PlateFit, PlateLocalizer, localize_clip, localize_clip_frames

__all__ = ["PlateFit", "PlateLocalizer", "localize_clip", "localize_clip_frames"]
