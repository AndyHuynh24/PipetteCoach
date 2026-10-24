import React from "react";
import { Series } from "remotion";
import { Act1Props, SceneScale } from "./SceneScale";
import { SceneToday, TODAY_FRAMES } from "./SceneToday";
import { SceneRisk } from "./SceneRisk";

/** Act 1: The problem, docs/demo_script.md. The three stats (SceneStats) moved to the end of the video. */
export const Act1: React.FC<Act1Props> = (props) => (
  <Series>
    <Series.Sequence name="How big it is" durationInFrames={300}>
      <SceneScale {...props} />
    </Series.Sequence>
    <Series.Sequence name="How it's done today" durationInFrames={TODAY_FRAMES}>
      <SceneToday {...props} />
    </Series.Sequence>
    <Series.Sequence name="The risk" durationInFrames={120}>
      <SceneRisk {...props} />
    </Series.Sequence>
  </Series>
);
