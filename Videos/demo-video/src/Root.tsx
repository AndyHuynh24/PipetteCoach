import React from "react";
import { Composition, Folder } from "remotion";
import { Act1 } from "./act1/Act1";
import { SceneScale } from "./act1/SceneScale";
import { SceneToday, TODAY_FRAMES } from "./act1/SceneToday";
import { SceneRisk } from "./act1/SceneRisk";
import { SceneStats } from "./act1/SceneStats";
import { FULL_FRAMES, FullVideo } from "./FullVideo";

// studies: fill in the real PubMed count; showCaptions: burn the voice-over lines in as captions
const act1Props = { studies: "10,000", showCaptions: true }; // PubMed: 10,484 papers with "96-well" in title/abstract (2026-10-24)

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Full" component={FullVideo} width={1920} height={1080} fps={30} durationInFrames={FULL_FRAMES} defaultProps={act1Props} />
    <Folder name="Act1-Scenes">
      <Composition id="Act1-Scale" component={SceneScale} width={1920} height={1080} fps={30} durationInFrames={300} defaultProps={act1Props} />
      <Composition id="Act1-Today" component={SceneToday} width={1920} height={1080} fps={30} durationInFrames={TODAY_FRAMES} defaultProps={act1Props} />
      <Composition id="Act1-Risk" component={SceneRisk} width={1920} height={1080} fps={30} durationInFrames={120} defaultProps={act1Props} />
      <Composition id="Act1-Stats" component={SceneStats} width={1920} height={1080} fps={30} durationInFrames={360} defaultProps={act1Props} />
    </Folder>
    <Composition id="Act1" component={Act1} width={1920} height={1080} fps={30} durationInFrames={300 + TODAY_FRAMES + 120} defaultProps={act1Props} />
  </>
);
