import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Caption } from "./style";
import { Act1Props } from "./SceneScale";

/** 0:16-0:20 The risk: slow push in on the tip over the plate (SOURCE 2, ~32 s, half speed). */
export const SceneRisk: React.FC<Act1Props> = ({ showCaptions }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <AbsoluteFill
        style={{
          transformOrigin: "905px 680px",
          scale: interpolate(frame, [0, 120], [1.9, 2.7], { extrapolateRight: "clamp" }),
        }}
      >
        <Video name="Tip over the plate" src={staticFile("source2.mp4")} muted trimBefore={960} playbackRate={0.5} durationInFrames={120} />
      </AbsoluteFill>
      <AbsoluteFill
        style={{
          background: "radial-gradient(circle at 50% 50%, rgba(239,68,68,0) 35%, rgba(239,68,68,0.55) 100%)",
          opacity: interpolate(frame, [0, 20, 50, 70, 100, 120], [0, 0.8, 0.35, 0.9, 0.4, 0.9]),
        }}
      />
      <Caption show={showCaptions} from={0} to={120} text="One wrong well can ruin the whole experiment, and you won't know for days." />
    </AbsoluteFill>
  );
};
