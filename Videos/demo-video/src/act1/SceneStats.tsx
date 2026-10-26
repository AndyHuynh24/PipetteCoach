import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Easing, Interactive, interpolate, staticFile, useCurrentFrame } from "remotion";
import { COLORS, Caption, Stat, fontFamily } from "./style";
import { Act1Props } from "./SceneScale";

/** 0:20-0:32 Why it matters: three stats pop in as each is said, then "So we fixed it." */
export const SceneStats: React.FC<Act1Props> = ({ showCaptions }) => {
  const frame = useCurrentFrame();
  const cardsOut = interpolate(frame, [296, 310], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.ink }}>
      <Video
        name="Writing it down (background)"
        src={staticFile("source2.mp4")}
        muted
        trimBefore={1770}
        durationInFrames={360}
        style={{ filter: "blur(18px) brightness(0.35)", scale: 1.08 }}
      />
      <Interactive.Div
        name="Stats"
        style={{
          position: "absolute",
          left: 160,
          right: 160,
          top: 170,
          display: "flex",
          flexDirection: "column",
          gap: 64,
          opacity: cardsOut,
        }}
      >
        <Stat value="🏥 Up to 70%" label="of lab errors happen in manual steps" at={20} />
        <Stat value="🧪 $28B / year" label="lost to irreproducible US research" at={125} />
        <Stat value="⏱️ 2 people per plate" label="or a $2,000+ tracker with special pipettes" at={230} />
      </Interactive.Div>
      <Interactive.Div
        name="So we fixed it"
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily,
          fontWeight: 800,
          fontSize: 170,
          color: COLORS.white,
          opacity: interpolate(frame, [306, 316], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
          scale: interpolate(frame, [306, 330], [0.9, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        So we fixed it.
      </Interactive.Div>
      <Caption show={showCaptions} from={0} to={105} text="Up to 70% of lab errors happen in manual steps like this." />
      <Caption show={showCaptions} from={105} to={210} text="Research that can't be reproduced costs the US an estimated $28 billion a year." />
      <Caption
        show={showCaptions}
        from={210}
        to={300}
        text="And today's fix? Two people per plate, or a $2,000 tracker that needs special pipettes."
      />
    </AbsoluteFill>
  );
};
