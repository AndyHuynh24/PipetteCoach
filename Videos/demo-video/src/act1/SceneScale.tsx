import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Interactive, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Caption, Stat } from "./style";

export type Act1Props = { studies: string; showCaptions: boolean };

/** 0:00-0:10 How big it is: the bench set up (SOURCE 1), then a close dispense (SOURCE 2, cycle 2). */
export const SceneScale: React.FC<Act1Props> = ({ studies, showCaptions }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Video name="Bench set up" src={staticFile("source1.mp4")} muted trimBefore={0} durationInFrames={157} />
      <AbsoluteFill
        style={{
          transformOrigin: "760px 560px",
          scale: interpolate(frame, [157, 300], [1.45, 1.75], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        }}
      >
        <Video name="Close dispense" src={staticFile("source2.mp4")} muted trimBefore={420} from={157} durationInFrames={143} />
      </AbsoluteFill>
      <AbsoluteFill
        style={{ background: "linear-gradient(90deg, rgba(11,18,32,0.75) 0%, rgba(11,18,32,0) 55%)" }}
      />
      <Interactive.Div name="Stats" style={{ position: "absolute", left: 140, top: 200, display: "flex", flexDirection: "column", gap: 70 }}>
        <Stat value="~$2B / year" label="spent on these plates" at={75} />
        <Stat value={`${studies}+ studies`} label="drug discovery · tests · genetics (PubMed)" at={190} />
      </Interactive.Div>
      <Caption show={showCaptions} from={0} to={150} text="This is a pipette and a 96-well plate." />
      <Caption
        show={showCaptions}
        from={150}
        to={300}
        text={`Labs spend nearly $2 billion a year on these plates, used in over ${studies} published studies.`}
      />
    </AbsoluteFill>
  );
};
