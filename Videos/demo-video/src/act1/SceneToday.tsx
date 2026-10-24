import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Easing, Interactive, interpolate, staticFile, useCurrentFrame } from "remotion";
import { COLORS, Caption, fontFamily } from "./style";
import { Act1Props } from "./SceneScale";

const Word: React.FC<{ text: string; at: number; color?: string }> = ({ text, at, color }) => {
  const frame = useCurrentFrame();
  return (
    <Interactive.Div
      name={text}
      style={{
        fontFamily,
        fontWeight: 800,
        fontSize: 130,
        color: color ?? COLORS.white,
        textShadow: "0 6px 30px rgba(0,0,0,0.55)",
        opacity: interpolate(frame, [at, at + 4], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        translate: interpolate(frame, [at, at + 10], ["0px 40px", "0px 0px"], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.16, 1, 0.3, 1),
        }),
      }}
    >
      {text}
    </Interactive.Div>
  );
};

export const FAST_AT = 255; // the first cycle (source frames 645-900) ends here
export const TODAY_FRAMES = FAST_AT + 52; // + public/today_fast.mp4 (source 30 s to the end at 25x)

/** How it's done today: the first cycle at normal speed, then the rest of SOURCE 2 very fast (25x). */
export const SceneToday: React.FC<Act1Props> = ({ showCaptions }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Video name="First cycle, normal speed" src={staticFile("source2.mp4")} muted trimBefore={645} durationInFrames={FAST_AT} />
      {/* the rest at 25x, pre-sped with ffmpeg (a 25x playbackRate drops to black): ffmpeg -ss 30 -i public/source2.mp4
          -vf "setpts=PTS/25,fps=30" -an public/today_fast.mp4 */}
      <Video name="The rest at 25x" src={staticFile("today_fast.mp4")} muted from={FAST_AT} durationInFrames={TODAY_FRAMES - FAST_AT} />
      <AbsoluteFill style={{ background: "linear-gradient(0deg, rgba(11,18,32,0.0) 55%, rgba(11,18,32,0.7) 100%)" }} />
      <Interactive.Div name="Words" style={{ position: "absolute", left: 140, right: 140, top: 110, display: "flex", gap: 48 }}>
        <Word text="Fill." at={8} />
        <Word text="Stop." at={95} />
        <Word text="Write." at={150} />
        <Word text="Repeat." at={FAST_AT} color={COLORS.alert} />
      </Interactive.Div>
      <Interactive.Div
        name="Speed badge"
        style={{
          position: "absolute",
          right: 120,
          bottom: 230,
          fontFamily,
          fontWeight: 800,
          fontSize: 72,
          color: COLORS.white,
          background: "rgba(11,18,32,0.7)",
          padding: "8px 26px",
          borderRadius: 16,
          opacity: interpolate(frame, [0, 6], [0, 1], { extrapolateRight: "clamp" }),
        }}
      >
        {frame < FAST_AT ? "Real speed" : "×25"}
      </Interactive.Div>
      <Caption
        show={showCaptions}
        from={0}
        to={TODAY_FRAMES}
        text="And it's still tracked by hand: fill, stop, write it down, find your place, repeat."
      />
    </AbsoluteFill>
  );
};
