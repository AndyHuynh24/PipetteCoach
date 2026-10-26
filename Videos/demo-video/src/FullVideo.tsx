import React from "react";
import { Audio, Video } from "@remotion/media";
import { AbsoluteFill, Easing, interpolate, Sequence, Series, staticFile, useCurrentFrame } from "remotion";
import { Act1 } from "./act1/Act1";
import { Act1Props } from "./act1/SceneScale";
import { SceneStats } from "./act1/SceneStats";
import { TODAY_FRAMES } from "./act1/SceneToday";
import { Caption, fontFamily } from "./act1/style";
import { DEMO_CALLOUTS, DEMO_CAPTIONS, DEMO_S, DEMO_SOUNDS } from "./demoTimeline";

// public/live_demo.mp4 (camera + app, dead time cut) and public/layers/*.mp4 (make_layers.py) are made outside Remotion;
// public/sfx/*.wav are the app's own sounds (make_sfx.py)
const F = 30;
export const NOTE_FRAMES = 60;
export const ACT1_FRAMES = 300 + TODAY_FRAMES + 120;
export const DEMO_FRAMES = Math.round(DEMO_S * F);
const LAYERS: [string, string, string][] = [
  ["camera", "One webcam, looking down", "No special pipette, no extra hardware. Just a camera above the bench."],
  ["plate", "Layer 1 · Find the plate", "AI finds the plate in every frame, and keeps up when it moves."],
  ["wells", "Layer 2 · Map all 96 wells", "Every well gets its exact position: A1 to H12."],
  ["tip", "Layer 3 · Track the pipette tip", "A second model follows the tip and finds the well under it."],
  ["dwell", "Layer 4 · Hold = dispense", "The tip holds on one well for 1.5 s: that well was filled. Then it's checked against your plan."],
];
const LAYER_FRAMES = 180;
export const HOW_FRAMES = LAYERS.length * LAYER_FRAMES;
export const FULL_FRAMES = NOTE_FRAMES + ACT1_FRAMES + DEMO_FRAMES + HOW_FRAMES + 360;

const Note: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: "#0B1220", justifyContent: "center", alignItems: "center", padding: 160 }}>
    <div style={{ fontFamily, color: "white", fontSize: 54, fontWeight: 700, textAlign: "center", lineHeight: 1.35 }}>
      Heads up: this video was edited in a last-minute rush, so expect a few rough edits.
      <div style={{ color: "#9FF2D9", marginTop: 28, fontSize: 46, fontWeight: 600 }}>
        The real product runs a lot smoother.
      </div>
    </div>
  </AbsoluteFill>
);

// the app's Tip box in the demo layout (app at x 740, y 180, 1140 px wide)
const TIP_BOX = { left: 1592, top: 248, width: 238, height: 90 };
const Callout: React.FC<{ kind: "go" | "warn" }> = ({ kind }) => {
  const frame = useCurrentFrame();
  const col = kind === "go" ? "#16A34A" : "#DC2626";
  const pop = interpolate(frame, [0, 8], [0.6, 1], { extrapolateRight: "clamp", easing: Easing.bezier(0.16, 1, 0.3, 1) });
  const op = interpolate(frame, [0, 4, 30, 36], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ opacity: op }}>
      <div style={{ position: "absolute", ...TIP_BOX, border: `7px solid ${col}`, borderRadius: 16,
        boxShadow: `0 0 40px ${col}` }} />
      <div style={{ position: "absolute", left: TIP_BOX.left - 200, top: TIP_BOX.top + TIP_BOX.height + 18, width: 440,
        display: "flex", justifyContent: "center", transform: `scale(${pop})` }}>
        <div style={{ fontFamily, fontWeight: 800, fontSize: 40, color: "white", background: col, padding: "10px 26px",
          borderRadius: 999, boxShadow: "0 8px 30px rgba(0,0,0,0.35)", whiteSpace: "nowrap" }}>
          {kind === "go" ? "✓ Right well" : "⚠ Wrong well: warned before the drop"}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const LiveDemo: React.FC<{ showCaptions: boolean }> = ({ showCaptions }) => (
  <AbsoluteFill>
    <Video src={staticFile("live_demo.mp4")} muted />
    {DEMO_CALLOUTS.map(([t, kind]) => (
      <Sequence key={`c${t}`} from={Math.round(t * F)} durationInFrames={36} layout="none">
        <Callout kind={kind} />
      </Sequence>
    ))}
    {DEMO_SOUNDS.map(([t, name]) => (
      <Sequence key={`s${t}${name}`} from={Math.round(t * F)} durationInFrames={30} layout="none">
        <Audio src={staticFile(`sfx/${name}.wav`)} volume={name === "dispensed" && t > 36 ? 0.5 : 1} />
      </Sequence>
    ))}
    {DEMO_CAPTIONS.map(([a, b, text]) => (
      <Caption key={text} show={showCaptions} from={Math.round(a * F)} to={Math.round(b * F)} text={text} />
    ))}
  </AbsoluteFill>
);

const Layer: React.FC<{ name: string; title: string; text: string }> = ({ name, title, text }) => {
  const frame = useCurrentFrame();
  const slide = interpolate(frame, [0, 14], [-40, 0], { extrapolateRight: "clamp", easing: Easing.bezier(0.16, 1, 0.3, 1) });
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Video src={staticFile(`layers/${name}.mp4`)} muted />
      <AbsoluteFill style={{ background: "linear-gradient(180deg, rgba(8,30,28,0.85) 0%, rgba(8,30,28,0) 30%)" }} />
      <div style={{ position: "absolute", left: 70, top: 50, right: 70, transform: `translateY(${slide}px)`, fontFamily, color: "white" }}>
        <div style={{ fontSize: 30, fontWeight: 700, color: "#9FF2D9", letterSpacing: 2 }}>HOW IT WORKS</div>
        <div style={{ fontSize: 76, fontWeight: 800 }}>{title}</div>
        <div style={{ fontSize: 40, fontWeight: 500, marginTop: 8, maxWidth: 1500 }}>{text}</div>
      </div>
    </AbsoluteFill>
  );
};

/** The whole video: a short note, Act 1, live demo, how it works (layer by layer), then the three stats. */
export const FullVideo: React.FC<Act1Props> = (props) => (
  <Series>
    <Series.Sequence name="Note" durationInFrames={NOTE_FRAMES}>
      <Note />
    </Series.Sequence>
    <Series.Sequence name="Act 1" durationInFrames={ACT1_FRAMES}>
      <Act1 {...props} />
    </Series.Sequence>
    <Series.Sequence name="Live demo" durationInFrames={DEMO_FRAMES}>
      <LiveDemo showCaptions={props.showCaptions} />
    </Series.Sequence>
    {LAYERS.map(([name, title, text]) => (
      <Series.Sequence key={name} name={`How it works: ${name}`} durationInFrames={LAYER_FRAMES}>
        <Layer name={name} title={title} text={text} />
      </Series.Sequence>
    ))}
    <Series.Sequence name="Why it matters (stats)" durationInFrames={360}>
      <SceneStats {...props} />
    </Series.Sequence>
  </Series>
);
