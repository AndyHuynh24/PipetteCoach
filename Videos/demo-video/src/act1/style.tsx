import React from "react";
import { loadFont } from "@remotion/google-fonts/Inter";
import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";

export const { fontFamily } = loadFont("normal", { weights: ["500", "700", "800"], subsets: ["latin"] });

export const COLORS = { ink: "#0B1220", accent: "#3B82F6", alert: "#EF4444", white: "#FFFFFF" };

/** Voice-over line as a caption, bottom centre (for viewers without sound, and to time the recording). */
export const Caption: React.FC<{ text: string; from: number; to: number; show: boolean }> = ({ text, from, to, show }) => {
  const frame = useCurrentFrame();
  if (!show) return null;
  return (
    <Interactive.Div
      name="Caption"
      style={{
        position: "absolute",
        left: 160,
        right: 160,
        bottom: 90,
        display: "flex",
        justifyContent: "center",
        opacity: interpolate(frame, [from, from + 6, to - 6, to], [0, 1, 1, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
      }}
    >
      <div
        style={{
          fontFamily,
          fontWeight: 500,
          fontSize: 46,
          lineHeight: 1.3,
          color: COLORS.white,
          background: "rgba(11, 18, 32, 0.72)",
          padding: "16px 32px",
          borderRadius: 18,
          textAlign: "center",
        }}
      >
        {text}
      </div>
    </Interactive.Div>
  );
};

/** A number + label that pops in at `at` (frames). */
export const Stat: React.FC<{ value: string; label: string; at: number; style?: React.CSSProperties }> = ({
  value,
  label,
  at,
  style,
}) => {
  const frame = useCurrentFrame();
  return (
    <Interactive.Div
      name={`Stat ${value}`}
      style={{
        fontFamily,
        color: COLORS.white,
        opacity: interpolate(frame, [at, at + 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        scale: interpolate(frame, [at, at + 14], [0.85, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.16, 1, 0.3, 1),
        }),
        ...style,
      }}
    >
      <div style={{ fontSize: 120, fontWeight: 800, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 50, fontWeight: 500, marginTop: 10, opacity: 0.92 }}>{label}</div>
    </Interactive.Div>
  );
};
