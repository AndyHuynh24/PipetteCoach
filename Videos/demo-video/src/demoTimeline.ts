// Made from the app's own screen (Tip box colour, progress bar) - see make_layers.py / the edit notes
export const DEMO_REAL_S = 36.00;
export const DEMO_S = 79.20;
export const DEMO_SOUNDS: [number, "go" | "warn" | "dispensed"][] = [[1.3, "warn"], [2.5, "warn"], [4.6, "go"], [6.1, "dispensed"], [10.8, "go"], [12.7, "dispensed"], [15.7, "go"], [17.7, "dispensed"], [19.3, "go"], [21.7, "go"], [23.3, "dispensed"], [25.6, "go"], [27.0, "go"], [29.6, "go"], [33.8, "warn"], [34.0, "go"], [36.3, "dispensed"], [39.43, "dispensed"], [40.55, "dispensed"], [41.7, "dispensed"], [43.0, "dispensed"], [46.35, "dispensed"], [50.45, "dispensed"], [57.7, "dispensed"], [59.15, "dispensed"], [65.5, "dispensed"], [66.95, "dispensed"], [71.5, "dispensed"]];
export const DEMO_CALLOUTS: [number, "go" | "warn"][] = [[1.3, "warn"], [2.5, "warn"], [4.6, "go"], [10.8, "go"], [15.7, "go"], [19.3, "go"], [21.7, "go"], [25.6, "go"], [27.0, "go"], [29.6, "go"], [33.8, "warn"], [34.0, "go"]];
export const DEMO_CAPTIONS: [number, number, string][] = [
 [
  0,
  3.6,
  "Stop over the wrong well: it warns me before a drop goes in."
 ],
 [
  3.6,
  4.6,
  "The camera follows my tip, well by well."
 ],
 [
  4.6,
  6.1,
  "The right well: it gives me the go."
 ],
 [
  6.1,
  10.6,
  "I dispense, and it's checked off. No pen, no paper."
 ],
 [
  12.1,
  29.0,
  "Right well, go. Dispense, checked off. Well after well."
 ],
 [
  29.0,
  33.8,
  "Each well keeps its own sequence, and its timer starts by itself."
 ],
 [
  33.8,
  36.0,
  "Wrong order? It warns me first."
 ],
 [
  36.0,
  78.7,
  "Every drop is checked and logged: which well, what substance, and when."
 ]
];
