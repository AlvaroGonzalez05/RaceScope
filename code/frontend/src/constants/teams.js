// Canonical F1 team colours used for row tinting.
export const TEAM_COLORS = {
  "Red Bull Racing": "#1f6cff",
  "Ferrari":         "#ff3b30",
  "Mercedes":        "#00d6c7",
  "McLaren":         "#ff8a00",
  "Aston":           "#2db56f",
  "Alpine":          "#ff4d94",
  "Williams":        "#4fa2ff",
  "RB":              "#3355ff",
  "Sauber":          "#6dcf38",
  "Haas":            "#bfc7d1",
};

export function teamTint(teamName) {
  const key = Object.keys(TEAM_COLORS).find((k) => teamName?.includes(k));
  return key ? TEAM_COLORS[key] : "#6f7a86";
}
