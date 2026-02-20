export default function ThemeToggle({ theme, onToggle }) {
  return (
    <button className="theme-toggle" onClick={onToggle} aria-label="Cambiar tema">
      <span>{theme === "dark" ? "Oscuro" : "Claro"}</span>
      <div className="toggle-track">
        <div className={`toggle-thumb ${theme === "dark" ? "right" : ""}`} />
      </div>
    </button>
  );
}
