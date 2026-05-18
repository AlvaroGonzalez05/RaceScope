import { useLang } from "../i18n/LangContext.jsx";

export default function ThemeToggle({ theme, onToggle }) {
  const { t } = useLang();
  return (
    <button className="theme-toggle" onClick={onToggle} aria-label={t("lang.toggle")}>
      <span>{theme === "dark" ? t("theme.dark") : t("theme.light")}</span>
      <div className="toggle-track">
        <div className={`toggle-thumb ${theme === "dark" ? "right" : ""}`} />
      </div>
    </button>
  );
}
