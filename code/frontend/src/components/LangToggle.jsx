import { useLang } from "../i18n/LangContext.jsx";

export default function LangToggle() {
  const { lang, setLang, t } = useLang();
  const isEs = lang === "es";

  return (
    <button
      className="lang-toggle"
      onClick={() => setLang(isEs ? "en" : "es")}
      aria-label={t("lang.toggle")}
      title={t("lang.toggle")}
    >
      <span className={`lang-flag ${!isEs ? "inactive" : ""}`}>🇪🇸</span>
      <div className="toggle-track">
        <div className={`toggle-thumb ${!isEs ? "right" : ""}`} />
      </div>
      <span className={`lang-flag ${isEs ? "inactive" : ""}`}>🇬🇧</span>
    </button>
  );
}
