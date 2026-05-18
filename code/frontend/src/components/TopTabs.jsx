import { useLang } from "../i18n/LangContext.jsx";

const TABS = ["home", "pre-race", "live", "rewatch"];

export default function TopTabs({ activeTab, onChange }) {
  const { t } = useLang();
  return (
    <nav className="top-tabs" aria-label="Secciones" role="tablist">
      {TABS.map((tab) => (
        <button
          key={tab}
          role="tab"
          aria-selected={activeTab === tab}
          className={activeTab === tab ? "active" : ""}
          onClick={() => onChange(tab)}
        >
          {t(`tabs.${tab}`)}
        </button>
      ))}
    </nav>
  );
}
