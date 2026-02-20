const TABS = ["home", "pre-race", "live", "rewatch", "explore"];

const labels = {
  home: "Home",
  "pre-race": "Pre-race",
  live: "Live",
  rewatch: "Rewatch",
  explore: "Explore",
};

export default function TopTabs({ activeTab, onChange }) {
  return (
    <nav className="top-tabs" aria-label="Secciones">
      {TABS.map((tab) => (
        <button
          key={tab}
          className={activeTab === tab ? "active" : ""}
          onClick={() => onChange(tab)}
        >
          {labels[tab]}
        </button>
      ))}
    </nav>
  );
}
