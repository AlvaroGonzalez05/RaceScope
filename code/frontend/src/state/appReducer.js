export const SET_THEME = "SET_THEME";
export const SET_ACTIVE_TAB = "SET_ACTIVE_TAB";
export const SET_METADATA_STATUS = "SET_METADATA_STATUS";
export const SET_METADATA_ERROR = "SET_METADATA_ERROR";
export const SET_SEASONS = "SET_SEASONS";
export const SET_SEASON = "SET_SEASON";
export const SET_CIRCUITS = "SET_CIRCUITS";
export const SET_CIRCUIT_ID = "SET_CIRCUIT_ID";
export const SET_DRIVERS = "SET_DRIVERS";
export const SET_TEAMS = "SET_TEAMS";
export const SET_ROWS = "SET_ROWS";
export const SET_RUNNING = "SET_RUNNING";
export const SET_MOBILE_LAYOUT = "SET_MOBILE_LAYOUT";

export const initialState = {
  theme: "dark",
  activeTab: "home",
  metadataStatus: "loading",
  metadataError: "",
  seasons: [],
  season: 0,
  circuits: [],
  circuitId: "",
  drivers: [],
  teams: [],
  rows: [],
  running: false,
  isMobileLayout: false,
};

export function appReducer(state, action) {
  switch (action.type) {
    case SET_THEME:           return { ...state, theme: action.payload };
    case SET_ACTIVE_TAB:      return { ...state, activeTab: action.payload };
    case SET_METADATA_STATUS: return { ...state, metadataStatus: action.payload };
    case SET_METADATA_ERROR:  return { ...state, metadataError: action.payload };
    case SET_SEASONS:         return { ...state, seasons: action.payload };
    case SET_SEASON:          return { ...state, season: action.payload };
    case SET_CIRCUITS:        return { ...state, circuits: action.payload };
    case SET_CIRCUIT_ID:      return { ...state, circuitId: action.payload };
    case SET_DRIVERS:         return { ...state, drivers: action.payload };
    case SET_TEAMS:           return { ...state, teams: action.payload };
    case SET_ROWS:            return { ...state, rows: action.payload };
    case SET_RUNNING:         return { ...state, running: action.payload };
    case SET_MOBILE_LAYOUT:   return { ...state, isMobileLayout: action.payload };
    default:                  return state;
  }
}
