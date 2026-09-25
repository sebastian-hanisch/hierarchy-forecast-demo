"""Konstanten der Hierarchie-Demo: Vehikel "Tagesaufträge einer Depot-Hierarchie" (Stück 8 der Zeitreihen-Prognose-Linie), Abstimmungsverfahren, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095
FIRST_TEST = 730
FIRST_ORIGIN = 91
LEVEL = 100.0
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
EVENTS = 0.5
SEASON_PERIOD = 7
AR_PHI = 0.98                 # Rückkehr der Niveauschwankungen (Region, Netz, Depot) zum Mittel
DEPOT_SWING = 0.03            # Streuung (stationär, im Log) der depoteigenen Niveauschwankung
NET_SHARE = 0.5               # Netz-Schwankung als Anteil der Regions-Schwankung

# --- Glättung aus Stück 2 (Basisprognose) ------------------------------------------------------------------------------------------------------
ETS_MODELS = {"hw_mult": ("add", "mul")}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Regler und Voreinstellungen ------------------------------------------------------------------------------------------------------------------
DEPOTS_MIN, DEPOTS_MAX, DEPOTS_STEP, DEFAULT_DEPOTS = 20, 100, 10, 30
REGIONS_MIN, REGIONS_MAX, DEFAULT_REGIONS = 2, 8, 5
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.04, 0.4, 0.02, 0.14
TREND_MEAN = 10.0
RHO_MIN, RHO_MAX, RHO_STEP, DEFAULT_RHO = 0.0, 0.8, 0.1, 0.3
SWING_MIN, SWING_MAX, SWING_STEP, DEFAULT_SWING = 0.0, 0.2, 0.02, 0.06
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
HISTORY_MIN, HISTORY_MAX, HISTORY_STEP, DEFAULT_HISTORY = 60, 600, 30, 600
SELECT_DAYS = 90               # Auswahlfenster der Modellwahl je Knoten: Fehler der letzten Trainingstage
MODELS = ("hw_mult", "regression", "snaive_k")
BASES = ("auto",) + MODELS
BASE_NAMES = {"auto": "Modellwahl je Knoten (das beste der drei auf den letzten 90 Trainingstagen)", "hw_mult": "Holt-Winters multiplikativ (Stück 2)", "regression": "Regression auf Kalender und Aktionsanteil (Stück 4)",
              "snaive_k": "Wochenmittel (Stück 1)"}
BASE_SHORT = {"auto": "Modellwahl", "hw_mult": "Holt-Winters", "regression": "Regression", "snaive_k": "Wochenmittel"}

# Abstimmungsverfahren (Reihenfolge = Anzeige)
METHODS = ("base", "bu", "td", "mo", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample")
METHOD_NAMES = {"base": "Basisprognose (nicht abgestimmt)", "bu": "Bottom-up", "td": "Top-down (mittlere Anteile)", "mo": "Middle-out (Regionen)", "ols": "OLS", "wls_struct": "WLS (Struktur)",
                "wls_var": "WLS (Fehlervarianz)", "mint_shrink": "MinT (geschrumpft)", "mint_sample": "MinT (Stichprobe)"}
METHOD_SHORT = {"base": "Basis", "bu": "Bottom-up", "td": "Top-down", "mo": "Middle-out", "ols": "OLS", "wls_struct": "WLS Struktur", "wls_var": "WLS Varianz", "mint_shrink": "MinT geschrumpft", "mint_sample": "MinT Stichprobe"}
LEVELS = ("Netz", "Regionen", "Depots")

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(3))
EXP_DEPOTS = 30
SWING_LEVELS = (0.0, 0.06, 0.12)
RHO_LEVELS = (0.0, 0.4, 0.8)
HISTORY_LEVELS = (60, 120, 240, 600)
HORIZON_LEVELS = (1, 7, 14, 28)
EXP_HISTORY_DEPOTS = 60
