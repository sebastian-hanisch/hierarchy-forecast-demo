# 🌳 Hierarchische Abstimmung – Prognosen, die sich addieren

**[→ Demo live ausprobieren](https://sebastianhanisch-hierarchy-forecast-demo.streamlit.app/)**

Achtes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Nachfolger der [Prognoseintervalle](https://github.com/sebastian-hanisch/forecast-interval-demo): dort ging es um die Unsicherheit einer Prognose, hier um Prognosen, die **zusammenpassen**.
Geplant sind drei weitere Stücke (Kombination, Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Ein Netz von Depots, in Regionen gegliedert: **jeder Knoten bekommt seine eigene Prognose** – das Netz, jede Region, jedes Depot. Die Prognosen widersprechen sich: die Summe der Depot-Prognosen ist nicht die Prognose der Region. **Hierarchische Abstimmung** macht aus den Basisprognosen kohärente, die sich exakt addieren:
**Bottom-up**, **Top-down**, **Middle-out** und die Verfahren, die alle Knoten zusammen verwenden – **OLS, WLS und MinT** (kleinste Fehlervarianz, Wickramasuriya et al. 2019; Stichprobe oder zur Diagonalen geschrumpft). Die Demo misst auf einem erzeugten Depot-Netz, **was das an Genauigkeit bringt**: je Ebene, je Horizont, bei wachsender gemeinsamer Bewegung und bei wenig Fehlerhistorie.
Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (`scipy` nur als Gegenprobe im Test).

**Bezug zu OR:** Netzplanung braucht Zahlen, die zusammenpassen – die Kapazität der Region muss die Summe der Depots tragen. Die Frage ist, welche Ebene dabei den Ton angibt.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Kohärenz kostet nichts und die Abstimmung verbessert die Genauigkeit." Die Messung sagt: **Kohärenz ist der Gewinn; die Genauigkeit ändert sich um wenige Prozent, und die Wahl der Methode zählt weniger als die Frage, ob die Basisprognosen überhaupt widersprüchlich sind.**

1. **Was zu wenig widersprüchlich ist, gewinnt nichts.** Das Wochenmittel ist linear und damit schon exakt kohärent: Bottom-up, OLS, WLS und MinT ändern nichts (Ø-RMSSE 0,941). Holt-Winters je Knoten ist fast kohärent (die Netz-Prognose weicht 0,8 % von der Summe der Depots ab): alle vier liegen zwischen −0,2 und +0,5 % der Basis (0,871).
   Erst die **Modellwahl je Knoten** (das beste von Holt-Winters, Regression und Wochenmittel auf den letzten 90 Trainingstagen) macht die Basisprognosen verschieden (Abweichung 1,5 %, bei starker gemeinsamer Bewegung 4,0 %).
2. **Top-down und Middle-out verlieren deutlich.** Sie verteilen nach mittleren Anteilen der Trainingstage: im Standardfall +23,1 % (Top-down) und +21,3 % (Middle-out) gegen die Basis, die Depot-Ebene liegt bei 1,35 statt 0,88. In allen Experimenten liegen sie zwischen +6 und +43 % schlechter.
3. **Bottom-up ist kein Selbstläufer und MinT auch nicht.** Im Standardfall (Ø-RMSSE der Basis 0,834) gewinnt **WLS (Struktur)** mit 0,814 (−2,4 %); OLS −1,2 %; **Bottom-up +2,0 % und MinT geschrumpft +1,5 % sind schlechter als die Basis**. MinT nimmt die Fehlerkovarianz aus **Ein-Schritt-Fehlern** (im Training, in der Stichprobe); bei Horizont 14 stimmt sie nur ungefähr.
4. **Die Gewinne kommen bei gemeinsamer Bewegung.** Bei starker gemeinsamer regionaler Niveauschwankung (0,12) liegt MinT geschrumpft bei −6,3 % (Mittel über drei Seeds), WLS (Struktur) bei −4,6 %, Bottom-up nur bei −1,9 %; ohne Schwankung ist Bottom-up mit −3,4 % vorn. Der gemeinsame **Tagesschock** ändert an den Gewinnen wenig (alle unter 2 %).
5. **Die Gewinne schrumpfen mit dem Horizont.** Bei Horizont 1 gewinnt WLS (Fehlervarianz) 1,7 %, bei 28 Tagen 0,6 %.
6. **Die Stichprobenkovarianz bricht zusammen.** 60 Depots (66 Knoten), Fehlerkovarianz aus 60 Tagen: MinT (Stichprobe) erreicht einen Ø-RMSSE von 36,8 gegen 1,008 der Basis (die Matrix hat höchstens Rang 60 bei 66 Knoten; der genaue Wert einer singulären Inversion hängt vom System ab, die Größenordnung nicht). MinT geschrumpft bleibt bei 0,985 (Schrumpfung λ = 0,41); ab 120 Tagen ist auch die Stichprobe brauchbar (1,017).

## Modell

- **Das Netz** (`hrc_scenario.py`): 1 095 Tage, Depots wie in den Vorgängern (multiplikativ: Niveau, Trend, Wochen- und Jahresmuster, Feiertage, Aktionen, log-normales Rauschen, eigene Parameter je Depot), in gleich große Regionen gegliedert (höchstens Depots / 2 Regionen). Zwei Quellen gemeinsamer Bewegung: **Niveauschwankungen** (mittelwertrückkehrende Zufallsgänge, $\varphi = 0{,}98$, im Log) je Region (Regler), je Netz (halb so stark) und je Depot (klein, fest),
  und der **Tagesschock** (Anteil $\rho$ des Rauschens, den sich alle Depots einer Region teilen). Reihen aggregieren durch Summen: Region = Summe ihrer Depots, Netz = Summe der Regionen.
- **Basisprognosen** (`hrc_baselines.py`, `hrc_ets.py`, `hrc_evaluation.py`): je Knoten Holt-Winters multiplikativ (Stück 2), Regression im Log auf Kalender und Aktionsanteil (Stück 4; der Aktionsanteil eines Knotens ist das nach Depot-Größe gewichtete Mittel der Depot-Aktionen), Wochenmittel (Stück 1) – oder **Modellwahl je Knoten**: das Modell mit dem kleinsten quadratischen Fehler auf den letzten 90 Trainingstagen (mit Parametern aus allen Trainingstagen, also nicht ehrlich außerhalb der Stichprobe).
- **Abstimmung** (`hrc_reconcile.py`): $\tilde y = S\,G\,\hat y$. Bottom-up $G = (0 \mid I)$; Top-down $G = (p \mid 0)$ mit mittleren Tagesanteilen der Trainingstage; Middle-out mit den mittleren Anteilen an der Region; OLS, WLS und MinT als $G = (S' W^{-1} S)^{-1} S' W^{-1}$ mit $W = I$, $\operatorname{diag}(S\mathbf 1)$ (Zahl der Depots), $\operatorname{diag}(\hat\sigma^2)$, der Stichprobenkovarianz der Ein-Schritt-Fehler und
  ihrer zur Diagonalen **geschrumpften** Fassung (Schäfer/Strimmer: $\lambda = \sum_{i\ne j}\widehat{\operatorname{Var}}(r_{ij}) / \sum_{i\ne j} r_{ij}^2$). Negative abgestimmte Werte werden auf 0 gesetzt.
- **Kennzahl** (`hrc_evaluation.py`): **RMSSE** je Knoten (Wurzel des mittleren quadratischen Fehlers über alle Ursprünge des Testjahres und Horizonte, geteilt durch den saisonal naiven Trainingsfehler), je Ebene (Netz, Regionen, Depots) gemittelt, "Ø Ebenen" der Mittelwert der drei Ebenen.

## Methodik

- **Handrechnungen:** die Summationsmatrix; Bottom-up, Top-down und Middle-out auf drei Knoten; **OLS** auf $S = \begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix}$ ($G = \tfrac13\begin{pmatrix}1&2&-1\\1&-1&2\end{pmatrix}$, Basisprognose (10, 3, 4) → (9, 4, 5)) und **WLS (Struktur)** ($G = \begin{pmatrix}.25&.75&-.25\\.25&-.25&.75\end{pmatrix}$); RMSSE und Ebenenmittel.
- **Gegenprobe:** die verallgemeinerte Kleinste-Quadrate-Lösung gegen `numpy.linalg.lstsq` auf dem gewichteten Problem; die Schrumpfung gegen eine **unabhängige Schleife** nach der Formel von Schäfer/Strimmer; die Auswahl je Knoten gegen eine direkte Rechnung.
- **Eigenschaften:** $G S = I$ für Bottom-up, OLS, WLS und MinT (unverzerrt); alle abgestimmten Prognosen addieren sich (außer in den Zellen, in denen bei 0 abgeschnitten wurde); **MinT minimiert die Spur der Fehlerkovarianz**: keine Störung $N$ mit $N S = 0$ senkt sie; Bottom-up lässt die Depot-Prognosen unverändert, Top-down die Netz-Prognose;
  die Stichprobenkovarianz ist bei weniger Fehlerzeilen als Knoten singulär, die geschrumpfte nicht; die Prognosen eines Ursprungs ändern sich nicht, wenn die Reihen ab dem Ursprung überschrieben werden; das Wochenmittel ist exakt kohärent.
- **Statistik:** die Experimente mitteln über **drei feste Seeds**, die Preset-Zeilen sind **Einzelnetze** (Seed 3); bei der Modellwahl je Knoten kann ein anderer Seed andere Modelle wählen, die Zahlen einzelner Netze schwanken entsprechend (Standardfall: 4 Knoten wählen Holt-Winters, 32 die Regression).
- **Literatur** (nicht nachgebaut): Hyndman/Ahmed/Athanasopoulos/Shang 2011 (optimale Kombination); Wickramasuriya/Athanasopoulos/Hyndman 2019 (MinT); Athanasopoulos et al. 2009 (Top-down); Schäfer/Strimmer 2005 (Schrumpfung); Hyndman/Athanasopoulos, FPP3, Kap. 11.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (Preset, 30 Depots, 5 Regionen, Modellwahl, Seed 3) | Ø-RMSSE: Basis 0,834; **WLS (Struktur) 0,814 (−2,4 %)**, OLS −1,2 %, Bottom-up +2,0 %, MinT geschrumpft +1,5 %, Top-down 1,026 (+23,1 %), Middle-out +21,3 %. Abweichung der Netz-Prognose von der Summe der Depots 1,5 %. | `test_standard_preset` |
| **Starke Bewegung** (Preset, Schwankung 0,12) | Basis 1,013; Bottom-up 0,971 (−4,2 %), WLS (Struktur) −3,8 %, WLS (Fehlervarianz) −6,3 %, **MinT geschrumpft 0,937 (−7,5 %)**, MinT (Stichprobe) −8,7 %; Abweichung 4,0 %. | `test_strong_swing_preset` |
| Nur Holt-Winters (Preset) | Basis 0,871, Abweichung 0,8 %; alle Verfahren außer Top-down und Middle-out zwischen −0,2 und +0,5 %; Top-down 1,104 (+26,7 %), Middle-out 1,028 (+18,0 %). | `test_holt_winters_preset` |
| Wochenmittel (Preset) | Exakt kohärent (Abweichung 0); Bottom-up, OLS, WLS, MinT ändern nichts (0,941); Top-down 1,131 (+20,1 %), Middle-out 1,075 (+14,2 %). | `test_weekly_mean_preset` |
| **Viele Depots, kurze Fehlerhistorie** (Preset, 100 Depots = 106 Knoten, 60 Tage) | MinT (Stichprobe) **über 70** (singulär; 2585 auf Windows, 72,8 auf dem CI-Linux); MinT geschrumpft 0,866 (**−13,8 %**, λ = 0,45), WLS (Fehlervarianz) 0,908 (−9,6 %), Bottom-up 0,916 (−8,8 %) gegen 1,004 der Basis (deren Netz-Ebene liegt bei 1,124). | `test_many_depots_short_history_preset` |
| Kurzer Horizont (Preset, 1 Tag) | Basis 0,823; WLS (Struktur) −2,5 %, WLS (Fehlervarianz) −2,0 %, OLS, Bottom-up und MinT geschrumpft je −1,2 %, MinT (Stichprobe) +0,6 %. | `test_short_horizon_preset` |
| **Gemeinsame Niveauschwankung** (Experiment, 30 Depots, 3 Seeds; Ø Ebenen gegenüber der Basis) | Schwankung 0 / 0,06 / 0,12: Bottom-up −3,4 / 0,0 / −1,9 %; WLS (Fehlervarianz) −2,3 / −0,7 / −4,7 %; MinT geschrumpft −1,9 / −0,3 / **−6,3 %**; WLS (Struktur) −1,5 / −0,7 / −4,6 %; MinT (Stichprobe) +0,1 / +3,6 / −4,4 %; Top-down +42,6 / +26,8 / +15,1 %. Basis 0,750 / 0,906 / 1,149. | `test_structure_experiment_swing` |
| **Gemeinsamer Tagesschock** (Experiment; ρ = 0 / 0,4 / 0,8) | Bottom-up, WLS und MinT geschrumpft liegen bei allen drei Stufen zwischen −1,4 und 0,0 % der Basis (WLS Fehlervarianz −1,4 % bei 0, WLS Struktur −0,8 % bei 0,8); OLS bis +0,3 %, MinT (Stichprobe) +3,5 % bei 0; Top-down +25 bis +29 %. | `test_structure_experiment_rho` |
| **Horizont** (Experiment, 30 Depots; 1 / 7 / 14 / 28 Tage) | Basis 0,889 … 0,908. WLS (Fehlervarianz) −1,7 / −1,0 / −0,7 / −0,6 %; Bottom-up −1,3 / −0,6 / 0,0 / −0,4 %; MinT geschrumpft −0,7 / −0,8 / −0,3 / 0,0 %; MinT (Stichprobe) +2,7 / +3,0 / +3,6 / +4,1 %. | `test_horizon_experiment` |
| **Fehlerhistorie** (Experiment, 60 Depots, 60 / 120 / 240 / 600 Tage; Ø-RMSSE) | Basis 1,008, Bottom-up 0,983, WLS (Fehlervarianz) 0,990 / 0,988 / 0,989 / 0,989; MinT geschrumpft 0,985 / 0,985 / 0,990 / 0,986 (λ = 0,41 / 0,25 / 0,15 / 0,06); MinT (Stichprobe) **36,8** (Windows; systemabhängig) / 1,017 / 1,015 / 0,995. | `test_history_experiment` |

Die Preset-Zeilen sind **Einzelnetze** (Seed 3); belastbar sind die Zeilen über drei Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Basisprognosen unterscheiden sich im Fehler** | Sind sie ohnehin fast kohärent (Wochenmittel: exakt; Holt-Winters je Knoten: fast), gibt es nichts zu gewinnen. | – |
| **Die Anteile bleiben stabil** | Top-down und Middle-out verteilen nach mittleren Anteilen der Trainingstage; ändern sie sich, verliert das Depot-Ergebnis deutlich. | Anteile nach Prognose statt nach Historie, Bottom-up |
| **Die Fehlerkovarianz ist schätzbar** | Bei vielen Knoten und wenig Fehlern ist die Stichprobenkovarianz unbrauchbar; die Schrumpfung hilft, ersetzt aber nicht die Daten. | WLS, Schrumpfung, weniger Knoten |
| **Ein-Schritt-Fehler beschreiben die Fehler bei Horizont $h$** | Die Gewichte stammen aus Ein-Schritt-Fehlern; bei langem Horizont passen sie schlechter (die Gewinne schrumpfen). | Fehler je Horizont schätzen |
| **Negative Prognosen sind kein Problem** | OLS, WLS und MinT können negative Werte liefern; hier werden sie abgeschnitten. | Nicht-negative Abstimmung |
| **Die Hierarchie ist ein Baum** | Nur Summen über eine Baumstruktur; gitterförmige und zeitliche Hierarchien fehlen. | Verallgemeinerte Summationsmatrizen |
| **Erzeugtes Netz, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, AR(1)-Schwankungen); echte Netze sind unordentlicher; die Modellwahl je Knoten reagiert empfindlich auf den Seed. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, 65 Tests, rund drei Minuten wegen der Experimente): die Matrizen von Hand und gegen unabhängige Rechnungen (kleinste Quadrate, Schleife, Störungstest der MinT-Optimalität), das Netz (Aggregation, Regionen, Tagesschock und Schwankung wirken), Basisprognosen und Modellwahl, Kennzahlen von Hand, Kohärenz aller Verfahren, kein Blick in die Zukunft,
Preset- und Permalink-Klemmen, AppTest-Rauchtests (Standard, jedes Preset, Knoten-, Depot- und Ursprungs-Auswahl, Extremwerte, drei Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen mit Bändern und Rangfolgen).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `hrc_constants.py` | Regler-Grenzen, Verfahren, Experiment-Seeds |
| `hrc_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `hrc_scenario.py` | Das Netz (Hierarchie, Niveauschwankungen, Tagesschocks) |
| `hrc_baselines.py`, `hrc_ets.py` | Basismodelle: Wochenmittel, Holt-Winters, Regression |
| `hrc_reconcile.py` | Die Matrizen $G$ und die Schrumpfung |
| `hrc_evaluation.py` | Basisprognosen und Modellwahl, Analyse, Kennzahlen, drei Experimente |
| `hrc_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Nicht-negative Abstimmung (Wickramasuriya et al. 2020) und Abstimmung von Prognoseverteilungen (Intervalle).
- Zeitliche Hierarchien (täglich, wöchentlich, monatlich) und mehrere Gliederungen (Produkt × Region).
- Top-down mit Prognoseanteilen; Fehlerkovarianz je Horizont.
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Gegenprobe im Test: scipy).
