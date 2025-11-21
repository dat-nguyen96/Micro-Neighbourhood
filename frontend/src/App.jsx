import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const PDOK_FREE_URL =
  "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free";
const BAG_PAND_URL =
  "https://api.pdok.nl/lv/bag/ogc/v1-demo/collections/pand/items";
const CBS_TABLE_URL =
  "https://datasets.cbs.nl/odata/v1/CBS/83765NED/Observations";

const markerIcon = new L.Icon({
  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const nf0 = new Intl.NumberFormat("nl-NL", {
  maximumFractionDigits: 0
});
const nf1 = new Intl.NumberFormat("nl-NL", {
  maximumFractionDigits: 1
});

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [storyLoading, setStoryLoading] = useState(false);
  const [storyText, setStoryText] = useState("");

  async function handleSearch(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    setStoryText("");

    if (!query.trim()) {
      setError('Vul een adres in (bijv. "Damrak 1, Amsterdam").');
      return;
    }

    setLoading(true);
    try {
      const locUrl = `${PDOK_FREE_URL}?q=${encodeURIComponent(
        query
      )}&rows=1&fq=type:adres`;
      const locResp = await fetch(locUrl, {
        headers: { Accept: "application/json" }
      });
      if (!locResp.ok) {
        throw new Error("Locatieserver gaf een foutmelding.");
      }
      const locData = await locResp.json();
      const docs = locData.response?.docs || [];
      if (docs.length === 0) {
        throw new Error("Geen adres gevonden voor deze zoekopdracht.");
      }

      const doc = docs[0];

      const formattedAddress =
        doc.weergavenaam ||
        `${doc.straatnaam || ""} ${doc.huisnummer || ""} ${
          doc.postcode || ""
        } ${doc.woonplaatsnaam || ""}`;

      // Coördinaten uit centroide_ll
      let coords = null;
      if (doc.centroide_ll) {
        const match = doc.centroide_ll.match(
          /POINT\(([^ ]+) ([^)]+)\)/
        );
        if (match) {
          const lon = parseFloat(match[1]);
          const lat = parseFloat(match[2]);
          if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
            coords = [lat, lon];
          }
        }
      }

      // Pand identificatie
      let pandIdentificatie = null;
      if (doc.id && doc.id.includes("pand")) {
        pandIdentificatie = doc.id.split(":").pop();
      }

      // BAG pand info
      let buildingInfo = null;
      if (pandIdentificatie) {
        const bagUrl = `${BAG_PAND_URL}?identificatie=${pandIdentificatie}&f=json`;
        const bagResp = await fetch(bagUrl);
        if (bagResp.ok) {
          const bagData = await bagResp.json();
          const features = bagData.features || bagData.items || [];
          if (features.length > 0) {
            const props =
              features[0].properties ||
              features[0].properties ||
              features[0];
            buildingInfo = {
              bouwjaar: props.bouwjaar,
              gebruiksdoel:
                props.gebruiksdoel || props.gebruiksdoelen,
              status: props.status
            };
          }
        }
      }

      // CBS kerncijfers
      const buurtCode =
        doc.buurtcode || doc.wijkcode || doc.gemeentecode || null;

      let cbsStats = null;
      if (buurtCode) {
        // Alleen T001036 (Totaal aantal inwoners) – dit weten we zeker bestaat
        const filter = encodeURIComponent(
          `WijkenEnBuurten eq '${buurtCode}' and Measure eq 'T001036'`
        );
        const cbsUrl = `${CBS_TABLE_URL}?$filter=${filter}&$top=5`;
        const cbsResp = await fetch(cbsUrl);
        if (cbsResp.ok) {
          const cbsData = await cbsResp.json();
          const rows = cbsData.value || cbsData.Observations || [];
          const popRow = rows[0];

          cbsStats = {
            buurtCode,
            population: popRow ? popRow.Value : null,
            density: null,          // TODO: later koppelen aan juiste CBS dataset
            pct65Plus: null,        // TODO: later koppelen aan juiste CBS dataset
            incomePerPerson: null   // TODO: later koppelen aan juiste CBS dataset
          };

          // Handige debug voor lokaal:
          console.log("CBS rows for buurt:", buurtCode, rows);
        }
      }

      setResult({
        address: formattedAddress,
        coords,
        buildingInfo,
        cbsStats
      });
    } catch (err) {
      console.error(err);
      setError(err.message || "Er ging iets mis.");
    } finally {
      setLoading(false);
    }
  }

  function buildAiData() {
    if (!result) return null;
    return {
      address: result.address,
      coords: result.coords,
      buildingInfo: result.buildingInfo,
      cbsStats: result.cbsStats
    };
  }

  async function generateStory(persona = "algemeen huishouden") {
    const data = buildAiData();
    if (!data) return;
    setStoryLoading(true);
    setStoryText("");
    try {
      const resp = await fetch("/api/neighbourhood-story", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data, persona })
      });
      const json = await resp.json();
      if (!resp.ok) throw new Error(json.detail || json.error || "AI-fout");
      setStoryText(json.story || "");
    } catch (err) {
      console.error(err);
      setStoryText(
        "Kon geen buurtverhaal genereren (AI-fout)."
      );
    } finally {
      setStoryLoading(false);
    }
  }


  function formatOrNA(value, formatter = nf0) {
    if (value === null || value === undefined || value === "") {
      return "n.v.t.";
    }
    return formatter.format(Number(value));
  }

  return (
    <div className="app-shell">
      <div className="card">
        <h1>Wat is mijn straat?</h1>
        <p className="subtitle">
          Snel inzicht in je pand en buurt op basis van BAG, PDOK, CBS en AI.
        </p>

        <form onSubmit={handleSearch}>
          <div className="form-row">
            <input
              type="text"
              placeholder='Bijv. "Damrak 1, Amsterdam"'
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" disabled={loading}>
              {loading ? "Bezig..." : "Bekijk buurt"}
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <>

            <h2>Adres</h2>
            <div className="badge-row">
              <span className="badge">{result.address}</span>
              {result.cbsStats?.buurtCode && (
                <span className="badge">
                  CBS buurtcode: {result.cbsStats.buurtCode.trim()}
                </span>
              )}
            </div>

            {result.coords && (
              <div className="map-card">
                <MapContainer
                  center={result.coords}
                  zoom={17}
                  scrollWheelZoom={false}
                  className="map"
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> bijdragers'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <Marker
                    position={result.coords}
                    icon={markerIcon}
                  >
                    <Popup>{result.address}</Popup>
                  </Marker>
                </MapContainer>
                <p className="small">
                  Benadering van de locatie. Geen juridisch kaartmateriaal.
                </p>
              </div>
            )}

            <h2>Buurt in één oogopslag</h2>
            {result.cbsStats ? (
              <div className="stat-grid">
                {result.cbsStats.population != null && (
                  <div className="stat-card">
                    <div className="stat-label">Inwoners (totaal)</div>
                    <div className="stat-value">
                      {formatOrNA(result.cbsStats.population, nf0)}
                    </div>
                    <div className="stat-help">
                      Hoeveel mensen er in de buurt wonen (CBS data).
                    </div>
                  </div>
                )}

                {result.cbsStats.density != null && (
                  <div className="stat-card">
                    <div className="stat-label">Bevolkingsdichtheid</div>
                    <div className="stat-value">
                      {formatOrNA(result.cbsStats.density, nf0)}
                      <span className="small"> / km²</span>
                    </div>
                    <div className="stat-help">
                      Hogere dichtheid betekent meestal een drukkere wijk met meer voorzieningen.
                    </div>
                  </div>
                )}

                {result.cbsStats.pct65Plus != null && (
                  <div className="stat-card">
                    <div className="stat-label">% 65-plus</div>
                    <div className="stat-value">
                      {formatOrNA(result.cbsStats.pct65Plus, nf1)}
                      <span className="small"> %</span>
                    </div>
                    <div className="stat-help">
                      Percentage bewoners van 65 jaar en ouder.
                    </div>
                  </div>
                )}

                {result.cbsStats.incomePerPerson != null && (
                  <div className="stat-card">
                    <div className="stat-label">
                      Gem. inkomen per persoon
                    </div>
                    <div className="stat-value">
                      €{" "}
                      {formatOrNA(
                        result.cbsStats.incomePerPerson,
                        nf0
                      )}
                    </div>
                    <div className="stat-help">
                      Gemiddeld besteedbaar inkomen per persoon (CBS).
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="small">
                Geen CBS-buurtcijfers gevonden.
              </p>
            )}

            {/* AI: Buurtverhaal */}
            <h2>Buurtverhaal (AI)</h2>
            <p className="small">
              Laat een korte beschrijving maken van de buurt op basis
              van de gegevens hierboven.
            </p>
            <div className="form-row" style={{ marginBottom: "0.75rem" }}>
              <button
                type="button"
                onClick={() => generateStory("starter")}
                disabled={storyLoading}
              >
                {storyLoading ? "AI is bezig..." : "Maak buurtverhaal"}
              </button>
            </div>
            {storyText && (
              <div
                className="stat-card"
                style={{ marginTop: "0.5rem" }}
              >
                <ReactMarkdown>{storyText}</ReactMarkdown>
              </div>
            )}

          </>

        )}

      </div>

    </div>

  );

}
