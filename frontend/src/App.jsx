// src/App.jsx

import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const PDOK_FREE_URL =
  "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free";
const BAG_PAND_URL =
  "https://api.pdok.nl/lv/bag/ogc/v1-demo/collections/pand/items";

// CBS Kerncijfers wijken en buurten 2017 (83765NED)
const CBS_BASE_URL = "https://opendata.cbs.nl/ODataApi/OData/83765NED";
const CBS_TYPED_URL = `${CBS_BASE_URL}/TypedDataSet`;

const MAP_STYLE_URL = {
  version: 8,
  sources: {
    'osm': {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors'
    }
  },
  layers: [{
    id: 'osm',
    type: 'raster',
    source: 'osm'
  }]
};

const nf0 = new Intl.NumberFormat("nl-NL", {
  maximumFractionDigits: 0,
});
const nf1 = new Intl.NumberFormat("nl-NL", {
  maximumFractionDigits: 1,
});

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [storyLoading, setStoryLoading] = useState(false);
  const [storyText, setStoryText] = useState("");
  const [storyAreaHa, setStoryAreaHa] = useState(null);

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  async function handleSearch(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    setStoryText("");
    setStoryAreaHa(null);

    if (!query.trim()) {
      setError('Vul een adres in (bijv. "Damrak 1, Amsterdam").');
      return;
    }

    setLoading(true);
    try {
      // 1) PDOK adreszoeker
      const locUrl = `${PDOK_FREE_URL}?q=${encodeURIComponent(
        query
      )}&rows=1&fq=type:adres`;
      const locResp = await fetch(locUrl, {
        headers: { Accept: "application/json" },
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
        const match = doc.centroide_ll.match(/POINT\(([^ ]+) ([^)]+)\)/);
        if (match) {
          const lon = parseFloat(match[1]);
          const lat = parseFloat(match[2]);
          if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
            coords = [lat, lon];
          }
        }
      }

      // 2) BAG pand info + geometry
      let buildingInfo = null;
      let geometry = null;
      let pandIdentificatie = null;
      if (doc.id && doc.id.includes("pand")) {
        pandIdentificatie = doc.id.split(":").pop();
      }
      if (pandIdentificatie) {
        const bagUrl = `${BAG_PAND_URL}?identificatie=${pandIdentificatie}&f=json`;
        const bagResp = await fetch(bagUrl);
        if (bagResp.ok) {
          const bagData = await bagResp.json();
          const features = bagData.features || bagData.items || [];
          if (features.length > 0) {
            const feature = features[0];
            const props = feature.properties || feature;
            buildingInfo = {
              bouwjaar: props.bouwjaar,
              gebruiksdoel: props.gebruiksdoel || props.gebruiksdoelen,
              status: props.status,
            };
            if (feature.geometry) {
              geometry = feature.geometry;
            }
          }
        }
      }

      // 3) CBS kerncijfers (Kerncijfers wijken en buurten 2017 - 83765NED)
      const buurtCode =
        doc.buurtcode || doc.wijkcode || doc.gemeentecode || null;

      let cbsStats = null;
      if (buurtCode) {
        const filter = encodeURIComponent(
          `WijkenEnBuurten eq '${buurtCode}'`
        );
        const cbsUrl = `${CBS_TYPED_URL}?$filter=${filter}&$top=1`;
        const cbsResp = await fetch(cbsUrl);
        if (cbsResp.ok) {
          const cbsData = await cbsResp.json();
          const rows = cbsData.value || [];
          if (rows.length > 0) {
            const row = rows[0];

            // Helper: probeer meerdere mogelijke sleutel-namen
            const pick = (obj, keys) => {
              for (const key of keys) {
                if (
                  Object.prototype.hasOwnProperty.call(obj, key) &&
                  obj[key] !== null &&
                  obj[key] !== undefined
                ) {
                  return obj[key];
                }
              }
              return null;
            };

            const population = pick(row, [
              "AantalInwoners_5",
            ]);
            const density = pick(row, [
              "Bevolkingsdichtheid_33",
            ]);
            // Calculate 65+ percentage from age groups
            const totalPopulation = row.AantalInwoners_5;
            const over65 = row.k_65JaarOfOuder_12;
            const pct65Plus = totalPopulation && over65 ?
              Math.round((over65 / totalPopulation) * 100 * 10) / 10 : null;

            const incomePerPerson = pick(row, [
              "GemiddeldInkomenPerInwoner_66",
            ]);

            cbsStats = {
              buurtCode,
              population,
              density,
              pct65Plus,
              incomePerPerson,
            };

            // Superhandig om 1x te inspecteren in de browser console:
            console.log("CBS row for buurt:", buurtCode, row);
          }
        }
      }

      setResult({
        address: formattedAddress,
        coords,
        buildingInfo,
        cbsStats,
        geometry,
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
      cbsStats: result.cbsStats,
      geometry: result.geometry,
    };
  }

  async function generateStory(persona = "algemeen huishouden") {
    const data = buildAiData();
    if (!data) return;
    setStoryLoading(true);
    setStoryText("");
    setStoryAreaHa(null);
    try {
      const resp = await fetch("/api/neighbourhood-story", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data, persona }),
      });
      const json = await resp.json();
      if (!resp.ok) throw new Error(json.detail || json.error || "AI-fout");

      setStoryText(json.story || "");
      setStoryAreaHa(
        typeof json.area_ha === "number" ? json.area_ha : null
      );
    } catch (err) {
      console.error(err);
      setStoryText("Kon geen buurtverhaal genereren (AI-fout).");
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

  useEffect(() => {
    if (!mapContainerRef.current || !result?.coords) {
      return;
    }

    // Ruim oude map op als we een nieuwe maken
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    const [lat, lon] = result.coords;
    const center = [lon, lat]; // maplibre = [lon, lat]

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE_URL,
      center,
      zoom: 17,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    // Marker op het adres
    new maplibregl.Marker({ color: "#22c55e" })
      .setLngLat(center)
      .addTo(map);

    // Teken BAG-polygon als we een geometry hebben
    map.on("load", () => {
      if (result.geometry) {
        const feature = {
          type: "Feature",
          geometry: result.geometry,
          properties: {},
        };

        if (!map.getSource("building")) {
          map.addSource("building", {
            type: "geojson",
            data: feature,
          });

          map.addLayer({
            id: "building-fill",
            type: "fill",
            source: "building",
            paint: {
              "fill-color": "#22c55e",
              "fill-opacity": 0.3,
            },
          });

          map.addLayer({
            id: "building-outline",
            type: "line",
            source: "building",
            paint: {
              "line-color": "#22c55e",
              "line-width": 2,
            },
          });
        } else {
          const src = map.getSource("building");
          src.setData(feature);
        }
      }
    });

    mapRef.current = map;

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [result?.coords, result?.geometry]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title-row">
          <span className="logo-pill">NL</span>
          <div>
            <h1>Wat is mijn straat?</h1>
            <p className="subtitle">
              In één scherm: adres, pand, buurtcijfers en een AI-buurtverhaal.
            </p>
          </div>
        </div>
      </header>

      <div className="card">
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
            {/* Adres & codes */}
            <section className="section">
              <h2>Adres</h2>
              <div className="badge-row">
                <span className="badge primary-badge">{result.address}</span>
                {result.cbsStats?.buurtCode && (
                  <span className="badge">
                    CBS buurtcode: {result.cbsStats.buurtCode.trim()}
                  </span>
                )}
                {result.buildingInfo?.bouwjaar && (
                  <span className="badge">
                    Bouwjaar pand: {result.buildingInfo.bouwjaar}
                  </span>
                )}
              </div>
            </section>

            {/* Pandinfo */}
            {result.buildingInfo && (
              <section className="section">
                <h2>Pand</h2>
                <div className="stat-grid">
                  {result.buildingInfo.bouwjaar && (
                    <div className="stat-card">
                      <div className="stat-label">Bouwjaar</div>
                      <div className="stat-value">
                        {result.buildingInfo.bouwjaar}
                      </div>
                      <div className="stat-help small">
                        Jaar waarin het pand volgens BAG is gebouwd.
                      </div>
                    </div>
                  )}
                  {result.buildingInfo.gebruiksdoel && (
                    <div className="stat-card">
                      <div className="stat-label">Gebruiksdoel</div>
                      <div className="stat-value">
                        {Array.isArray(result.buildingInfo.gebruiksdoel)
                          ? result.buildingInfo.gebruiksdoel.join(", ")
                          : result.buildingInfo.gebruiksdoel}
                      </div>
                      <div className="stat-help small">
                        Registratie van het hoofdzakelijke gebruik.
                      </div>
                    </div>
                  )}
                  {result.buildingInfo.status && (
                    <div className="stat-card">
                      <div className="stat-label">Status</div>
                      <div className="stat-value">
                        {result.buildingInfo.status}
                      </div>
                      <div className="stat-help small">
                        BAG-status (bijv. in gebruik, in aanbouw).
                      </div>
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* Map + cijfers in twee kolommen */}
            <section className="section two-column">
              <div className="column">
                {result.coords && (
                  <div className="map-card">
                    <div ref={mapContainerRef} className="map" />
                    <p className="small">
                      Benadering van de locatie. Geen juridisch kaartmateriaal.
                    </p>
                  </div>
                )}
              </div>

              <div className="column">
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
                          Hoeveel mensen er in de buurt wonen (CBS-data).
                        </div>
                      </div>
                    )}

                    {storyAreaHa != null && (
                      <div className="stat-card">
                        <div className="stat-label">Oppervlakte pand (ongeveer)</div>
                        <div className="stat-value">
                          {formatOrNA(storyAreaHa, nf1)}
                          <span className="small"> ha</span>
                        </div>
                        <div className="stat-help">
                          Berekend met GeoPandas op basis van BAG-geometrie
                          (indicatief).
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
                          Hogere dichtheid betekent meestal een drukkere wijk met
                          meer voorzieningen.
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
                  <p className="small">Geen CBS-buurtcijfers gevonden.</p>
                )}
              </div>
            </section>

            {/* AI: Buurtverhaal */}
            <section className="section">
              <h2>Buurtverhaal (AI)</h2>
              <p className="small">
                Laat een korte beschrijving maken van de buurt op basis van de
                gegevens hierboven.
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
                <div className="stat-card story-card">
                  <ReactMarkdown>{storyText}</ReactMarkdown>
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  );
}