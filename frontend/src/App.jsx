// src/App.jsx

import React, { useState, useEffect, useRef, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

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
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    {
      id: "osm",
      type: "raster",
      source: "osm",
    },
  ],
};

const nf0 = new Intl.NumberFormat("nl-NL", {
  maximumFractionDigits: 0,
});
const nf1 = new Intl.NumberFormat("nl-NL", {
  maximumFractionDigits: 1,
});

function formatOrNA(value, formatter = nf0) {
  if (value === null || value === undefined || value === "") {
    return "n.v.t.";
  }
  const num = Number(value);
  if (Number.isNaN(num)) return "n.v.t.";
  return formatter.format(num);
}

function getIncomeBadge(incomePerPerson) {
  if (incomePerPerson === null || incomePerPerson === undefined) {
    return { label: "Onbekend inkomen", level: "neutral" };
  }
  const v = Number(incomePerPerson);
  if (v < 30) return { label: "Relatief lager inkomen", level: "low" };
  if (v > 45) return { label: "Relatief hoger inkomen", level: "high" };
  return { label: "Ongeveer gemiddeld inkomen", level: "mid" };
}

// Simpele mapping van cluster → icoon + kleurklasse
function getClusterVisual(clusterId, labelShort) {
  const text = (labelShort || "").toLowerCase();

  if (text.includes("sted") || text.includes("centrum")) {
    return { icon: "🏙️", toneClass: "cluster-pill-city" };
  }
  if (text.includes("rustig") || text.includes("dorps")) {
    return { icon: "🏡", toneClass: "cluster-pill-suburban" };
  }
  if (text.includes("welvar") || text.includes("rijk") || text.includes("hoog")) {
    return { icon: "💎", toneClass: "cluster-pill-wealthy" };
  }

  // fallback: gebaseerd op id
  if (clusterId % 3 === 0) {
    return { icon: "🏙️", toneClass: "cluster-pill-city" };
  }
  if (clusterId % 3 === 1) {
    return { icon: "🏡", toneClass: "cluster-pill-suburban" };
  }
  return { icon: "💎", toneClass: "cluster-pill-wealthy" };
}

function capitalizeFirst(str) {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function getClusterIcon(labelShort) {
  const s = (labelShort || "").toLowerCase();
  if (!s) return "📍";

  if (s.includes("sted") || s.includes("centrum") || s.includes("druk")) {
    return "🏙️";
  }
  if (s.includes("dorps") || s.includes("rust") || s.includes("groen")) {
    return "🏡";
  }
  if (s.includes("welvar") || s.includes("rijk") || s.includes("hoog inkomen")) {
    return "💰";
  }
  if (s.includes("student") || s.includes("jong")) {
    return "🎓";
  }
  if (s.includes("vergrijzend") || s.includes("oud")) {
    return "🧓";
  }
  return "📍";
}

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [storyLoading, setStoryLoading] = useState(false);
  const [storyText, setStoryText] = useState("");
  const [storyAreaHa, setStoryAreaHa] = useState(null);

  const [similarBuurten, setSimilarBuurten] = useState(null); // { base_buurt_code, neighbours: [...] }
  const [clusterInfo, setClusterInfo] = useState(null); // { buurt_code, cluster, label, label_long }
  const [buurtNamen, setBuurtNamen] = useState(new Map()); // Map<buurtCode, buurtNaam>
  const [mlLoading, setMlLoading] = useState(false);

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  // Load buurt namen on startup
  useEffect(() => {
    fetch("/data/cbs_buurt_namen_83765.csv")
      .then((resp) => resp.text())
      .then((csvText) => {
        const lines = csvText.trim().split("\n");
        if (lines.length === 0) return;

        const header = lines[0].split(",");
        const identifierIdx = header.indexOf("Identifier");
        const titleIdx = header.indexOf("Title");

        if (identifierIdx === -1 || titleIdx === -1) {
          console.warn("Buurt namen CSV heeft niet de verwachte kolommen");
          return;
        }

        const namenMap = new Map();
        for (let i = 1; i < lines.length; i++) {
          const cols = lines[i].split(",");
          const buurtCode = cols[identifierIdx]?.replace(/"/g, "").trim();
          const buurtNaam = cols[titleIdx]?.replace(/"/g, "").trim();
          if (buurtCode && buurtNaam) {
            namenMap.set(buurtCode, buurtNaam);
          }
        }

        setBuurtNamen(namenMap);
        console.log(`Geladen: ${namenMap.size} buurt namen`);
      })
      .catch((err) => {
        console.warn("Kon buurt namen niet laden:", err);
      });
  }, []);

  // Helper: get buurt naam from code
  function getBuurtNaam(buurtCode) {
    if (!buurtCode) return null;
    return buurtNamen.get(buurtCode.trim()) || null;
  }

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
        const filter = encodeURIComponent(`WijkenEnBuurten eq '${buurtCode}'`);
        const cbsUrl = `${CBS_TYPED_URL}?$filter=${filter}&$top=1`;
        const cbsResp = await fetch(cbsUrl);
        if (cbsResp.ok) {
          const cbsData = await cbsResp.json();
          const rows = cbsData.value || [];
          if (rows.length > 0) {
            const row = rows[0];

            // Helper: accepteert string of array van keys
            const pick = (obj, keys) => {
              const arr = Array.isArray(keys) ? keys : [keys];
              for (const k of arr) {
                if (
                  Object.prototype.hasOwnProperty.call(obj, k) &&
                  obj[k] !== null &&
                  obj[k] !== undefined
                ) {
                  return obj[k];
                }
              }
              return null;
            };

            const population = pick(row, "AantalInwoners_5");
            const density = pick(row, "Bevolkingsdichtheid_33");
            const gemeenteNaam = pick(row, "Gemeentenaam_1");

            // Leeftijdsgroepen (absolute aantallen)
            const ageGroups = {
              "0–15": pick(row, "k_0Tot15Jaar_8"),
              "15–25": pick(row, "k_15Tot25Jaar_9"),
              "25–45": pick(row, "k_25Tot45Jaar_10"),
              "45–65": pick(row, "k_45Tot65Jaar_11"),
              "65+": pick(row, "k_65JaarOfOuder_12"),
            };

            const totalPopulation = population || 0;
            const over65 = ageGroups["65+"] || 0;
            const pct65Plus =
              totalPopulation && over65
                ? Math.round((over65 / totalPopulation) * 100 * 10) / 10
                : null;

            // Inkomen
            const incomePerPerson = pick(row, "GemiddeldInkomenPerInwoner_66");
            const incomePerReceiver = pick(
              row,
              "GemiddeldInkomenPerInkomensontvanger_65"
            );
            const pctLowIncomeHouseholds = pick(
              row,
              "k_40HuishoudensMetLaagsteInkomen_70"
            );
            const pctHighIncomeHouseholds = pick(
              row,
              "k_20HuishoudensMetHoogsteInkomen_71"
            );
            const shareLowIncomePersons = pick(
              row,
              "k_40PersonenMetLaagsteInkomen_67"
            );
            const shareHighIncomePersons = pick(
              row,
              "k_20PersonenMetHoogsteInkomen_68"
            );

            const carsPerHousehold = pick(
              row,
              "PersonenautoSPerHuishouden_91"
            );
            const totalCars = pick(row, "PersonenautoSTotaal_86");

            // Voorzieningen-afstand (km)
            const amenities = {
              supermarket_km: pick(row, "AfstandTotGroteSupermarkt_95"),
              huisarts_km: pick(row, "AfstandTotHuisartsenpraktijk_94"),
              kinderdagverblijf_km: pick(
                row,
                "AfstandTotKinderdagverblijf_96"
              ),
              school_km: pick(row, "AfstandTotSchool_97"),
            };

            cbsStats = {
              buurtCode,
              gemeenteNaam,
              population,
              density,
              pct65Plus,
              incomePerPerson,
              incomePerReceiver,
              pctLowIncomeHouseholds,
              pctHighIncomeHouseholds,
              shareLowIncomePersons,
              shareHighIncomePersons,
              carsPerHousehold,
              totalCars,
              ageGroups,
              amenities,
            };

            console.log("CBS row for buurt:", buurtCode, row);
          }
        }
      }

      // 4) Crime data (if available in backend)
      let crimeData = null;
      if (buurtCode) {
        try {
          const crimeResp = await fetch(`/api/buurt-crime?buurt_code=${encodeURIComponent(buurtCode)}`);
          if (crimeResp.ok) {
            crimeData = await crimeResp.json();
            console.log("Crime data for buurt:", buurtCode, crimeData);
          }
        } catch (err) {
          console.log("No crime data available for buurt:", buurtCode);
        }
      }

      setResult({
        address: formattedAddress,
        coords,
        buildingInfo,
        cbsStats,
        geometry,
        crimeData,
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

    console.log("buildAiData - clusterInfo:", clusterInfo);
    console.log("buildAiData - result.cbsStats:", result.cbsStats);
    console.log("buildAiData - gemeenteNaam:", result.cbsStats?.gemeenteNaam);

    // Zorg altijd voor buurt naam in clusterInfo
    let enrichedClusterInfo = clusterInfo ? { ...clusterInfo } : null;
    if (!enrichedClusterInfo && result.cbsStats?.buurtCode) {
      // Fallback: maak basic clusterInfo met buurt naam
      const buurtNaam = getBuurtNaam(result.cbsStats.buurtCode);
      enrichedClusterInfo = {
        buurt_code: result.cbsStats.buurtCode,
        buurt_naam: buurtNaam || result.cbsStats.buurtCode,
      };
      console.log("buildAiData - created fallback clusterInfo:", enrichedClusterInfo);
    } else if (enrichedClusterInfo && !enrichedClusterInfo.buurt_naam && result.cbsStats?.buurtCode) {
      // Voeg buurt naam toe aan bestaande clusterInfo
      enrichedClusterInfo.buurt_naam = getBuurtNaam(result.cbsStats.buurtCode) || result.cbsStats.buurtCode;
      console.log("buildAiData - enriched clusterInfo:", enrichedClusterInfo);
    }

    return {
      address: result.address,
      coords: result.coords,
      buildingInfo: result.buildingInfo,
      cbsStats: result.cbsStats,
      geometry: result.geometry,
      clusterInfo: enrichedClusterInfo,
      similarBuurten: similarBuurten,
      crimeData: result.crimeData,
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

  // Highcharts: leeftijdsopbouw
  const ageChartOptions = useMemo(() => {
    const stats = result?.cbsStats;
    if (!stats?.ageGroups || !stats.population) return null;

    return {
      chart: {
        type: "column",
        backgroundColor: "transparent",
        height: 220,
      },
      title: {
        text: "Leeftijdsopbouw buurt",
        style: { color: "#e5e7eb", fontSize: "12px" },
      },
      xAxis: {
        categories: Object.keys(stats.ageGroups),
        crosshair: true,
        labels: { style: { color: "#9ca3af", fontSize: "11px" } },
      },
      yAxis: {
        min: 0,
        title: {
          text: "Aantal inwoners",
          style: { color: "#9ca3af", fontSize: "11px" },
        },
        gridLineColor: "rgba(55,65,81,0.4)",
        labels: { style: { color: "#9ca3af", fontSize: "10px" } },
      },
      legend: { enabled: false },
      series: [
        {
          name: "Inwoners",
          data: Object.values(stats.ageGroups).map((v) =>
            typeof v === "number" ? v : null
          ),
          color: "#22c55e",
        },
      ],
      credits: { enabled: false },
    };
  }, [result?.cbsStats]);

  // Highcharts: inkomenprofiel
  const incomeChartOptions = useMemo(() => {
    const stats = result?.cbsStats;
    if (!stats?.incomePerPerson) return null;

    const income = Number(stats.incomePerPerson);
    const low = stats.shareLowIncomePersons ?? null;
    const high = stats.shareHighIncomePersons ?? null;

    const categories = [];
    const data = [];

    categories.push("Gem. inkomen (x 1000 €)");
    data.push(income);

    if (low !== null) {
      categories.push("Laagste 40% (%)");
      data.push(low);
    }
    if (high !== null) {
      categories.push("Hoogste 20% (%)");
      data.push(high);
    }

    return {
      chart: {
        type: "column",
        backgroundColor: "transparent",
        height: 220,
      },
      title: {
        text: "Inkomenprofiel",

        style: { color: "#e5e7eb", fontSize: "12px" },
      },
      xAxis: {
        categories,
        labels: { style: { color: "#e5e7eb", fontSize: "11px" } },
      },
      yAxis: {
        min: 0,
        title: { text: null },
        labels: { style: { color: "#9ca3af" } },
        gridLineColor: "rgba(55,65,81,0.5)",
      },
      legend: { enabled: false },
      credits: { enabled: false },
      series: [
        {
          name: "Inkomen / verdeling",
          data,
          borderRadius: 3,
          color: "#22c55e",
        },
      ],
      tooltip: {
        backgroundColor: "#020617",
        borderColor: "#4b5563",
        style: { color: "#e5e7eb", fontSize: "11px" },
        formatter: function () {
          const key = this.key;
          if (key.includes("Gem. inkomen")) {
            return `${key}: ${this.y.toFixed(1)} × 1000 €`;
          }
          return `${key}: ${this.y.toFixed(1)} %`;
        },
      },
    };
  }, [result?.cbsStats]);

  // PCA-chart opties
  const pcaChartOptions = React.useMemo(() => {
    if (!similarBuurten || similarBuurten.base_pca_x == null) return null;

    const basePoint = {
      x: similarBuurten.base_pca_x,
      y: similarBuurten.base_pca_y,
      name: "Deze buurt",
    };

    const neighbourPoints =
      similarBuurten.neighbours
        ?.filter((n) => n.pca_x != null && n.pca_y != null)
        .map((n) => ({
          x: n.pca_x,
          y: n.pca_y,
          name: `${n.buurt_code} – ${n.gemeente}`,
          cluster: n.cluster_label_short,
        })) || [];

    if (!neighbourPoints.length) return null;

    return {
      chart: {
        type: "scatter",
        backgroundColor: "transparent",
        height: 260,
      },
      title: {
        text: "Vergelijkbare buurten (PCA-ruimte)",
        style: { color: "#e5e7eb", fontSize: "12px" },
      },
      xAxis: {
        title: { text: "PCA 1", style: { color: "#9ca3af", fontSize: "11px" } },
        gridLineColor: "rgba(55,65,81,0.4)",
        labels: { style: { color: "#9ca3af", fontSize: "10px" } },
      },
      yAxis: {
        title: { text: "PCA 2", style: { color: "#9ca3af", fontSize: "11px" } },
        gridLineColor: "rgba(55,65,81,0.4)",
        labels: { style: { color: "#9ca3af", fontSize: "10px" } },
      },
      legend: { enabled: false },
      tooltip: {
        backgroundColor: "#020617",
        borderColor: "#4b5563",
        style: { color: "#e5e7eb", fontSize: "11px" },
        formatter: function () {
          if (this.series.name === "Deze buurt") {
            return "Deze buurt";
          }
          const p = this.point;
          return `${p.name}<br/>Cluster: ${p.cluster || "-"}`;
        },
      },
      series: [
        {
          name: "Deze buurt",
          data: [basePoint],
          marker: { symbol: "circle", radius: 6 },
        },
        {
          name: "Vergelijkbare buurten",
          data: neighbourPoints,
          marker: { symbol: "circle", radius: 4 },
        },
      ],
      credits: { enabled: false },
    };
  }, [similarBuurten]);

  // "PCA-achtige" KNN scatter (we gebruiken afstand + hoek om puntjes in 2D te leggen)
  const knnScatterOptions = useMemo(() => {
    if (!similarBuurten?.neighbours || similarBuurten.neighbours.length === 0) {
      return null;
    }

    const points = [];
    // huidige buurt in het midden
    if (result?.address) {
      points.push({
        name: result.address,
        x: 0,
        y: 0,
        cluster: clusterInfo?.cluster ?? null,
        isBase: true,
      });
    }

    const neighbours = similarBuurten.neighbours;
    const n = neighbours.length;
    neighbours.forEach((nb, i) => {
      const angle = (2 * Math.PI * i) / n;
      const r = nb.distance || 0.5;
      points.push({
        name: `${nb.naam.trim()} – ${nb.gemeente.trim()}`,
        x: r * Math.cos(angle),
        y: r * Math.sin(angle),
        cluster: nb.cluster,
        label_short: nb.cluster_label_short || null,
        isBase: false,
      });
    });

    return {
      chart: {
        type: "scatter",
        backgroundColor: "transparent",
        height: 240,
      },
      title: {
        text: "Vergelijkbare buurten (ML-ruimte)",
        style: { color: "#e5e7eb", fontSize: "12px" },
      },
      xAxis: {
        title: { text: null },
        labels: { enabled: false },
        gridLineColor: "rgba(55,65,81,0.5)",
      },
      yAxis: {
        title: { text: null },
        labels: { enabled: false },
        gridLineColor: "rgba(55,65,81,0.5)",
      },
      legend: { enabled: false },
      credits: { enabled: false },
      series: [
        {
          name: "Buurten",
          data: points,
          color: "#22c55e",
          marker: {
            radius: 4,
            symbol: "circle",
          },
        },
      ],
      tooltip: {
        backgroundColor: "#020617",
        borderColor: "#4b5563",
        style: { color: "#e5e7eb", fontSize: "11px" },
        formatter: function () {
          const p = this.point;
          if (p.isBase) {
            return `<b>Deze buurt</b><br/>${p.name}`;
          }
          return `<b>${p.name}</b><br/>Cluster: ${
            p.cluster ?? "n.b."
          }<br/>Afstand (ML): ${Math.sqrt(
            p.x * p.x + p.y * p.y
          ).toFixed(2)}`;
        },
      },
    };
  }, [similarBuurten, result?.address, clusterInfo]);

  // MapLibre: render when coords/geometry change
  useEffect(() => {
    if (!mapContainerRef.current || !result?.coords) {
      return;
    }

    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    const [lat, lon] = result.coords;
    const center = [lon, lat]; // [lon, lat] voor MapLibre

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE_URL,
      center,
      zoom: 17,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    new maplibregl.Marker({ color: "#22c55e" }).setLngLat(center).addTo(map);

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

  // ML: fetch similar buurten + cluster info when we know the buurtCode
  useEffect(() => {
    const buurtCode = result?.cbsStats?.buurtCode;
    if (!buurtCode) {
      setSimilarBuurten(null);
      setClusterInfo(null);
      return;
    }

    const trimmed = buurtCode.trim();
    setMlLoading(true);

    Promise.all([
      fetch(
        `/api/similar-buurten?buurt_code=${encodeURIComponent(trimmed)}&k=5`
      ).then((r) => (r.ok ? r.json() : Promise.reject(r))),
      fetch(`/api/buurt-cluster?buurt_code=${encodeURIComponent(trimmed)}`).then(
        (r) => (r.ok ? r.json() : Promise.reject(r))
      ),
    ])
      .then(([similarJson, clusterJson]) => {
        setSimilarBuurten(similarJson);
        setClusterInfo(clusterJson);
      })
      .catch((err) => {
        console.error("ML endpoints error:", err);
        setSimilarBuurten(null);
        setClusterInfo(null);
      })
      .finally(() => setMlLoading(false));
  }, [result?.cbsStats?.buurtCode]);

  const incomeBadge =
    result?.cbsStats?.incomePerPerson != null
      ? getIncomeBadge(result.cbsStats.incomePerPerson)
      : null;

  const clusterVisual = clusterInfo
    ? getClusterVisual(clusterInfo.cluster, capitalizeFirst(clusterInfo.label))
    : null;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title-row">
          <span className="logo-pill">NL</span>
          <div>
            <h1>Wat is mijn straat?</h1>
            <p className="subtitle">
              In één scherm: pand, buurtcijfers, ML-inzichten en een AI-buurtverhaal.
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
                    Buurt: {getBuurtNaam(result.cbsStats.buurtCode) || result.cbsStats.buurtCode.trim()}
                  </span>
                )}
                {clusterInfo && (
                  <span
                    className={
                      "badge cluster-pill " + (clusterVisual?.toneClass || "")
                    }
                  >
                    {clusterVisual?.icon && (
                      <span className="cluster-icon">
                        {clusterVisual.icon}
                      </span>
                    )}
                    {capitalizeFirst(clusterInfo.label) || `Cluster ${clusterInfo.cluster}`}
                  </span>
                )}
                {incomeBadge && (
                  <span
                    className={
                      "badge income-badge income-" + incomeBadge.level
                    }
                  >
                    {incomeBadge.label}
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

            {/* MAP + STATS + AI/ML INSIGHTS */}
            <section className="section insights-grid">
              {/* Left: big map */}
              <div className="insights-map-column">
                {result.coords && (
                  <div className="map-card">
                    <div ref={mapContainerRef} className="map" />
                    <p className="small">
                      Benadering van de locatie. Geen juridisch kaartmateriaal.
                    </p>
                  </div>
                )}

              </div>

              {/* Right: panels */}
              <div className="insights-panel-column">
                {/* Buurtcijfers + Geo feature */}
                <div className="panel-block">
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
                            € {formatOrNA(result.cbsStats.incomePerPerson, nf0)}
                          </div>
                          <div className="stat-help">
                            Gemiddeld besteedbaar inkomen per persoon (CBS).
                          </div>
                        </div>
                      )}

                      {storyAreaHa != null && (
                        <div className="stat-card">
                          <div className="stat-label">
                            Oppervlakte pand (ongeveer)
                          </div>
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

                      {result.crimeData && result.crimeData.total_crimes != null && (
                        <div className="stat-card">
                          <div className="stat-label">Criminaliteit</div>
                          <div className="stat-value">
                            {formatOrNA(result.crimeData.total_crimes, nf0)}
                          </div>
                          <div className="stat-help">
                            Geregistreerde misdrijven in deze buurt (CBS Politie data).
                            {result.crimeData.crime_rate_per_1000 != null && (
                              <div style={{ marginTop: "0.25rem" }}>
                                <small>
                                  {formatOrNA(result.crimeData.crime_rate_per_1000, nf1)} per 1000 inwoners
                                </small>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="small">Geen CBS-buurtcijfers gevonden.</p>
                  )}
                </div>

                {/* Leeftijd + inkomen charts */}
                {(ageChartOptions || incomeChartOptions) && (
                  <div className="panel-block charts-row">
                    {ageChartOptions && (
                      <div className="chart-card">
                        <HighchartsReact
                          highcharts={Highcharts}
                          options={ageChartOptions}
                        />
                      </div>
                    )}
                    {incomeChartOptions && (
                      <div className="chart-card">
                        <HighchartsReact
                          highcharts={Highcharts}
                          options={incomeChartOptions}
                        />
                      </div>
                    )}
                  </div>
                )}

                {/* AI + ML panel */}
                <div className="panel-block">
                  <h2>AI & ML-inzichten</h2>

                  {clusterInfo && (
                    <div className="stat-card" style={{ marginBottom: "0.6rem" }}>
                      <div className="stat-label">Buurtprofiel (cluster)</div>
                      <div className="stat-value">
                        <span style={{ marginRight: "0.35rem" }}>
                          {getClusterIcon(capitalizeFirst(clusterInfo.label))}
                        </span>
                        {capitalizeFirst(clusterInfo.label) || clusterInfo.label}
                      </div>
                      {clusterInfo.label_long && (
                        <div className="stat-help small">{clusterInfo.label_long}</div>
                      )}
                    </div>
                  )}

                  {/* AI verhaal */}
                  <div className="stat-card" style={{ marginBottom: "0.6rem" }}>
                    <div className="stat-label">AI-buurtverhaal</div>
                    <div className="form-row" style={{ margin: "0.4rem 0 0.4rem" }}>
                      <button
                        type="button"
                        onClick={() => generateStory("starter")}
                        disabled={storyLoading}
                      >
                        {storyLoading ? "AI is bezig..." : "Maak buurtverhaal"}
                      </button>
                    </div>
                    {storyText && (
                      <div className="story-card">
                        <ReactMarkdown>{storyText}</ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Vergelijkbare buurten */}
                  {similarBuurten && similarBuurten.neighbours?.length > 0 && (
                    <div className="stat-card" style={{ marginBottom: "0.6rem" }}>
                      <div className="stat-label">Vergelijkbare buurten (KNN)</div>
                      <ul className="similar-list">
                        {similarBuurten.neighbours.map((b) => (
                          <li key={b.buurt_code}>
                            <span className="similar-icon">
                              {getClusterIcon(capitalizeFirst(b.cluster_label_short))}
                            </span>
                            <div className="similar-main">
                              <div className="similar-title">
                                {b.buurt_code} – {b.gemeente}
                              </div>
                              <div className="small">
                                {capitalizeFirst(b.cluster_label_short)} •{" "}
                                {b.income_per_person != null
                                  ? `inkomen: ${formatOrNA(b.income_per_person, nf1)} × 1000 €`
                                  : "inkomen: n.v.t."}
                                {b.population != null
                                  ? ` • inwoners: ${formatOrNA(b.population, nf0)}`
                                  : ""}
                              </div>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {mlLoading && (
                    <p className="small" style={{ marginTop: "0.4rem" }}>
                      ML-inzichten worden geladen...
                    </p>
                  )}
                </div>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}