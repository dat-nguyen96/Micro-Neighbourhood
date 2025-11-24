// src/App.jsx

import React, { useState, useEffect, useRef, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import AddressSearchBar from "./components/AddressSearchBar";

const PDOK_FREE_URL =
  "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free";
const BAG_PAND_URL =
  "https://api.pdok.nl/lv/bag/ogc/v1-demo/collections/pand/items";

// CBS Kerncijfers wijken en buurten 2017 (83765NED)
const CBS_BASE_URL = "https://opendata.cbs.nl/ODataApi/OData/85984NED";
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

  // Comparison neighborhood state
  const [compareQuery, setCompareQuery] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");
  const [compareResult, setCompareResult] = useState(null);
  const [compareClusterInfo, setCompareClusterInfo] = useState(null);

  // Sync zoom between main and comparison maps
  function syncMaps(sourceMap, targetMap) {
    let lock = false;
    sourceMap.on("move", () => {
      if (lock) return;
      lock = true;
      const c = sourceMap.getCenter();
      const z = sourceMap.getZoom();
      targetMap.jumpTo({ center: c, zoom: z });
      lock = false;
    });
  }

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  // Comparison map refs
  const compareMapContainerRef = useRef(null);
  const compareMapRef = useRef(null);

  // Neighborhood boundary state
  const [buurtBoundary, setBuurtBoundary] = useState(null);
  const [compareBuurtBoundary, setCompareBuurtBoundary] = useState(null);

  // Load neighborhood boundary for main neighborhood
  useEffect(() => {
    if (!result?.cbsStats?.buurtCode) {
      setBuurtBoundary(null);
      return;
    }

    fetch(`/api/buurt-geometry?buurt_code=${encodeURIComponent(result.cbsStats.buurtCode)}`)
      .then((resp) => {
        if (!resp.ok) throw new Error("Buurt geometry ophalen mislukt");
        return resp.json();
      })
      .then((feature) => {
        setBuurtBoundary(feature);
        console.log("Main buurt boundary geladen:", feature);
      })
      .catch((err) => {
        console.warn("Kon main buurt boundary niet laden:", err);
        setBuurtBoundary(null);
      });
  }, [result?.cbsStats?.buurtCode]);

  // Load cluster info for comparison neighborhood
  useEffect(() => {
    if (!compareResult?.cbsStats?.buurtCode) {
      setCompareClusterInfo(null);
      return;
    }

    setMlLoading(true);
    fetch(`/api/buurt-cluster?buurt_code=${encodeURIComponent(compareResult.cbsStats.buurtCode)}`)
      .then((resp) => {
        if (!resp.ok) throw new Error("Cluster info ophalen mislukt");
        return resp.json();
      })
      .then((info) => {
        setCompareClusterInfo(info);
        console.log("Compare cluster info geladen:", info);
      })
      .catch((err) => {
        console.warn("Kon compare cluster info niet laden:", err);
        setCompareClusterInfo(null);
      })
      .finally(() => {
        setMlLoading(false);
      });
  }, [compareResult?.cbsStats?.buurtCode]);

  // Load neighborhood boundary for comparison neighborhood
  useEffect(() => {
    if (!compareResult?.cbsStats?.buurtCode) {
      setCompareBuurtBoundary(null);
      return;
    }

    fetch(`/api/buurt-geometry?buurt_code=${encodeURIComponent(compareResult.cbsStats.buurtCode)}`)
      .then((resp) => {
        if (!resp.ok) throw new Error("Compare buurt geometry ophalen mislukt");
        return resp.json();
      })
      .then((feature) => {
        setCompareBuurtBoundary(feature);
        console.log("Compare buurt boundary geladen:", feature);
      })
      .catch((err) => {
        console.warn("Kon compare buurt boundary niet laden:", err);
        setCompareBuurtBoundary(null);
      });
  }, [compareResult?.cbsStats?.buurtCode]);

  // Load buurt namen on startup
  useEffect(() => {
    fetch("/data/cbs_buurt_namen_85984.csv")
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

  async function handleSearchFromSuggestion(address, pdokDoc) {
    console.log("App: handleSearchFromSuggestion called with", { address, pdokDoc });
    setQuery(address);
    setError("");
    setResult(null);
    setStoryText("");
    setStoryAreaHa(null);

    if (!address.trim()) {
      setError('Vul een adres in (bijv. "Damrak 1, Amsterdam").');
      return;
    }

    setLoading(true);
    try {
      await runSearch(address, pdokDoc);
    } catch (err) {
      console.error(err);
      setError(err.message || "Er ging iets mis.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCompareSearchFromSuggestion(address, pdokDoc) {
    setCompareQuery(address);
    setCompareError("");
    setCompareResult(null);
    setCompareClusterInfo(null);

    if (!address.trim()) {
      setCompareError('Vul een adres in (bijv. "Damrak 1, Amsterdam").');
      return;
    }

    setCompareLoading(true);
    try {
      await runCompareSearch(address, pdokDoc);
    } catch (err) {
      console.error(err);
      setCompareError(err.message || "Er ging iets mis.");
    } finally {
      setCompareLoading(false);
    }
  }

  async function runSearch(address, pdokDoc) {
    console.log("App: runSearch called with", { address, pdokDoc });
    // Als we een pdokDoc hebben, gebruik deze direct
    // Anders doe een normale PDOK zoekopdracht
    let doc;
    if (pdokDoc && pdokDoc.id) {
      // Check if pdokDoc has required fields, otherwise fall back to free search
      doc = pdokDoc;
    } else {
      // Fallback: doe normale PDOK zoekopdracht
      const locUrl = `${PDOK_FREE_URL}?q=${encodeURIComponent(
        address
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
      doc = docs[0];
      }

      const formattedAddress =
        doc.weergavenaam ||
        `${doc.straatnaam || ""} ${doc.huisnummer || ""} ${
          doc.postcode || ""
        } ${doc.woonplaatsnaam || ""}`;

      // Coördinaten uit centroide_ll (PDOK free API)
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

      // Fallback: als geen coords gevonden, doe een nieuwe free search
      if (!coords && pdokDoc) {
        console.log("No coords in pdokDoc, doing fallback free search");
        const fallbackUrl = `${PDOK_FREE_URL}?q=${encodeURIComponent(
          formattedAddress
        )}&rows=1&fq=type:adres`;
        const fallbackResp = await fetch(fallbackUrl, {
          headers: { Accept: "application/json" },
        });
        if (fallbackResp.ok) {
          const fallbackData = await fallbackResp.json();
          const fallbackDocs = fallbackData.response?.docs || [];
          if (fallbackDocs.length > 0) {
            const fallbackDoc = fallbackDocs[0];
            if (fallbackDoc.centroide_ll) {
              const match = fallbackDoc.centroide_ll.match(/POINT\(([^ ]+) ([^)]+)\)/);
              if (match) {
                const lon = parseFloat(match[1]);
                const lat = parseFloat(match[2]);
                if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
                  coords = [lat, lon];
                  // Update doc with fallback data
                  doc = { ...doc, ...fallbackDoc };
                }
              }
            }
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
      // Gebruik onze eigen backend endpoint voor voorbewerkte CBS data
      const statsResp = await fetch(`/api/buurt-stats?buurt_code=${encodeURIComponent(buurtCode)}`);
      if (statsResp.ok) {
        cbsStats = await statsResp.json();
        console.log("CBS stats for buurt:", buurtCode, cbsStats);
      }
    }

    setResult({
      address: formattedAddress,
      coords,
      buildingInfo,
      cbsStats,
      geometry,
    });
  }

  async function runCompareSearch(address, pdokDoc) {
    // Als we een pdokDoc hebben, gebruik deze direct
    // Anders doe een normale PDOK zoekopdracht
    let doc;
    if (pdokDoc) {
      doc = pdokDoc;
    } else {
      // Fallback: doe normale PDOK zoekopdracht
      const locUrl = `${PDOK_FREE_URL}?q=${encodeURIComponent(
        address
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
      doc = docs[0];
    }

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

          // Huishoudens
          const pctLowIncomeHouseholdsValue = pick(
            row,
            "HuishoudensMetEenLaagInkomen_72"
          );
          const pctLowIncomeHouseholdsPercent =
            pctLowIncomeHouseholdsValue && totalPopulation
              ? Math.round((pctLowIncomeHouseholdsValue / totalPopulation) * 100 * 10) / 10
              : null;

          // Sociale zekerheid
          const shareBijstand = pick(row, "PersonenPerSoortUitkeringBijstand_74");
          const shareWW = pick(row, "PersonenPerSoortUitkeringWW_76");
          const shareAOW = pick(row, "PersonenPerSoortUitkeringAOW_77");

          // Economie
          const totalBedrijven = pick(row, "BedrijfsvestigingenTotaal_78");
          const bedrijvenLandbouw = pick(row, "ALandbouwBosbouwEnVisserij_79");
          const bedrijvenIndustrie = pick(row, "BFNijverheidEnEnergie_80");
          const bedrijvenHandel = pick(row, "GIHandelEnHoreca_81");

          // Mobiliteit
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

          // Criminaliteitscijfers uit 83765NED
          const geweldsMisdrijven = pick(row, "GeweldsEnSeksueleMisdrijven_108");
          const vermogensMisdrijven = pick(row, "TotaalDiefstalUitWoningSchuurED_106");

          // Mate van stedelijkheid (1=zeer sterk stedelijk, 5=niet stedelijk)
          const stedelijkheid = pick(row, "MateVanStedelijkheid_104");

          // Woningtypes (%)
          const pctAppartementen = pick(row, "PercentageEengezinswoning_36");
          const pctEengezinswoningen = pick(row, "PercentageMeergezinswoning_37");

          // Woning leeftijd (%)
          const pctWoningenVoor2000 = pick(row, "BouwjaarVoor2000_45");
          const pctWoningenVanaf2000 = pick(row, "BouwjaarVanaf2000_46");

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
            // Nieuwe criminaliteitscijfers
            geweldsMisdrijven,
            vermogensMisdrijven,
            // Nieuwe leefbaarheid stats
            stedelijkheid,
            pctAppartementen,
            pctEengezinswoningen,
            pctWoningenVoor2000,
            pctWoningenVanaf2000,
          };

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
      crimeData,
    });
  }

  async function handleSearch(e) {
    e.preventDefault();

    // Gebruik de nieuwe runSearch functie zonder pdokDoc
    await handleSearchFromSuggestion(query, null);
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

            // Criminaliteitscijfers uit 83765NED
            const geweldsMisdrijven = pick(row, "GeweldsEnSeksueleMisdrijven_108");
            const vermogensMisdrijven = pick(row, "TotaalDiefstalUitWoningSchuurED_106");

            // Mate van stedelijkheid (1=zeer sterk stedelijk, 5=niet stedelijk)
            const stedelijkheid = pick(row, "MateVanStedelijkheid_104");

            // Woningtypes (%)
            const pctAppartementen = pick(row, "PercentageEengezinswoning_36");
            const pctEengezinswoningen = pick(row, "PercentageMeergezinswoning_37");

            // Woning leeftijd (%)
            const pctWoningenVoor2000 = pick(row, "BouwjaarVoor2000_45");
            const pctWoningenVanaf2000 = pick(row, "BouwjaarVanaf2000_46");

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
              // Nieuwe criminaliteitscijfers
              geweldsMisdrijven,
              vermogensMisdrijven,
              // Nieuwe leefbaarheid stats
              stedelijkheid,
              pctAppartementen,
              pctEengezinswoningen,
              pctWoningenVoor2000,
              pctWoningenVanaf2000,
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

  // Radar chart: buurtvergelijking
  const comparisonRadarOptions = useMemo(() => {
    if (!result?.cbsStats || !compareResult?.cbsStats) {
      return null;
    }

    // Normaliseer waarden naar 0-100 schaal voor radar chart
    const normalize = (value, min, max) => {
      if (value === null || value === undefined) return 0;
      return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
    };

    // Categorieën en hun ranges voor normalisatie
    const categories = [
      "Bevolkingsdichtheid",
      "Inkomen p.p.",
      "65+ percentage",
      "Mate stedelijkheid",
      "Seksueel geweld",
      "Criminaliteit geweld",
      "Criminaliteit vermogens",
      "Vernieling"
    ];

    const mainData = [
      normalize(result.cbsStats.density, 0, 20000), // dichtheid 0-20k/km²
      normalize(result.cbsStats.incomePerPerson, 20, 60), // inkomen 20k-60k ×1000€
      normalize(result.cbsStats.pct65Plus, 0, 40), // 65+ 0-40%
      normalize(result.cbsStats.stedelijkheid, 1, 5) * 20, // stedelijkheid 1-5 → 20-100
      normalize(result.cbsStats.seksueelGeweld, 0, 2), // seksueel geweld 0-2 per 1000
      normalize(result.cbsStats.geweldsMisdrijven, 0, 10), // geweld 0-10 per 1000
      normalize(result.cbsStats.vermogensMisdrijven, 0, 20), // vermogens 0-20 per 1000
      normalize(result.cbsStats.vernielingsMisdrijven, 0, 5), // vernieling 0-5 per 1000
    ];

    const compareData = [
      normalize(compareResult.cbsStats.density, 0, 20000),
      normalize(compareResult.cbsStats.incomePerPerson, 20, 60),
      normalize(compareResult.cbsStats.pct65Plus, 0, 40),
      normalize(compareResult.cbsStats.stedelijkheid, 1, 5) * 20,
      normalize(compareResult.cbsStats.seksueelGeweld, 0, 2),
      normalize(compareResult.cbsStats.geweldsMisdrijven, 0, 10),
      normalize(compareResult.cbsStats.vermogensMisdrijven, 0, 20),
      normalize(compareResult.cbsStats.vernielingsMisdrijven, 0, 5),
    ];

    return {
      chart: {
        polar: true,
        type: "area",
        backgroundColor: "transparent",
        height: 280,
      },
      title: {
        text: "Buurtvergelijking",
        style: { color: "#e5e7eb", fontSize: "14px", fontWeight: "600" },
      },
      pane: {
        size: "80%",
      },
      xAxis: {
        categories: categories,
        tickmarkPlacement: "on",
        lineWidth: 0,
        labels: {
          style: { color: "#9ca3af", fontSize: "11px" },
        },
        gridLineColor: "rgba(55,65,81,0.5)",
      },
      yAxis: {
        gridLineInterpolation: "polygon",
        lineWidth: 0,
        min: 0,
        max: 100,
        labels: {
          enabled: false,
        },
        gridLineColor: "rgba(55,65,81,0.3)",
      },
      tooltip: {
        shared: true,
        backgroundColor: "#020617",
        borderColor: "#4b5563",
        style: { color: "#e5e7eb", fontSize: "11px" },
        formatter: function() {
          const category = this.x;
          const mainValue = this.points[0].y;
          const compareValue = this.points[1].y;

          let mainLabel = "n.v.t.";
          let compareLabel = "n.v.t.";

          // Denormaliseer voor tooltip
          if (category === "Bevolkingsdichtheid") {
            mainLabel = result.cbsStats.density ? `${result.cbsStats.density} /km²` : "n.v.t.";
            compareLabel = compareResult.cbsStats.density ? `${compareResult.cbsStats.density} /km²` : "n.v.t.";
          } else if (category === "Inkomen p.p.") {
            mainLabel = result.cbsStats.incomePerPerson ? `€${result.cbsStats.incomePerPerson * 1000}` : "n.v.t.";
            compareLabel = compareResult.cbsStats.incomePerPerson ? `€${compareResult.cbsStats.incomePerPerson * 1000}` : "n.v.t.";
          } else if (category === "65+ percentage") {
            mainLabel = result.cbsStats.pct65Plus ? `${result.cbsStats.pct65Plus.toFixed(1)}%` : "n.v.t.";
            compareLabel = compareResult.cbsStats.pct65Plus ? `${compareResult.cbsStats.pct65Plus.toFixed(1)}%` : "n.v.t.";
          } else if (category === "Mate stedelijkheid") {
            mainLabel = result.cbsStats.stedelijkheid ? `${result.cbsStats.stedelijkheid}/5` : "n.v.t.";
            compareLabel = compareResult.cbsStats.stedelijkheid ? `${compareResult.cbsStats.stedelijkheid}/5` : "n.v.t.";
          } else if (category === "Seksueel geweld") {
            mainLabel = result.cbsStats.seksueelGeweld ? `${result.cbsStats.seksueelGeweld.toFixed(1)} per 1000` : "n.v.t.";
            compareLabel = compareResult.cbsStats.seksueelGeweld ? `${compareResult.cbsStats.seksueelGeweld.toFixed(1)} per 1000` : "n.v.t.";
          } else if (category === "Criminaliteit geweld") {
            mainLabel = result.cbsStats.geweldsMisdrijven ? `${result.cbsStats.geweldsMisdrijven.toFixed(1)} per 1000` : "n.v.t.";
            compareLabel = compareResult.cbsStats.geweldsMisdrijven ? `${compareResult.cbsStats.geweldsMisdrijven.toFixed(1)} per 1000` : "n.v.t.";
          } else if (category === "Criminaliteit vermogens") {
            mainLabel = result.cbsStats.vermogensMisdrijven ? `${result.cbsStats.vermogensMisdrijven.toFixed(1)} per 1000` : "n.v.t.";
            compareLabel = compareResult.cbsStats.vermogensMisdrijven ? `${compareResult.cbsStats.vermogensMisdrijven.toFixed(1)} per 1000` : "n.v.t.";
          } else if (category === "Vernieling") {
            mainLabel = result.cbsStats.vernielingsMisdrijven ? `${result.cbsStats.vernielingsMisdrijven.toFixed(1)} per 1000` : "n.v.t.";
            compareLabel = compareResult.cbsStats.vernielingsMisdrijven ? `${compareResult.cbsStats.vernielingsMisdrijven.toFixed(1)} per 1000` : "n.v.t.";
          }

          return `<b>${category}</b><br/>
                  <span style="color: #22c55e">■ ${result.address.split(',')[0]}:</span> ${mainLabel}<br/>
                  <span style="color: #3b82f6">■ ${compareResult.address.split(',')[0]}:</span> ${compareLabel}`;
        },
      },
      plotOptions: {
        area: {
          fillOpacity: 0.1,
          lineWidth: 2,
          marker: {
            enabled: false,
          },
        },
      },
      series: [
        {
          name: result.address.split(',')[0], // Eerste deel van adres
          data: mainData,
          color: "#22c55e",
          fillColor: "rgba(34, 197, 94, 0.1)",
        },
        {
          name: compareResult.address.split(',')[0], // Eerste deel van adres
          data: compareData,
          color: "#3b82f6",
          fillColor: "rgba(59, 130, 246, 0.1)",
        },
      ],
      legend: {
        itemStyle: { color: "#9ca3af", fontSize: "11px" },
        itemHoverStyle: { color: "#e5e7eb" },
      },
      credits: {
        enabled: false,
      },
    };
  }, [result, compareResult]);

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
      // Building geometry
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

      // Neighborhood boundary
      if (buurtBoundary) {
        if (!map.getSource("buurt")) {
          map.addSource("buurt", {
            type: "geojson",
            data: buurtBoundary,
          });

          map.addLayer({
            id: "buurt-fill",
            type: "fill",
            source: "buurt",
            paint: {
              "fill-color": "#22c55e",
              "fill-opacity": 0.1,
            },
          });

          map.addLayer({
            id: "buurt-outline",
            type: "line",
            source: "buurt",
            paint: {
              "line-color": "#22c55e",
              "line-width": 3,
            },
          });
        } else {
          const src = map.getSource("buurt");
          src.setData(buurtBoundary);
        }
      }
    });

    mapRef.current = map;

    // Sync zoom with comparison map if it exists
    if (compareMapRef.current) {
      syncMaps(map, compareMapRef.current);
      syncMaps(compareMapRef.current, map);
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [result?.coords, result?.geometry, compareResult?.coords, buurtBoundary]);

  // Comparison MapLibre: render when compare coords/geometry change
  useEffect(() => {
    if (!compareMapContainerRef.current || !compareResult?.coords) {
      return;
    }

    if (compareMapRef.current) {
      compareMapRef.current.remove();
      compareMapRef.current = null;
    }

    const [lat, lon] = compareResult.coords;
    const center = [lon, lat]; // [lon, lat] voor MapLibre

    const compareMap = new maplibregl.Map({
      container: compareMapContainerRef.current,
      style: MAP_STYLE_URL,
      center,
      zoom: 16, // iets lager zoom voor vergelijkingskaart
    });

    compareMap.addControl(new maplibregl.NavigationControl(), "top-right");

    // Blauwe marker voor vergelijkingsbuurt
    new maplibregl.Marker({ color: "#3b82f6" }).setLngLat(center).addTo(compareMap);

    compareMap.on("load", () => {
      // Building geometry
      if (compareResult.geometry) {
        const feature = {
          type: "Feature",
          geometry: compareResult.geometry,
          properties: {},
        };

        if (!compareMap.getSource("compare-building")) {
          compareMap.addSource("compare-building", {
            type: "geojson",
            data: feature,
          });

          compareMap.addLayer({
            id: "compare-building-fill",
            type: "fill",
            source: "compare-building",
            paint: {
              "fill-color": "#3b82f6",
              "fill-opacity": 0.3,
            },
          });

          compareMap.addLayer({
            id: "compare-building-outline",
            type: "line",
            source: "compare-building",
            paint: {
              "line-color": "#3b82f6",
              "line-width": 2,
            },
          });
        } else {
          const src = compareMap.getSource("compare-building");
          src.setData(feature);
        }
      }

      // Neighborhood boundary
      if (compareBuurtBoundary) {
        if (!compareMap.getSource("compare-buurt")) {
          compareMap.addSource("compare-buurt", {
            type: "geojson",
            data: compareBuurtBoundary,
          });

          compareMap.addLayer({
            id: "compare-buurt-fill",
            type: "fill",
            source: "compare-buurt",
            paint: {
              "fill-color": "#3b82f6",
              "fill-opacity": 0.1,
            },
          });

          compareMap.addLayer({
            id: "compare-buurt-outline",
            type: "line",
            source: "compare-buurt",
            paint: {
              "line-color": "#3b82f6",
              "line-width": 3,
            },
          });
        } else {
          const src = compareMap.getSource("compare-buurt");
          src.setData(compareBuurtBoundary);
        }
      }
    });

    compareMapRef.current = compareMap;

    // Sync zoom with main map if it exists
    if (mapRef.current) {
      syncMaps(compareMap, mapRef.current);
      syncMaps(mapRef.current, compareMap);
    }

    return () => {
      if (compareMapRef.current) {
        compareMapRef.current.remove();
        compareMapRef.current = null;
      }
    };
  }, [compareResult?.coords, compareResult?.geometry, result?.coords, compareBuurtBoundary]);

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
        <AddressSearchBar
          loading={loading}
          onSelect={(address, rawDoc) => {
            setQuery(address);
            // Direct starten met zoek-functie
            handleSearchFromSuggestion(address, rawDoc);
          }}
        />

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

            {/* MAIN NEIGHBORHOOD MAP - FULL WIDTH */}
            <section className="section">
              <div className="main-map-container">
            {result.coords && (
              <div className="map-card">
                    <div ref={mapContainerRef} className="map" />
                <p className="small">
                  Benadering van de locatie. Geen juridisch kaartmateriaal.
                </p>
              </div>
            )}
              </div>
            </section>

            {/* ANALYSIS & COMPARISON SECTION */}
            <div className="analysis-section">
              <div className="analysis-layout">
                {/* Left: ML & AI insights */}
                <div className="analysis-left">
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
                                  {b.naam || b.buurt_code} – {b.gemeente}
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
                {/* Right: comparison section */}
                <div className="analysis-right">
                  {/* Buurtcijfers */}
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

                        {/* Leeftijd chart */}
                        {ageChartOptions && (
                          <div className="chart-card" style={{ marginTop: "1rem", marginBottom: "0.5rem" }}>
                            <HighchartsReact
                              highcharts={Highcharts}
                              options={ageChartOptions}
                            />
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

                        {/* Inkomen chart */}
                        {incomeChartOptions && (
                          <div className="chart-card" style={{ marginTop: "1rem", marginBottom: "0.5rem" }}>
                            <HighchartsReact
                              highcharts={Highcharts}
                              options={incomeChartOptions}
                            />
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
                            <div className="stat-label">Criminaliteit (2024)</div>
                            <div className="stat-value">
                              {formatOrNA(result.crimeData.total_crimes, nf0)}
                            </div>
                            <div className="stat-help">
                              Geregistreerde misdrijven in deze buurt (CBS Politie data).
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.seksueelGeweld != null && (
                          <div className="stat-card">
                            <div className="stat-label">Seksueel geweld</div>
                            <div
                              className="stat-value"
                              title={result.cbsStats.criminaliteitDetail ? `Opbouw:\n• Zedenmisdrijven tegen jeugdigen: ${result.cbsStats.criminaliteitDetail.seksueelGeweld_1_4_1 || 0}\n• Overige zedenmisdrijven: ${result.cbsStats.criminaliteitDetail.seksueelGeweld_1_4_2 || 0}` : ""}
                            >
                              {formatOrNA(result.cbsStats.seksueelGeweld, nf1)}
                            </div>
                            <div className="stat-help">
                              Seksueel geweld per 1.000 inwoners (CBS 2023). Klik voor details.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.geweldsMisdrijven != null && (
                          <div className="stat-card">
                            <div className="stat-label">Geweldsmisdrijven</div>
                            <div
                              className="stat-value"
                              title={result.cbsStats.criminaliteitDetail ? `Opbouw:\n• Mishandeling: ${result.cbsStats.criminaliteitDetail.geweld_1_4_4 || 0}\n• Bedreiging: ${result.cbsStats.criminaliteitDetail.geweld_1_4_3 || 0}\n• Straatroof: ${result.cbsStats.criminaliteitDetail.geweld_1_4_6 || 0}\n• Overvallen: ${result.cbsStats.criminaliteitDetail.geweld_1_4_7 || 0}\n• Moord/doodslag: ${result.cbsStats.criminaliteitDetail.geweld_1_4_5 || 0}` : ""}
                            >
                              {formatOrNA(result.cbsStats.geweldsMisdrijven, nf1)}
                            </div>
                            <div className="stat-help">
                              Geweldsmisdrijven per 1.000 inwoners (CBS 2023). Klik voor details.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.vermogensMisdrijven != null && (
                          <div className="stat-card">
                            <div className="stat-label">Vermogensmisdrijven</div>
                            <div
                              className="stat-value"
                              title={result.cbsStats.criminaliteitDetail ? `Opbouw:\n• Diefstal/inbraak woning: ${result.cbsStats.criminaliteitDetail.vermogens_1_1_1 || 0}\n• Diefstal motorvoertuigen: ${result.cbsStats.criminaliteitDetail.vermogens_1_2_1 || 0}` : ""}
                            >
                              {formatOrNA(result.cbsStats.vermogensMisdrijven, nf1)}
                            </div>
                            <div className="stat-help">
                              Vermogensmisdrijven per 1.000 inwoners (CBS 2023). Klik voor details.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.vernielingsMisdrijven != null && (
                          <div className="stat-card">
                            <div className="stat-label">Vernieling/openbare orde</div>
                            <div
                              className="stat-value"
                              title={result.cbsStats.criminaliteitDetail ? `Opbouw:\n• Vernieling zaakbeschadiging: ${result.cbsStats.criminaliteitDetail.vernieling_2_2_1 || 0}\n• Openbare orde: ${result.cbsStats.criminaliteitDetail.vernieling_3_6_4 || 0}` : ""}
                            >
                              {formatOrNA(result.cbsStats.vernielingsMisdrijven, nf1)}
                            </div>
                            <div className="stat-help">
                              Vernieling en openbare orde per 1.000 inwoners (CBS 2023). Klik voor details.
                            </div>
                          </div>
                        )}

                        {/* Voorzieningen afstanden */}
                        {result.cbsStats?.amenities?.supermarket_km != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand supermarkt</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.amenities.supermarket_km, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Gemiddelde afstand tot dichtstbijzijnde grote supermarkt.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.amenities?.huisarts_km != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand huisarts</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.amenities.huisarts_km, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Gemiddelde afstand tot dichtstbijzijnde huisartsenpraktijk.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.amenities?.school_km != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand school</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.amenities.school_km, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Gemiddelde afstand tot dichtstbijzijnde school.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.amenities?.kinderdagverblijf_km != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand kinderdagverblijf</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.amenities.kinderdagverblijf_km, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Gemiddelde afstand tot dichtstbijzijnde kinderdagverblijf.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.stedelijkheid != null && (
                          <div className="stat-card">
                            <div className="stat-label">Mate van stedelijkheid</div>
                            <div className="stat-value">
                              {result.cbsStats.stedelijkheid}/5
                            </div>
                            <div className="stat-help">
                              1=zeer sterk stedelijk, 5=niet stedelijk (CBS indeling).
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.totalHouseholds != null && (
                          <div className="stat-card">
                            <div className="stat-label">Totaal huishoudens</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.totalHouseholds, nf0)}
                            </div>
                            <div className="stat-help">
                              Totaal aantal huishoudens in deze buurt.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.avgHouseholdSize != null && (
                          <div className="stat-card">
                            <div className="stat-label">Gem. huishoudensgrootte</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.avgHouseholdSize, nf1)}
                            </div>
                            <div className="stat-help">
                              Gemiddeld aantal personen per huishouden.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.totaalMisdrijven != null && (
                          <div className="stat-card">
                            <div className="stat-label">Totaal misdrijven</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.totaalMisdrijven, nf0)}
                            </div>
                            <div className="stat-help">
                              Totaal aantal geregistreerde misdrijven per jaar.
                            </div>
                          </div>
                        )}

                        {/* Afstanden tot voorzieningen */}
                        {result.cbsStats?.afstandHuisarts != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand huisarts</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.afstandHuisarts, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Afstand tot dichtstbijzijnde huisartsenpraktijk.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.afstandSupermarkt != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand supermarkt</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.afstandSupermarkt, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Afstand tot dichtstbijzijnde grote supermarkt.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.afstandSchool != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand school</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.afstandSchool, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Afstand tot dichtstbijzijnde school.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.afstandKinderdagverblijf != null && (
                          <div className="stat-card">
                            <div className="stat-label">Afstand kinderopvang</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.afstandKinderdagverblijf, nf1)}
                              <span className="small"> km</span>
                            </div>
                            <div className="stat-help">
                              Afstand tot dichtstbijzijnde kinderdagverblijf.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.scholenBinnen3Km != null && (
                          <div className="stat-card">
                            <div className="stat-label">Scholen binnen 3km</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.scholenBinnen3Km, nf0)}
                            </div>
                            <div className="stat-help">
                              Aantal scholen binnen 3 kilometer afstand.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.pctAppartementen != null && (
                          <div className="stat-card">
                            <div className="stat-label">Woningen</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.pctAppartementen, nf1)}%
                              <span className="small"> appartementen</span>
                            </div>
                            <div className="stat-help">
                              Percentage appartementen in deze buurt.
                            </div>
                          </div>
                        )}

                        {result.cbsStats?.carsPerHousehold != null && (
                          <div className="stat-card">
                            <div className="stat-label">Auto's per huishouden</div>
                            <div className="stat-value">
                              {formatOrNA(result.cbsStats.carsPerHousehold, nf1)}
                            </div>
                            <div className="stat-help">
                              Gemiddeld aantal personenauto's per huishouden.
                            </div>
                          </div>
                        )}
              </div>
            ) : (
                      <p className="small">Geen CBS-buurtcijfers gevonden.</p>
            )}
                  </div>

                  <h2>Vergelijk met andere buurt</h2>

                  <AddressSearchBar
                    loading={compareLoading}
                    onSelect={(address, rawDoc) => {
                      setCompareQuery(address);
                      // Direct starten met zoek-functie
                      handleCompareSearchFromSuggestion(address, rawDoc);
                    }}
                  />

                  {compareError && <div className="error">{compareError}</div>}

                  {compareResult && compareResult.coords && (
                    <div className="map-card" style={{ marginTop: "1rem" }}>
                      <div ref={compareMapContainerRef} className="compare-map" />
            <p className="small">
                        Vergelijkingslocatie. Geen juridisch kaartmateriaal.
            </p>
              </div>
            )}

                  {/* Vergelijkingspanel - alleen tonen als beide buurten geladen zijn */}
                  {compareResult && result && (
                    <div className="panel-block" style={{ marginTop: "1rem" }}>
                      <h3>Vergelijking</h3>

                      {/* Radar chart voor visuele vergelijking */}
                      {comparisonRadarOptions && (
                        <div className="chart-card" style={{ marginBottom: "1.5rem" }}>
                          <HighchartsReact
                            highcharts={Highcharts}
                            options={comparisonRadarOptions}
                          />
                        </div>
                      )}

                      <div className="comparison-grid">
                        {/* Bevolking vergelijking */}
                        {(result.cbsStats?.population != null || compareResult.cbsStats?.population != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Inwoners</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.population, nf0)}
      </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.population, nf0)}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.population && compareResult.cbsStats?.population
                                ? `${result.cbsStats.population > compareResult.cbsStats.population ? '+' : ''}${(result.cbsStats.population - compareResult.cbsStats.population).toLocaleString('nl-NL')}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Dichtheid vergelijking */}
                        {(result.cbsStats?.density != null || compareResult.cbsStats?.density != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Dichtheid (/km²)</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.density, nf0)}
    </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.density, nf0)}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.density && compareResult.cbsStats?.density
                                ? `${result.cbsStats.density > compareResult.cbsStats.density ? '+' : ''}${(result.cbsStats.density - compareResult.cbsStats.density).toFixed(0)}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Inkomen vergelijking */}
                        {(result.cbsStats?.incomePerPerson != null || compareResult.cbsStats?.incomePerPerson != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Inkomen p.p. (×1000€)</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.incomePerPerson, nf1)}
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.incomePerPerson, nf1)}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.incomePerPerson && compareResult.cbsStats?.incomePerPerson
                                ? `${result.cbsStats.incomePerPerson > compareResult.cbsStats.incomePerPerson ? '+' : ''}${(result.cbsStats.incomePerPerson - compareResult.cbsStats.incomePerPerson).toFixed(1)}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Stedelijkheid vergelijking */}
                        {(result.cbsStats?.stedelijkheid != null || compareResult.cbsStats?.stedelijkheid != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Stedelijkheid</div>
                            <div className="comparison-value main-value">
                              {result.cbsStats?.stedelijkheid ? `${result.cbsStats.stedelijkheid}/5` : 'n.v.t.'}
                            </div>
                            <div className="comparison-value compare-value">
                              {compareResult.cbsStats?.stedelijkheid ? `${compareResult.cbsStats.stedelijkheid}/5` : 'n.v.t.'}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.stedelijkheid && compareResult.cbsStats?.stedelijkheid
                                ? result.cbsStats.stedelijkheid === compareResult.cbsStats.stedelijkheid ? 'gelijk' : `${result.cbsStats.stedelijkheid > compareResult.cbsStats.stedelijkheid ? 'steds.' : 'minder steds.'}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Afstand huisarts vergelijking */}
                        {(result.cbsStats?.afstandHuisarts != null || compareResult.cbsStats?.afstandHuisarts != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Huisarts</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.afstandHuisarts, nf1)} km
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.afstandHuisarts, nf1)} km
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.afstandHuisarts && compareResult.cbsStats?.afstandHuisarts
                                ? `${result.cbsStats.afstandHuisarts < compareResult.cbsStats.afstandHuisarts ? 'dichterbij' : result.cbsStats.afstandHuisarts > compareResult.cbsStats.afstandHuisarts ? 'verder weg' : 'gelijk'}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Afstand supermarkt vergelijking */}
                        {(result.cbsStats?.afstandSupermarkt != null || compareResult.cbsStats?.afstandSupermarkt != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Supermarkt</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.afstandSupermarkt, nf1)} km
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.afstandSupermarkt, nf1)} km
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.afstandSupermarkt && compareResult.cbsStats?.afstandSupermarkt
                                ? `${result.cbsStats.afstandSupermarkt < compareResult.cbsStats.afstandSupermarkt ? 'dichterbij' : result.cbsStats.afstandSupermarkt > compareResult.cbsStats.afstandSupermarkt ? 'verder weg' : 'gelijk'}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Afstand school vergelijking */}
                        {(result.cbsStats?.afstandSchool != null || compareResult.cbsStats?.afstandSchool != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">School</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.afstandSchool, nf1)} km
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.afstandSchool, nf1)} km
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.afstandSchool && compareResult.cbsStats?.afstandSchool
                                ? `${result.cbsStats.afstandSchool < compareResult.cbsStats.afstandSchool ? 'dichterbij' : result.cbsStats.afstandSchool > compareResult.cbsStats.afstandSchool ? 'verder weg' : 'gelijk'}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Huishoudens vergelijking */}
                        {(result.cbsStats?.totalHouseholds != null || compareResult.cbsStats?.totalHouseholds != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Huishoudens</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.totalHouseholds, nf0)}
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.totalHouseholds, nf0)}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.totalHouseholds && compareResult.cbsStats?.totalHouseholds
                                ? `${result.cbsStats.totalHouseholds > compareResult.cbsStats.totalHouseholds ? '+' : ''}${(result.cbsStats.totalHouseholds - compareResult.cbsStats.totalHouseholds).toLocaleString('nl-NL')}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Huishoudensgrootte vergelijking */}
                        {(result.cbsStats?.avgHouseholdSize != null || compareResult.cbsStats?.avgHouseholdSize != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Gem. gezin</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.avgHouseholdSize, nf1)}
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.avgHouseholdSize, nf1)}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.avgHouseholdSize && compareResult.cbsStats?.avgHouseholdSize
                                ? `${result.cbsStats.avgHouseholdSize > compareResult.cbsStats.avgHouseholdSize ? '+' : ''}${(result.cbsStats.avgHouseholdSize - compareResult.cbsStats.avgHouseholdSize).toFixed(1)}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Totaal misdrijven vergelijking */}
                        {(result.cbsStats?.totaalMisdrijven != null || compareResult.cbsStats?.totaalMisdrijven != null) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Misdaad totaal</div>
                            <div className="comparison-value main-value">
                              {formatOrNA(result.cbsStats?.totaalMisdrijven, nf0)}
                            </div>
                            <div className="comparison-value compare-value">
                              {formatOrNA(compareResult.cbsStats?.totaalMisdrijven, nf0)}
                            </div>
                            <div className="comparison-diff">
                              {result.cbsStats?.totaalMisdrijven && compareResult.cbsStats?.totaalMisdrijven
                                ? `${result.cbsStats.totaalMisdrijven > compareResult.cbsStats.totaalMisdrijven ? '+' : ''}${(result.cbsStats.totaalMisdrijven - compareResult.cbsStats.totaalMisdrijven).toLocaleString('nl-NL')}`
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}

                        {/* Cluster vergelijking */}
                        {(clusterInfo?.label || compareClusterInfo?.label) && (
                          <div className="comparison-row">
                            <div className="comparison-label">Buurt type</div>
                            <div className="comparison-value main-value">
                              {clusterInfo?.label || 'n.v.t.'}
                            </div>
                            <div className="comparison-value compare-value">
                              {compareClusterInfo?.label || 'n.v.t.'}
                            </div>
                            <div className="comparison-diff">
                              {clusterInfo?.label && compareClusterInfo?.label
                                ? clusterInfo.label === compareClusterInfo.label ? 'gelijk' : 'verschilt'
                                : 'n.v.t.'}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
