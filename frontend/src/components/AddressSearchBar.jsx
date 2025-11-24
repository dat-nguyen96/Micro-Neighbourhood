import React, { useState, useEffect, useRef } from "react";

const SUGGEST_URL =
  "https://api.pdok.nl/bzk/locatieserver/search/v3_1/suggest";

export default function AddressSearchBar({ onSelect, loading }) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [open, setOpen] = useState(false);
  const [buurtNames, setBuurtNames] = useState(new Map());

  const boxRef = useRef(null);

  // --- Fetch suggestions with buurt names ---
  useEffect(() => {
    if (!query || query.trim().length < 2) {
      setSuggestions([]);
      setOpen(false);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      try {
        const resp = await fetch(
          `${SUGGEST_URL}?q=${encodeURIComponent(query)}&rows=8`,
          { signal: controller.signal }
        );
        if (!resp.ok) return;

        const json = await resp.json();
        const docs = json.response?.docs || [];

        // Enrich each suggestion with buurt info
        const enrichedDocs = await Promise.all(
          docs.map(async (doc) => {
            try {
              // Get full address details to find buurt code
              const freeResp = await fetch(
                `https://api.pdok.nl/bzk/locatieserver/search/v3_1/free?q=${encodeURIComponent(doc.weergavenaam)}&rows=1&fq=type:adres`,
                { signal: controller.signal }
              );
              if (!freeResp.ok) return doc;

              const freeJson = await freeResp.json();
              const freeDocs = freeJson.response?.docs || [];
              if (freeDocs.length === 0) return doc;

              const freeDoc = freeDocs[0];
              const buurtCode = freeDoc.buurtcode || freeDoc.wijkcode || freeDoc.gemeentecode;

              if (buurtCode) {
                // Check cache first
                if (buurtNames.has(buurtCode)) {
                  return { ...doc, _buurtNaam: buurtNames.get(buurtCode), _buurtCode: buurtCode };
                }

                // Fetch buurt info
                const clusterResp = await fetch(`/api/buurt-cluster?buurt_code=${encodeURIComponent(buurtCode)}`);
                if (clusterResp.ok) {
                  const clusterInfo = await clusterResp.json();
                  const buurtNaam = clusterInfo.buurt_naam;

                  // Update cache
                  setBuurtNames(prev => new Map(prev).set(buurtCode, buurtNaam));

                  return { ...doc, _buurtNaam: buurtNaam, _buurtCode: buurtCode };
                }
              }
            } catch (e) {
              // ignore errors, just return doc without buurt info
            }
            return doc;
          })
        );

        setSuggestions(enrichedDocs);
        setOpen(true);
        setActiveIndex(-1);
      } catch (e) {
        // ignore abort
      }
    }, 300); // debounce 300ms for API calls

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [query, buurtNames]);

  // --- Close dropdown when clicking outside ---
  useEffect(() => {
    const handler = (e) => {
      if (boxRef.current && !boxRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);

  // --- Handle keyboard navigation ---
  function onKeyDown(e) {
    if (!open || suggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) =>
        i <= 0 ? suggestions.length - 1 : i - 1
      );
    } else if (e.key === "Enter") {
      if (activeIndex >= 0) {
        e.preventDefault();
        chooseSuggestion(suggestions[activeIndex]);
      }
    }
  }

  // --- When user clicks a suggestion ---
  function chooseSuggestion(s) {
    const label =
      s.weergavenaam ||
      `${s.straatnaam || ""} ${s.huisnummer || ""} ${s.woonplaatsnaam || ""}`;

    setQuery(label);
    setOpen(false);
    onSelect(label, s); // pass raw PDOK doc
  }

  return (
    <div className="address-search" ref={boxRef} style={{ position: "relative" }}>
      <div className="form-row">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          type="text"
          placeholder='Bijv. "Damrak 1, Amsterdam"'
        />
        <button disabled={loading} onClick={() => onSelect(query)}>
          {loading ? "Bezig..." : "Bekijk buurt"}
        </button>
      </div>

      {open && suggestions.length > 0 && (
        <div className="dropdown">
          {suggestions.map((s, i) => {
            const label =
              s.weergavenaam ||
              `${s.straatnaam || ""} ${s.huisnummer || ""} ${
                s.woonplaatsnaam || ""
              }`;

            return (
              <div
                key={s.id}
                className={
                  "dropdown-item " + (i === activeIndex ? "active" : "")
                }
                onMouseDown={() => chooseSuggestion(s)}
              >
                {label}
                {s.postcode && (
                  <span className="extra"> • {s.postcode}</span>
                )}
                {s._buurtNaam && (
                  <div className="buurt-label">{s._buurtNaam}</div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Basic styling */}
      <style>{`
        .dropdown {
          position: absolute;
          top: 3.2rem;
          left: 0;
          right: 0;
          background: #0f172a;
          border: 1px solid #1e293b;
          border-radius: 6px;
          max-height: 260px;
          overflow-y: auto;
          z-index: 100;
        }
        .dropdown-item {
          padding: 0.55rem 0.8rem;
          cursor: pointer;
          border-bottom: 1px solid #1e293b;
          color: #cbd5e1;
        }
        .dropdown-item:last-child {
          border-bottom: none;
        }
        .dropdown-item.active,
        .dropdown-item:hover {
          background: #1e293b;
          color: #fff;
        }
        .dropdown-item .extra {
          color: #64748b;
          margin-left: 6px;
          font-size: 0.8rem;
        }
        .dropdown-item .buurt-label {
          color: #3b82f6;
          font-size: 0.75rem;
          font-weight: 500;
          margin-top: 2px;
        }
      `}</style>
    </div>
  );
}
