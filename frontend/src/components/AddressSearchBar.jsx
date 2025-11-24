import React, { useState, useEffect, useRef } from "react";

const SUGGEST_URL =
  "https://api.pdok.nl/bzk/locatieserver/search/v3_1/suggest";

export default function AddressSearchBar({ onSelect, loading }) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [open, setOpen] = useState(false);

  const boxRef = useRef(null);

  // --- Fetch suggestions ---
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

        setSuggestions(docs);
        setOpen(true);
        setActiveIndex(-1);
      } catch (e) {
        // ignore abort
      }
    }, 200); // debounce 200–250ms

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [query]);

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
      `}</style>
    </div>
  );
}
