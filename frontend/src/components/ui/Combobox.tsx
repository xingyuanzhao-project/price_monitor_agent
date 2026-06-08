/**
 * Searchable combobox – a text input with a filtered dropdown list.
 *
 * Supports keyboard navigation (ArrowUp / ArrowDown / Enter / Escape),
 * click-to-select, and free-text entry when no option matches.
 *
 * Ported from nocode-workflow/gui/src/components/ui/combobox.tsx.
 */

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  type KeyboardEvent,
} from "react";

export interface ComboboxOption {
  value: string;
  label: string;
}

export interface ComboboxProps {
  options: ComboboxOption[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  loading?: boolean;
  loadingText?: string;
  errorText?: string;
}

export function Combobox({
  options,
  value,
  onChange,
  placeholder,
  disabled,
  className,
  loading,
  loadingText = "Loading…",
  errorText,
}: ComboboxProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlightIdx, setHighlightIdx] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const displayValue = value || "";

  const filtered = query
    ? options.filter((o) => {
        const q = query.toLowerCase();
        return (
          o.value.toLowerCase().includes(q) ||
          o.label.toLowerCase().includes(q)
        );
      })
    : options;

  useEffect(() => {
    setHighlightIdx(-1);
  }, [query]);

  useEffect(() => {
    if (highlightIdx >= 0 && listRef.current) {
      const item = listRef.current.children[highlightIdx] as
        | HTMLElement
        | undefined;
      item?.scrollIntoView({ block: "nearest" });
    }
  }, [highlightIdx]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
        setQuery("");
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const select = useCallback(
    (val: string) => {
      onChange(val);
      setOpen(false);
      setQuery("");
      inputRef.current?.blur();
    },
    [onChange],
  );

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        setOpen(true);
        e.preventDefault();
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setHighlightIdx((prev) =>
          prev < filtered.length - 1 ? prev + 1 : 0,
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightIdx((prev) =>
          prev > 0 ? prev - 1 : filtered.length - 1,
        );
        break;
      case "Enter":
        e.preventDefault();
        if (highlightIdx >= 0 && highlightIdx < filtered.length) {
          select(filtered[highlightIdx].value);
        } else if (query.length > 0) {
          select(query);
        }
        break;
      case "Escape":
        e.preventDefault();
        setOpen(false);
        setQuery("");
        break;
    }
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ position: "relative" }}
    >
      <input
        ref={inputRef}
        type="text"
        autoComplete="off"
        className="form-input"
        placeholder={placeholder}
        disabled={disabled}
        value={open ? query : displayValue}
        onFocus={() => {
          setOpen(true);
          setQuery("");
        }}
        onChange={(e) => {
          setQuery(e.target.value);
          if (!open) setOpen(true);
        }}
        onKeyDown={handleKeyDown}
        style={{ paddingRight: "1.75rem" }}
      />
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          position: "absolute",
          right: "0.5rem",
          top: "50%",
          transform: "translateY(-50%)",
          pointerEvents: "none",
          opacity: 0.5,
        }}
      >
        <path d="m6 9 6 6 6-6" />
      </svg>

      {open && (
        <ul
          ref={listRef}
          className="combobox-dropdown"
          style={{
            position: "absolute",
            zIndex: 50,
            marginTop: "0.25rem",
            maxHeight: "15rem",
            width: "100%",
            overflowY: "auto",
            borderRadius: "0.375rem",
            border: "1px solid var(--border-color, #e2e8f0)",
            backgroundColor: "var(--bg-panel, #fff)",
            fontSize: "0.875rem",
            boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)",
            listStyle: "none",
            padding: 0,
            margin: 0,
          }}
        >
          {loading ? (
            <li style={{ padding: "0.375rem 0.5rem", opacity: 0.6 }}>
              {loadingText}
            </li>
          ) : errorText ? (
            <li style={{ padding: "0.375rem 0.5rem", color: "#ef4444" }}>
              {errorText}
            </li>
          ) : filtered.length === 0 ? (
            <li style={{ padding: "0.375rem 0.5rem", opacity: 0.6 }}>
              {query
                ? "No matches \u2014 press Enter to use custom value"
                : "No models available"}
            </li>
          ) : (
            filtered.map((opt, idx) => (
              <li
                key={opt.value}
                style={{
                  padding: "0.375rem 0.5rem",
                  cursor: "pointer",
                  backgroundColor:
                    idx === highlightIdx
                      ? "var(--accent-bg, #f1f5f9)"
                      : "transparent",
                  fontWeight: opt.value === value ? 600 : 400,
                }}
                onMouseDown={(e) => {
                  e.preventDefault();
                  select(opt.value);
                }}
                onMouseEnter={() => setHighlightIdx(idx)}
              >
                <span
                  style={{
                    display: "block",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {opt.value}
                </span>
                {opt.label !== opt.value && (
                  <span
                    style={{
                      display: "block",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontSize: "0.75rem",
                      opacity: 0.6,
                    }}
                  >
                    {opt.label}
                  </span>
                )}
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
