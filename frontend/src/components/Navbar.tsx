import React, { useEffect, useState } from "react";
import {
  Activity,
  ShieldCheck,
  Database,
  Zap,
  UploadCloud,
  ChevronDown,
  Sparkles,
  FileSpreadsheet,
  Check,
  Download,
  ShoppingBag,
  Layers,
  ArrowUpRight,
} from "lucide-react";
import { api } from "../services/api";
import { ActiveDatasetContext } from "../types";

interface NavbarProps {
  currentTab: string;
  activeContext: ActiveDatasetContext | null;
  onOpenUploadModal: () => void;
  onSwitchMode: (mode: string, datasetId?: string, datasetLabel?: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  activeContext,
  onOpenUploadModal,
  onSwitchMode,
}) => {
  const [health, setHealth] = useState<{ status: string; total_customers: number; trained_models: number } | null>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [presets, setPresets] = useState<any[]>([]);
  const [showHistoryDropdown, setShowHistoryDropdown] = useState(false);

  useEffect(() => {
    api.getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "CONNECTING", total_customers: 0, trained_models: 0 }));
  }, []);

  const loadDatasetHistory = () => {
    Promise.all([
      api.getPresets().catch(() => []),
      api.getDatasets().catch(() => []),
    ]).then(([pr, ds]) => {
      setPresets(pr || []);
      setDatasets(ds || []);
    });
  };

  const currentDatasetLabel = activeContext?.mode_label || activeContext?.dataset_name || "Amazon E-Commerce Benchmark";

  return (
    <header style={{
      height: "64px",
      borderBottom: "1px solid var(--border-subtle)",
      background: "rgba(9, 13, 22, 0.85)",
      backdropFilter: "blur(12px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 24px",
      zIndex: 20
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: "34px",
            height: "34px",
            borderRadius: "10px",
            background: "linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 16px rgba(59, 130, 246, 0.4)"
          }}>
            <Zap size={20} color="#FFFFFF" />
          </div>
          <div>
            <h1 style={{ fontSize: "1.15rem", fontWeight: 800, letterSpacing: "-0.02em", color: "#FFFFFF" }}>
              CustomerPulse <span style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>AI</span>
            </h1>
          </div>
        </div>

        <span style={{ color: "var(--border-subtle)", margin: "0 4px" }}>|</span>

        {/* Dataset Switcher Pill */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => {
              loadDatasetHistory();
              setShowHistoryDropdown(!showHistoryDropdown);
            }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "6px 14px",
              borderRadius: "8px",
              background: activeContext?.active_mode === "UPLOAD" ? "rgba(6, 182, 212, 0.12)" : "rgba(59, 130, 246, 0.12)",
              border: `1px solid ${activeContext?.active_mode === "UPLOAD" ? "rgba(6, 182, 212, 0.35)" : "rgba(59, 130, 246, 0.3)"}`,
              color: activeContext?.active_mode === "UPLOAD" ? "var(--accent-cyan)" : "#93C5FD",
              fontSize: "0.8rem",
              fontWeight: 700,
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            <Database size={14} />
            <span style={{ maxWidth: "260px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {currentDatasetLabel}
            </span>
            <ChevronDown size={14} />
          </button>

          {/* Dataset Switcher Dropdown */}
          {showHistoryDropdown && (
            <div
              style={{
                position: "absolute",
                top: "120%",
                left: 0,
                width: "360px",
                background: "#0E131F",
                border: "1px solid var(--border-subtle)",
                borderRadius: "12px",
                padding: "10px",
                boxShadow: "0 15px 40px rgba(0, 0, 0, 0.9)",
                zIndex: 50,
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 8px" }}>
                <span style={{ fontSize: "0.7rem", fontWeight: 800, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  E-Commerce Benchmarks
                </span>
                <span style={{ fontSize: "0.68rem", color: "var(--accent-cyan)" }}>1-Click Grounding</span>
              </div>

              {/* Standard E-Commerce Benchmark Presets */}
              {presets.map((p) => {
                const isActive = (activeContext?.active_mode === "DEMO" && (activeContext?.dataset_name === p.filename || activeContext?.mode_label?.includes(p.name))) ||
                  (activeContext?.dataset_name === p.filename);

                return (
                  <div
                    key={p.preset_id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "8px 10px",
                      borderRadius: "8px",
                      background: isActive ? "rgba(59, 130, 246, 0.15)" : "rgba(255, 255, 255, 0.02)",
                      border: isActive ? "1px solid rgba(59, 130, 246, 0.4)" : "1px solid var(--border-subtle)",
                    }}
                  >
                    <button
                      onClick={() => {
                        onSwitchMode(p.preset_id, p.filename, p.name);
                        setShowHistoryDropdown(false);
                      }}
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "#FFFFFF",
                        cursor: "pointer",
                        textAlign: "left",
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        gap: "2px",
                      }}
                    >
                      <div style={{ fontWeight: 700, fontSize: "0.82rem", color: isActive ? "var(--accent-cyan)" : "#FFFFFF" }}>
                        {p.name}
                      </div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                        {p.row_count?.toLocaleString()} orders &middot; {p.domain}
                      </div>
                    </button>

                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <a
                        href={api.getPresetDownloadUrl(p.preset_id)}
                        download={p.filename}
                        title={`Download ${p.filename}`}
                        style={{
                          padding: "4px 8px",
                          borderRadius: "6px",
                          background: "rgba(255, 255, 255, 0.06)",
                          color: "var(--text-secondary)",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                          fontSize: "0.68rem",
                          textDecoration: "none",
                        }}
                      >
                        <Download size={12} />
                        <span>CSV</span>
                      </a>
                      {isActive && <Check size={16} color="var(--accent-cyan)" />}
                    </div>
                  </div>
                );
              })}

              {/* Uploaded Datasets History */}
              {datasets.length > 0 && (
                <>
                  <div style={{ fontSize: "0.7rem", fontWeight: 800, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", padding: "8px 8px 2px 8px" }}>
                    Custom Uploaded Datasets
                  </div>

                  {datasets.map((d) => (
                    <button
                      key={d.dataset_id}
                      onClick={() => {
                        onSwitchMode("UPLOAD", d.dataset_id, d.filename);
                        setShowHistoryDropdown(false);
                      }}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "8px 10px",
                        borderRadius: "8px",
                        background: activeContext?.dataset_id === d.dataset_id ? "rgba(6, 182, 212, 0.15)" : "transparent",
                        border: "none",
                        color: "#FFFFFF",
                        fontSize: "0.8rem",
                        cursor: "pointer",
                        textAlign: "left",
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", maxWidth: "240px" }}>
                          {d.filename}
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                          {d.row_count?.toLocaleString()} rows &middot; {d.dataset_mode}
                        </div>
                      </div>
                      {activeContext?.dataset_id === d.dataset_id && <Check size={14} color="var(--accent-cyan)" />}
                    </button>
                  ))}
                </>
              )}
            </div>
          )}
        </div>

        <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)", fontWeight: 500 }}>
          {currentTab.replace("-", " ").toUpperCase()}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        {/* Upload CSV Primary Action Button */}
        <button
          className="btn-primary"
          onClick={onOpenUploadModal}
          style={{ padding: "6px 14px", fontSize: "0.8rem" }}
        >
          <UploadCloud size={15} />
          <span>Upload CSV</span>
        </button>

        {/* Engine Status */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "4px 10px",
          borderRadius: "8px",
          background: health?.status === "HEALTHY" ? "rgba(16, 185, 129, 0.1)" : "rgba(245, 158, 11, 0.1)",
          border: `1px solid ${health?.status === "HEALTHY" ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)"}`,
          fontSize: "0.75rem",
          fontWeight: 600,
          color: health?.status === "HEALTHY" ? "var(--accent-emerald)" : "var(--accent-amber)"
        }}>
          <Activity size={13} />
          <span>{health?.status === "HEALTHY" ? "ONLINE" : "CONNECTING"}</span>
        </div>

        {/* Role Badge */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "4px 10px",
          borderRadius: "8px",
          background: "rgba(59, 130, 246, 0.12)",
          border: "1px solid rgba(59, 130, 246, 0.3)",
          fontSize: "0.75rem",
          fontWeight: 700,
          color: "#93C5FD"
        }}>
          <ShieldCheck size={14} />
          <span>ADMIN</span>
        </div>
      </div>
    </header>
  );
};
