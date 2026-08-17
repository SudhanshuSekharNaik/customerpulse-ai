import React, { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  X,
  ArrowRight,
  Sparkles,
  Layers,
  Database,
  Sliders,
  ShieldCheck,
  Info,
  Activity,
  Download,
  ShoppingBag,
} from "lucide-react";
import { api } from "../services/api";
import { UploadProfileResponse, DetectedColumn } from "../types";

interface UniversalUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAnalysisComplete: (report: any) => void;
  onStartAnalysis?: (datasetName: string) => void;
  onSelectPreset?: (presetId: string, filename: string, label: string) => void;
}

export const UniversalUploadModal: React.FC<UniversalUploadModalProps> = ({
  isOpen,
  onClose,
  onAnalysisComplete,
  onStartAnalysis,
  onSelectPreset,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [profiling, setProfiling] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadData, setUploadData] = useState<UploadProfileResponse | null>(null);
  const [mappingOverrides, setMappingOverrides] = useState<Record<string, string>>({});
  const [presets, setPresets] = useState<any[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getPresets().then(setPresets).catch(() => []);
  }, []);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processFile = (uploadedFile: File) => {
    if (!uploadedFile.name.endsWith(".csv") && !uploadedFile.name.endsWith(".txt")) {
      setError("Please upload a valid CSV file (.csv format).");
      return;
    }
    setError(null);
    setFile(uploadedFile);
    setProfiling(true);

    api.uploadCsv(uploadedFile)
      .then((res: UploadProfileResponse) => {
        setUploadData(res);
        setMappingOverrides(res.initial_mapping || {});
        setProfiling(false);
      })
      .catch((err: any) => {
        setError(err.message || "Failed to profile uploaded CSV.");
        setProfiling(false);
      });
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleMappingChange = (colName: string, newSemanticType: string) => {
    setMappingOverrides((prev) => ({
      ...prev,
      [colName]: newSemanticType,
    }));
  };

  const handleConfirmAndAnalyze = () => {
    if (!uploadData) return;
    const targetName = uploadData.filename || file?.name || "Custom Uploaded Dataset";
    if (onStartAnalysis) {
      onStartAnalysis(targetName);
    }
    onClose();

    api.confirmAndAnalyze(uploadData.dataset_id, mappingOverrides)
      .then((res) => {
        onAnalysisComplete(res.report);
      })
      .catch((err: any) => {
        console.error("Adaptive analysis error:", err);
      });
  };

  const SEMANTIC_OPTIONS = [
    "CUSTOMER_ID",
    "USER_ID",
    "ACCOUNT_ID",
    "TRANSACTION_ID",
    "TIMESTAMP",
    "DATE",
    "PRODUCT_ID",
    "PRODUCT_CATEGORY",
    "EVENT_TYPE",
    "QUANTITY",
    "PRICE",
    "REVENUE",
    "ORDER_VALUE",
    "SESSION_ID",
    "CAMPAIGN_ID",
    "TREATMENT",
    "CONVERSION",
    "CHURN_LABEL",
    "TARGET",
    "COUNTRY",
    "DEVICE",
    "CHANNEL",
    "RATING",
    "RETURNED",
    "EMAIL",
    "PHONE",
    "NAME",
    "UNKNOWN",
  ];

  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      width: "100vw",
      height: "100vh",
      background: "rgba(0, 0, 0, 0.8)",
      backdropFilter: "blur(12px)",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      zIndex: 200,
      padding: "20px",
    }}>
      <div className="glass-card" style={{
        width: "95%",
        maxWidth: "1050px",
        maxHeight: "92vh",
        overflowY: "auto",
        padding: "28px",
        background: "#0C101A",
        border: "1px solid rgba(59, 130, 246, 0.4)",
        boxShadow: "0 25px 60px rgba(0, 0, 0, 0.9)",
      }}>
        {/* Modal Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "16px", marginBottom: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ width: "42px", height: "42px", borderRadius: "12px", background: "linear-gradient(135deg, #3B82F6 0%, #06B6D4 100%)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <UploadCloud size={24} color="#FFFFFF" />
            </div>
            <div>
              <h2 style={{ fontSize: "1.3rem", fontWeight: 800, color: "#FFFFFF" }}>
                Universal Data Mode &mdash; CSV Intelligence Ingestion
              </h2>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                Upload any e-commerce or customer CSV to automatically infer schemas, detect capabilities, and train grounded ML models.
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
          >
            <X size={24} />
          </button>
        </div>

        {error && (
          <div style={{ padding: "12px 16px", borderRadius: "8px", background: "rgba(239, 68, 68, 0.15)", border: "1px solid rgba(239, 68, 68, 0.4)", color: "#FCA5A5", fontSize: "0.85rem", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* STEP 1: Upload Dropzone if no data */}
        {!uploadData && !profiling && (
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: dragActive ? "2px dashed var(--accent-blue)" : "2px dashed var(--border-subtle)",
                borderRadius: "16px",
                padding: "40px 20px",
                textAlign: "center",
                background: dragActive ? "rgba(59, 130, 246, 0.08)" : "rgba(255, 255, 255, 0.02)",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.txt"
                onChange={handleFileInputChange}
                style={{ display: "none" }}
              />
              <div style={{ width: "56px", height: "56px", borderRadius: "16px", background: "rgba(59, 130, 246, 0.15)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 12px auto" }}>
                <FileSpreadsheet size={28} color="var(--accent-blue)" />
              </div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
                Drag &amp; Drop Your CSV Here, or <span style={{ color: "var(--accent-cyan)" }}>Browse Files</span>
              </h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", maxWidth: "500px", margin: "0 auto" }}>
                Supports e-commerce orders (Amazon, Flipkart, Myntra), clickstream event logs, marketing campaigns, or generic customer tables.
              </p>
            </div>

            {/* Quick Benchmark Templates Section */}
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                <Sparkles size={16} color="var(--accent-cyan)" />
                <h4 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF" }}>
                  Or Test with Standardized E-Commerce Benchmarks
                </h4>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "12px" }}>
                {presets.map((p) => (
                  <div
                    key={p.preset_id}
                    style={{
                      padding: "14px 16px",
                      borderRadius: "10px",
                      background: "rgba(255, 255, 255, 0.02)",
                      border: "1px solid var(--border-subtle)",
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "space-between",
                      gap: "10px",
                    }}
                  >
                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                        <strong style={{ fontSize: "0.9rem", color: "#FFFFFF" }}>{p.name}</strong>
                        <span style={{ fontSize: "0.7rem", color: "var(--accent-cyan)", background: "rgba(6, 182, 212, 0.1)", padding: "2px 6px", borderRadius: "4px" }}>
                          {p.row_count?.toLocaleString()} rows
                        </span>
                      </div>
                      <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: "1.4" }}>
                        {p.description}
                      </p>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px" }}>
                      {onSelectPreset && (
                        <button
                          className="btn-secondary"
                          onClick={() => {
                            onSelectPreset(p.preset_id, p.filename, p.name);
                            onClose();
                          }}
                          style={{ padding: "4px 10px", fontSize: "0.75rem", flex: 1, justifyContent: "center" }}
                        >
                          <Database size={13} />
                          <span>Load Benchmark</span>
                        </button>
                      )}
                      <a
                        href={api.getPresetDownloadUrl(p.preset_id)}
                        download={p.filename}
                        className="btn-secondary"
                        style={{ padding: "4px 10px", fontSize: "0.75rem", textDecoration: "none", color: "var(--text-secondary)" }}
                      >
                        <Download size={13} />
                        <span>Download CSV</span>
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Profiling State */}
        {profiling && (
          <div style={{ padding: "60px 20px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <Activity className="animate-spin" size={36} color="var(--accent-blue)" />
            <h4 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#FFFFFF" }}>
              Profiling Dataset &amp; Inferring Semantic Schema...
            </h4>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
              Analyzing columns, verifying data types, detecting ID entities, and calculating capability matrix.
            </p>
          </div>
        )}

        {/* STEP 2: Profile Metrics + Column Mapping Confirmation Screen */}
        {uploadData && !profiling && (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* Dataset Profile Metric Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
              <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Filename</div>
                <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {uploadData.filename}
                </div>
                <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                  {(uploadData.profile.file_size_bytes / 1024).toFixed(1)} KB &middot; {uploadData.profile.encoding}
                </div>
              </div>

              <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Dataset Mode</div>
                <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                  {uploadData.capabilities.mode_label}
                </div>
                <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                  Domain: {uploadData.capabilities.domain}
                </div>
              </div>

              <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Dimensions</div>
                <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF" }}>
                  {uploadData.profile.total_rows.toLocaleString()} &times; {uploadData.profile.total_columns}
                </div>
                <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                  Rows &times; Columns
                </div>
              </div>

              <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Missing Values</div>
                <div style={{ fontSize: "0.95rem", fontWeight: 700, color: uploadData.profile.missing_rate_pct < 5 ? "var(--accent-emerald)" : "var(--accent-amber)" }}>
                  {uploadData.profile.missing_rate_pct.toFixed(2)}%
                </div>
                <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                  {uploadData.profile.status}
                </div>
              </div>
            </div>

            {/* Inferred Capabilities Grid */}
            <div style={{ padding: "16px", borderRadius: "10px", background: "rgba(59, 130, 246, 0.04)", border: "1px solid rgba(59, 130, 246, 0.2)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                <h4 style={{ fontSize: "0.88rem", fontWeight: 700, color: "#FFFFFF", display: "flex", alignItems: "center", gap: "6px" }}>
                  <Layers size={16} color="var(--accent-cyan)" />
                  <span>Module Capability Matrix (Zero Fabrication Gating)</span>
                </h4>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Modules without required data are safely locked with clear guidance.
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px" }}>
                {uploadData.capabilities.enabled_capabilities?.map((cap) => (
                  <div
                    key={cap.name}
                    style={{
                      padding: "8px 10px",
                      borderRadius: "6px",
                      background: "rgba(16, 185, 129, 0.08)",
                      border: "1px solid rgba(16, 185, 129, 0.25)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <span style={{ fontSize: "0.78rem", fontWeight: 600, color: "#FFFFFF" }}>
                      {cap.name}
                    </span>
                    <span style={{
                      fontSize: "0.68rem",
                      fontWeight: 700,
                      padding: "2px 6px",
                      borderRadius: "4px",
                      background: "rgba(16, 185, 129, 0.2)",
                      color: "var(--accent-emerald)",
                    }}>
                      ENABLED
                    </span>
                  </div>
                ))}

                {uploadData.capabilities.disabled_capabilities?.map((cap) => (
                  <div
                    key={cap.name}
                    style={{
                      padding: "8px 10px",
                      borderRadius: "6px",
                      background: "rgba(255, 255, 255, 0.02)",
                      border: "1px solid var(--border-subtle)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <span style={{ fontSize: "0.78rem", fontWeight: 500, color: "var(--text-muted)" }}>
                      {cap.name}
                    </span>
                    <span style={{
                      fontSize: "0.68rem",
                      fontWeight: 700,
                      padding: "2px 6px",
                      borderRadius: "4px",
                      background: "rgba(255, 255, 255, 0.05)",
                      color: "var(--text-muted)",
                    }}>
                      LOCKED 🔒
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Column Semantic Mapping Table */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                <div>
                  <h4 style={{ fontSize: "0.92rem", fontWeight: 700, color: "#FFFFFF" }}>
                    Confirm or Override Semantic Column Mappings
                  </h4>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Our AI inferred semantic types based on entropy and patterns. You can adjust mappings if necessary.
                  </p>
                </div>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    setUploadData(null);
                    setFile(null);
                  }}
                  style={{ padding: "4px 10px", fontSize: "0.75rem" }}
                >
                  Choose Different File
                </button>
              </div>

              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column Name</th>
                    <th>Inferred Type</th>
                    <th>Confidence</th>
                    <th>Raw Dtype</th>
                    <th>Unique Values</th>
                    <th>Assign Semantic Role</th>
                  </tr>
                </thead>
                <tbody>
                  {uploadData.detected_columns.map((col: DetectedColumn) => {
                    const currentType = mappingOverrides[col.column_name] || col.detected_semantic_type;
                    return (
                      <tr key={col.column_name}>
                        <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--accent-cyan)" }}>
                          {col.column_name}
                        </td>
                        <td>
                          <span style={{ fontSize: "0.72rem", background: "rgba(255, 255, 255, 0.05)", padding: "2px 6px", borderRadius: "4px" }}>
                            {col.detected_semantic_type}
                          </span>
                        </td>
                        <td>
                          <span style={{
                            fontSize: "0.75rem",
                            fontWeight: 700,
                            color: col.confidence > 0.8 ? "var(--accent-emerald)" : col.confidence > 0.5 ? "var(--accent-amber)" : "var(--text-muted)",
                          }}>
                            {(col.confidence * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                          {col.raw_dtype}
                        </td>
                        <td style={{ fontSize: "0.78rem" }}>
                          {col.unique_values.toLocaleString()}
                        </td>
                        <td>
                          <select
                            value={currentType}
                            onChange={(e) => handleMappingChange(col.column_name, e.target.value)}
                            style={{
                              background: "#161E2E",
                              border: "1px solid var(--border-subtle)",
                              borderRadius: "6px",
                              color: "#FFFFFF",
                              padding: "4px 8px",
                              fontSize: "0.75rem",
                              width: "100%",
                            }}
                          >
                            {SEMANTIC_OPTIONS.map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Actions */}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px", borderTop: "1px solid var(--border-subtle)", paddingTop: "16px" }}>
              <button className="btn-secondary" onClick={onClose} disabled={analyzing}>
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={handleConfirmAndAnalyze}
                disabled={analyzing}
                style={{ minWidth: "220px", display: "flex", justifyContent: "center", alignItems: "center", gap: "8px" }}
              >
                {analyzing ? (
                  <>
                    <Activity className="animate-spin" size={16} />
                    <span>Training ML Pipeline...</span>
                  </>
                ) : (
                  <>
                    <span>Confirm &amp; Run Analytics</span>
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
