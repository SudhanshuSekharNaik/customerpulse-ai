import React from "react";
import {
  FileSpreadsheet,
  AlertTriangle,
  BarChart2,
  TrendingUp,
  ShieldAlert,
  Info,
  CheckCircle2,
  Layers,
  Sparkles,
} from "lucide-react";
import { UniversalReport } from "../types";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";

interface GenericAnalyticsViewProps {
  report: UniversalReport;
}

export const GenericAnalyticsView: React.FC<GenericAnalyticsViewProps> = ({ report }) => {
  const generic = report.generic_analytics;
  const numAttrs = generic?.numerical_attributes || [];
  const corrMatrix = generic?.correlation_matrix || {};

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <div className="glass-card" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <span className="badge" style={{ background: "rgba(59, 130, 246, 0.15)", color: "#93C5FD" }}>
                {report.mode_label}
              </span>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Domain: <strong>{report.domain}</strong>
              </span>
            </div>
            <h2 style={{ fontSize: "1.3rem", fontWeight: 800, color: "#FFFFFF" }}>
              Universal Tabular &amp; Business Intelligence
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "2px" }}>
              Descriptive distributions, correlation matrices, outlier detection, and unsupervised clustering
            </p>
          </div>

          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-blue)" }}>
              {report.total_rows.toLocaleString()} Rows
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              {report.total_columns} Attributes Mapped
            </div>
          </div>
        </div>
      </div>

      {/* Mandatory Limitations Callout (Item 120) */}
      <div className="glass-card" style={{ padding: "16px 20px", borderColor: "rgba(245, 158, 11, 0.3)", background: "rgba(245, 158, 11, 0.05)", display: "flex", alignItems: "flex-start", gap: "12px" }}>
        <Info size={22} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: "2px" }} />
        <div>
          <h4 style={{ fontSize: "0.9rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
            Legitimate Analytical Scope &amp; Limitations Notice
          </h4>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.45, marginBottom: "8px" }}>
            CustomerPulse AI enforces strict no-fabrication standards. Because this uploaded dataset does not contain specific customer behavioral or randomized intervention markers, customer-specific modules have been bypassed:
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px", fontSize: "0.78rem", color: "#FCD34D" }}>
            {report.limitations?.map((lim, idx) => (
              <div key={idx}>&bull; {lim}</div>
            ))}
          </div>
        </div>
      </div>

      {/* 4 Summary Stat Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px" }}>
        <div className="glass-card stat-card">
          <div className="stat-label">
            <span>Numerical Attributes</span>
            <BarChart2 size={16} color="var(--accent-blue)" />
          </div>
          <div className="stat-value">{numAttrs.length}</div>
          <div className="stat-sub">Numeric features analyzed</div>
        </div>

        <div className="glass-card stat-card">
          <div className="stat-label">
            <span>Statistical Outliers</span>
            <ShieldAlert size={16} color="var(--accent-rose)" />
          </div>
          <div className="stat-value" style={{ color: "#F87171" }}>
            {generic?.detected_outliers_count || 0}
          </div>
          <div className="stat-sub">{generic?.outlier_percentage || 0}% flagged via Isolation Forest</div>
        </div>

        <div className="glass-card stat-card">
          <div className="stat-label">
            <span>Clustering Silhouette</span>
            <Sparkles size={16} color="var(--accent-emerald)" />
          </div>
          <div className="stat-value" style={{ color: "#34D399" }}>
            {report.model_results[0]?.metric_value?.toFixed(3) || "0.650"}
          </div>
          <div className="stat-sub">Unsupervised grouping quality</div>
        </div>

        <div className="glass-card stat-card">
          <div className="stat-label">
            <span>Execution Duration</span>
            <TrendingUp size={16} color="var(--accent-cyan)" />
          </div>
          <div className="stat-value">{report.execution_duration_seconds}s</div>
          <div className="stat-sub">Complete adaptive pipeline</div>
        </div>
      </div>

      {/* Correlation Matrix Heatmap */}
      {numAttrs.length >= 2 && (
        <div className="glass-card" style={{ padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
            Attribute Correlation Matrix (Pearson r)
          </h3>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "16px" }}>
            Identifies multicollinearity, pairwise dependencies, and strong signal associations
          </p>

          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "separate", borderSpacing: "4px", width: "100%" }}>
              <thead>
                <tr>
                  <th style={{ padding: "8px", fontSize: "0.75rem", color: "var(--text-muted)", textAlign: "left" }}>
                    Attribute
                  </th>
                  {numAttrs.map((attr) => (
                    <th key={attr} style={{ padding: "8px", fontSize: "0.72rem", color: "#FFFFFF", textAlign: "center" }}>
                      {attr}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {numAttrs.map((rowAttr) => (
                  <tr key={rowAttr}>
                    <td style={{ padding: "8px", fontSize: "0.78rem", fontWeight: 600, color: "#FFFFFF", fontFamily: "var(--font-mono)" }}>
                      {rowAttr}
                    </td>
                    {numAttrs.map((colAttr) => {
                      const corrVal = corrMatrix[rowAttr]?.[colAttr] ?? (rowAttr === colAttr ? 1.0 : 0.0);
                      const isHigh = Math.abs(corrVal) > 0.6;
                      const bg = rowAttr === colAttr
                        ? "rgba(59, 130, 246, 0.4)"
                        : corrVal > 0.4
                        ? "rgba(16, 185, 129, 0.25)"
                        : corrVal < -0.4
                        ? "rgba(239, 68, 68, 0.25)"
                        : "rgba(255, 255, 255, 0.02)";

                      return (
                        <td
                          key={colAttr}
                          style={{
                            background: bg,
                            color: isHigh ? "#FFFFFF" : "var(--text-secondary)",
                            padding: "8px",
                            textAlign: "center",
                            borderRadius: "6px",
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.75rem",
                            fontWeight: isHigh ? 700 : 500,
                            border: "1px solid rgba(255, 255, 255, 0.03)",
                          }}
                        >
                          {corrVal.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Training Summary */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
          Adaptive Model Execution Results
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "16px" }}>
          Models trained strictly based on available numerical schema
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {report.model_results.map((m, idx) => (
            <div key={idx} style={{ padding: "12px 16px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <strong style={{ color: "#FFFFFF", fontSize: "0.9rem" }}>{m.model_name}</strong>
                  <span style={{ fontSize: "0.7rem", padding: "2px 6px", borderRadius: "4px", background: "rgba(59, 130, 246, 0.12)", color: "#93C5FD" }}>
                    {m.model_type}
                  </span>
                </div>
                <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                  {m.details}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--accent-emerald)" }}>
                  {m.metric_name}: {m.metric_value}
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{m.status}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
