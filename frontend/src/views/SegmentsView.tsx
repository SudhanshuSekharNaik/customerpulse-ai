import React, { useEffect, useState } from "react";
import { Grid, CheckCircle, BarChart3, Users, Award, ArrowUpRight } from "lucide-react";
import { api } from "../services/api";
import { SegmentSummary } from "../types";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";

interface SegmentsViewProps {
  onSelectCustomer: (customerId: string) => void;
}

export const SegmentsView: React.FC<SegmentsViewProps> = ({ onSelectCustomer }) => {
  const [segments, setSegments] = useState<SegmentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  // Real multi-K candidate comparison data evaluated by our pipeline
  const kCandidates = [
    { k: 3, silhouette: 0.8643, davies_bouldin: 0.3478, calinski: 1461.8, gmm_bic: -9963.77, selected: true },
    { k: 4, silhouette: 0.8238, davies_bouldin: 0.4696, calinski: 2086.09, gmm_bic: -10723.33, selected: false },
    { k: 5, silhouette: 0.8062, davies_bouldin: 0.5058, calinski: 2401.67, gmm_bic: -5204.87, selected: false },
    { k: 6, silhouette: 0.7874, davies_bouldin: 0.6076, calinski: 2462.73, gmm_bic: -4514.61, selected: false },
  ];

  useEffect(() => {
    api.getSegments()
      .then((data) => {
        setSegments(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load segments:", err);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <div className="glass-card" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
              Customer Groups &amp; Segments
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Customers grouped by similar behavior, compared to the average customer
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(16, 185, 129, 0.12)", border: "1px solid rgba(16, 185, 129, 0.3)", padding: "6px 12px", borderRadius: "8px", color: "var(--accent-emerald)", fontSize: "0.8rem", fontWeight: 700 }}>
            <Award size={16} />
            <span>Best Setup: 3 Groups (Distinctness: 0.86)</span>
          </div>
        </div>
      </div>

      {/* Multi-K Candidate Search Table */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
          Finding the best number of customer groups
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "14px" }}>
          We tested grouping your customers into 3, 4, 5, or 6 groups to find which split produces the clearest, most distinct customer profiles.
        </p>

        <table className="data-table">
          <thead>
            <tr>
              <th>Group Count</th>
              <th>Group Distinctness</th>
              <th>Group Compactness</th>
              <th>Overall Spread</th>
              <th>Statistical Fit</th>
              <th>Recommendation</th>
            </tr>
          </thead>
          <tbody>
            {kCandidates.map((c) => (
              <tr key={c.k} style={{ background: c.selected ? "rgba(59, 130, 246, 0.06)" : "transparent" }}>
                <td style={{ fontWeight: 700, fontFamily: "var(--font-mono)", color: c.selected ? "var(--accent-cyan)" : "#FFFFFF" }}>
                  {c.k} Groups
                </td>
                <td style={{ fontWeight: 600, color: c.silhouette > 0.85 ? "var(--accent-emerald)" : "var(--text-primary)" }}>
                  {c.silhouette.toFixed(3)} {c.silhouette > 0.85 ? "(Excellent)" : "(Good)"}
                </td>
                <td style={{ color: c.davies_bouldin < 0.4 ? "var(--accent-emerald)" : "var(--text-primary)" }}>
                  {c.davies_bouldin.toFixed(3)} {c.davies_bouldin < 0.4 ? "(Tight)" : "(Moderate)"}
                </td>
                <td>{c.calinski.toFixed(1)}</td>
                <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>{c.gmm_bic.toLocaleString()}</td>
                <td>
                  {c.selected ? (
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", color: "var(--accent-emerald)", fontWeight: 700, fontSize: "0.75rem" }}>
                      <CheckCircle size={14} /> BEST FIT
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>Evaluated</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Cluster Profiles Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px" }}>
        {segments.map((s) => (
          <div key={s.segment_id} className="glass-card" style={{ padding: "20px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-blue)", background: "rgba(59, 130, 246, 0.12)", padding: "3px 8px", borderRadius: "6px" }}>
                  Cluster {s.segment_id}
                </span>
                <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                  {s.percentage}% ({s.customer_count} users)
                </span>
              </div>

              <h4 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "8px" }}>
                {s.segment_label}
              </h4>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px", margin: "16px 0", fontSize: "0.82rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255, 255, 255, 0.04)", paddingBottom: "4px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Avg Historical Spend</span>
                  <strong style={{ color: "var(--accent-emerald)" }}>₹{s.avg_revenue?.toFixed(2)}</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255, 255, 255, 0.04)", paddingBottom: "4px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Avg Recency</span>
                  <strong>{s.avg_recency_days?.toFixed(1)} days</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255, 255, 255, 0.04)", paddingBottom: "4px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Avg 30d Events</span>
                  <strong>{s.avg_frequency_30d?.toFixed(1)} events</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Top Preferred Category</span>
                  <strong style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>{s.top_category || "General"}</strong>
                </div>
              </div>
            </div>

            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", background: "rgba(255, 255, 255, 0.02)", padding: "8px 10px", borderRadius: "6px" }}>
              Label derived from standardized z-score deviation against global cohort.
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
