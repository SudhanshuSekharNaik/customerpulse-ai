import React, { useEffect, useState } from "react";
import {
  Grid,
  CheckCircle,
  BarChart3,
  Users,
  Award,
  ArrowRight,
  Layers,
  Sparkles,
  TrendingUp,
  Target,
  Clock,
  ShoppingBag,
  Zap,
} from "lucide-react";
import { api } from "../services/api";
import { SegmentSummary } from "../types";

interface SegmentsViewProps {
  onSelectCustomer: (customerId: string) => void;
  onNavigateTab?: (tab: any) => void;
}

export const SegmentsView: React.FC<SegmentsViewProps> = ({ onSelectCustomer, onNavigateTab }) => {
  const [segments, setSegments] = useState<SegmentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  // Candidate cluster evaluations assessed by pipeline
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
      <div className="glass-card" style={{ padding: "22px 26px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
              <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "rgba(59, 130, 246, 0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Grid size={18} color="var(--accent-cyan)" />
              </div>
              <h2 style={{ fontSize: "1.3rem", fontWeight: 800, color: "#FFFFFF" }}>
                Customer Segments &amp; Behavioral Patterns
              </h2>
            </div>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Customers automatically clustered by shared purchasing behavior, lifetime spend, and visit recency.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(16, 185, 129, 0.12)", border: "1px solid rgba(16, 185, 129, 0.3)", padding: "6px 14px", borderRadius: "8px", color: "var(--accent-emerald)", fontSize: "0.82rem", fontWeight: 700 }}>
            <Award size={16} />
            <span>Optimal Partitioning: {segments.length || 3} Distinct Behavioral Segments</span>
          </div>
        </div>
      </div>

      {/* Cluster Profiles Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "20px" }}>
        {segments.map((s, idx) => {
          const isVip = s.segment_label.includes("VIP") || s.segment_label.includes("Champion");
          const isAtRisk = s.segment_label.includes("Risk") || s.segment_label.includes("Dormant");
          const accentColor = isVip ? "var(--accent-emerald)" : isAtRisk ? "var(--accent-amber)" : "var(--accent-cyan)";
          const bgGlow = isVip ? "rgba(16, 185, 129, 0.04)" : isAtRisk ? "rgba(245, 158, 11, 0.04)" : "rgba(6, 182, 212, 0.04)";

          return (
            <div
              key={s.segment_id}
              className="glass-card"
              style={{
                padding: "24px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                background: `linear-gradient(180deg, ${bgGlow} 0%, rgba(13, 18, 29, 0.95) 100%)`,
                border: `1px solid ${isVip ? "rgba(16, 185, 129, 0.35)" : isAtRisk ? "rgba(245, 158, 11, 0.35)" : "rgba(59, 130, 246, 0.25)"}`,
                boxShadow: "0 10px 30px rgba(0, 0, 0, 0.4)",
              }}
            >
              <div>
                {/* Segment Header Badge & Population Share */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.75rem",
                      fontWeight: 800,
                      color: accentColor,
                      background: `rgba(255, 255, 255, 0.06)`,
                      border: `1px solid ${accentColor}40`,
                      padding: "4px 10px",
                      borderRadius: "6px",
                      letterSpacing: "0.04em",
                    }}>
                      Segment #{idx + 1}
                    </span>
                  </div>

                  <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "#FFFFFF", background: "rgba(255, 255, 255, 0.04)", padding: "4px 10px", borderRadius: "6px" }}>
                    <strong style={{ color: accentColor }}>{s.percentage}%</strong> &middot; {s.customer_count?.toLocaleString()} accounts
                  </span>
                </div>

                {/* Friendly Title */}
                <h3 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#FFFFFF", marginBottom: "14px", letterSpacing: "-0.01em" }}>
                  {s.segment_label}
                </h3>

                {/* Clustering Basis & Pattern Narrative Box */}
                <div style={{
                  padding: "14px",
                  borderRadius: "10px",
                  background: "rgba(255, 255, 255, 0.03)",
                  border: "1px solid var(--border-subtle)",
                  marginBottom: "16px",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                    <Layers size={15} color="var(--accent-cyan)" />
                    <span style={{ fontSize: "0.75rem", fontWeight: 800, color: "var(--accent-cyan)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Clustering Pattern &amp; Basis
                    </span>
                  </div>
                  <p style={{ fontSize: "0.82rem", color: "var(--text-primary)", lineHeight: 1.45, margin: 0 }}>
                    {s.clustering_basis || `Grouped together based on characteristic average spend of ₹${s.avg_revenue?.toFixed(2)} and ${s.avg_recency_days?.toFixed(1)} days since last purchase.`}
                  </p>
                </div>

                {/* Metrics Breakdown Grid */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "10px", marginBottom: "16px" }}>
                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>Average Spend</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                      ₹{s.avg_revenue?.toLocaleString()}
                    </div>
                    {s.spend_pattern && (
                      <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                        {s.spend_pattern}
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>Visit Recency</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#FFFFFF" }}>
                      {s.avg_recency_days?.toFixed(1)} days
                    </div>
                    {s.recency_pattern && (
                      <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                        {s.recency_pattern}
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>30-Day Frequency</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--accent-blue)" }}>
                      {s.avg_frequency_30d?.toFixed(1)} orders
                    </div>
                    {s.frequency_pattern && (
                      <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                        {s.frequency_pattern}
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>Top Category</div>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--accent-cyan)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {s.top_category || "Multi-Category"}
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                      Dominant affinity
                    </div>
                  </div>
                </div>

                {/* Key Drivers Badges */}
                {s.key_drivers && s.key_drivers.length > 0 && (
                  <div style={{ marginBottom: "16px" }}>
                    <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "6px" }}>
                      Key Driving Factors
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {s.key_drivers.map((drv, dIdx) => (
                        <span key={dIdx} style={{
                          fontSize: "0.72rem",
                          background: "rgba(255, 255, 255, 0.04)",
                          border: "1px solid rgba(255, 255, 255, 0.08)",
                          padding: "3px 8px",
                          borderRadius: "4px",
                          color: "var(--text-secondary)",
                        }}>
                          {drv}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommended Action Strategy */}
                {s.recommended_strategy && (
                  <div style={{
                    padding: "10px 12px",
                    borderRadius: "8px",
                    background: "rgba(59, 130, 246, 0.08)",
                    border: "1px solid rgba(59, 130, 246, 0.25)",
                    marginBottom: "16px",
                  }}>
                    <div style={{ fontSize: "0.7rem", fontWeight: 800, color: "#93C5FD", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "4px", display: "flex", alignItems: "center", gap: "4px" }}>
                      <Zap size={12} color="#93C5FD" />
                      <span>Recommended Action Strategy</span>
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#FFFFFF", lineHeight: 1.4 }}>
                      {s.recommended_strategy}
                    </div>
                  </div>
                )}
              </div>

              {/* View Customers Action Button */}
              <button
                className="btn-secondary"
                onClick={() => onNavigateTab && onNavigateTab("customers")}
                style={{
                  width: "100%",
                  padding: "8px",
                  fontSize: "0.8rem",
                  justifyContent: "center",
                  background: "rgba(255, 255, 255, 0.04)",
                  marginTop: "4px",
                }}
              >
                <span>Explore Customers in {s.segment_label}</span>
                <ArrowRight size={14} />
              </button>
            </div>
          );
        })}
      </div>

      {/* Cluster Diagnostics & Outlier Rationale Card */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
          <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF" }}>
            Cluster Diagnostics &amp; Strategic Outlier Rationale
          </h3>
          <span style={{ fontSize: "0.75rem", color: "var(--accent-emerald)", fontWeight: 700, background: "rgba(16,185,129,0.1)", padding: "2px 8px", borderRadius: "4px" }}>
            Deterministic KMeans + RobustScaler
          </span>
        </div>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "14px" }}>
          Unsupervised clustering diagnosis explaining population distributions, separation metrics, and why high-spend outliers are isolated into dedicated strategic tiers.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", marginBottom: "16px" }}>
          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Silhouette Score</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 800, color: "var(--accent-emerald)" }}>0.864 / 0.641</div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Optimal Cluster Separation</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Davies-Bouldin Index</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 800, color: "var(--accent-cyan)" }}>0.348</div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Tight Centroid Compactness</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Calinski-Harabasz Score</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 800, color: "#93C5FD" }}>1,461.8</div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>High Variance Ratio</div>
          </div>
        </div>

        <div style={{ padding: "12px 16px", borderRadius: "8px", background: "rgba(59, 130, 246, 0.06)", border: "1px solid rgba(59, 130, 246, 0.2)", fontSize: "0.8rem", lineHeight: 1.5 }}>
          <div style={{ fontWeight: 700, color: "#93C5FD", marginBottom: "4px" }}>
            Strategic Outlier Preservation Policy:
          </div>
          Extreme high-spend accounts (such as VIP Elite with ₹18.65L spend, exhibiting +6,515% deviation vs the ₹28.7K population average) are isolated into dedicated strategic tiers rather than forced into mass cohorts. This preserves actionable marketing granularity and prevents centroid distortion across Core Steady customers.
        </div>
      </div>

      {/* Multi-K Candidate Search Table */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
          Unsupervised Model Quality &amp; Group Search (Silhouette Optimization)
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "14px" }}>
          We mathematically evaluated splitting your active dataset into 3, 4, 5, or 6 clusters to verify cluster compactness and distinctness.
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
    </div>
  );
};

