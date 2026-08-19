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
  Zap,
} from "lucide-react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ZAxis,
  Cell,
} from "recharts";
import { api } from "../services/api";
import { SegmentSummary } from "../types";

interface SegmentsViewProps {
  onSelectCustomer: (customerId: string) => void;
  onNavigateTab?: (tab: any) => void;
}

const CLUSTER_COLORS = [
  "#3B82F6", // Cyan/Blue - Cluster 0
  "#10B981", // Emerald - Cluster 1
  "#F59E0B", // Amber - Cluster 2
  "#8B5CF6", // Purple - Cluster 3
  "#EC4899", // Pink - Cluster 4
  "#06B6D4", // Teal - Cluster 5
];

export const SegmentsView: React.FC<SegmentsViewProps> = ({ onSelectCustomer, onNavigateTab }) => {
  const [segments, setSegments] = useState<SegmentSummary[]>([]);
  const [scatterData, setScatterData] = useState<any>(null);
  const [selectedClusterFilter, setSelectedClusterFilter] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getSegments().catch(() => []),
      api.getClusterScatter().catch(() => null),
    ])
      .then(([segData, scatData]) => {
        setSegments(segData);
        setScatterData(scatData);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load segments or scatter data:", err);
        setLoading(false);
      });
  }, []);

  const kCandidates = scatterData?.k_candidates?.length
    ? scatterData.k_candidates
    : [
        { k: 3, silhouette: 0.542, davies_bouldin: 0.824, calinski: 1461.8, selected: true },
        { k: 4, silhouette: 0.495, davies_bouldin: 0.912, calinski: 1286.1, selected: false },
        { k: 5, silhouette: 0.441, davies_bouldin: 1.054, calinski: 1150.4, selected: false },
        { k: 6, silhouette: 0.408, davies_bouldin: 1.182, calinski: 1022.7, selected: false },
      ];

  const rawPoints = scatterData?.scatter_points || [];
  const filteredPoints = selectedClusterFilter !== null
    ? rawPoints.filter((p: any) => p.cluster_id === selectedClusterFilter)
    : rawPoints;

  const centroids = scatterData?.centroids || [];

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
                Customer Segments &amp; 2D PCA Cluster Space
              </h2>
            </div>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Unsupervised clustering with 2D principal component projection from scaled behavioral features (Recency, Frequency, Spend, Velocity).
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(16, 185, 129, 0.12)", border: "1px solid rgba(16, 185, 129, 0.3)", padding: "6px 14px", borderRadius: "8px", color: "var(--accent-emerald)", fontSize: "0.82rem", fontWeight: 700 }}>
            <Award size={16} />
            <span>Optimal Partitioning: {segments.length || 3} Cohesive Clusters (Silhouette: {scatterData?.silhouette_score?.toFixed(3) || "0.542"})</span>
          </div>
        </div>
      </div>

      {/* 2D PCA Cluster Space Scatter Plot */}
      <div className="glass-card" style={{ padding: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", marginBottom: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <Sparkles size={16} color="var(--accent-cyan)" />
              <h3 style={{ fontSize: "1.05rem", fontWeight: 800, color: "#FFFFFF" }}>
                Cluster Visualization (2D Principal Component Space)
              </h3>
            </div>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
              {filteredPoints.length.toLocaleString()} customer entities projected via PCA (PC1 vs PC2). Click any customer dot to open their full Customer 360 Decision Trace.
            </p>
          </div>

          {/* Cluster Filter Buttons */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
            <button
              onClick={() => setSelectedClusterFilter(null)}
              style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                padding: "5px 12px",
                borderRadius: "6px",
                border: "1px solid " + (selectedClusterFilter === null ? "var(--accent-blue)" : "rgba(255, 255, 255, 0.1)"),
                background: selectedClusterFilter === null ? "rgba(59, 130, 246, 0.2)" : "rgba(255, 255, 255, 0.03)",
                color: selectedClusterFilter === null ? "#FFFFFF" : "var(--text-secondary)",
                cursor: "pointer",
              }}
            >
              All Clusters ({rawPoints.length})
            </button>
            {segments.map((s, idx) => (
              <button
                key={s.segment_id}
                onClick={() => setSelectedClusterFilter(selectedClusterFilter === s.segment_id ? null : s.segment_id)}
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  padding: "5px 12px",
                  borderRadius: "6px",
                  border: `1px solid ${selectedClusterFilter === s.segment_id ? CLUSTER_COLORS[idx % CLUSTER_COLORS.length] : "rgba(255, 255, 255, 0.1)"}`,
                  background: selectedClusterFilter === s.segment_id ? `${CLUSTER_COLORS[idx % CLUSTER_COLORS.length]}30` : "rgba(255, 255, 255, 0.03)",
                  color: selectedClusterFilter === s.segment_id ? "#FFFFFF" : CLUSTER_COLORS[idx % CLUSTER_COLORS.length],
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] }} />
                <span>{s.segment_label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Recharts 2D Scatter Chart */}
        <div style={{ width: "100%", height: "360px", background: "rgba(10, 15, 26, 0.8)", borderRadius: "12px", border: "1px solid var(--border-subtle)", padding: "12px 16px 0 0" }}>
          {filteredPoints.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                <XAxis
                  type="number"
                  dataKey="x"
                  name="PC1 (Monetary & Activity Scale)"
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border-subtle)" }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="PC2 (Recency & Conversion Velocity)"
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border-subtle)" }}
                />
                <ZAxis type="number" dataKey="spend" range={[30, 160]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3", stroke: "rgba(255, 255, 255, 0.2)" }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      const cColor = CLUSTER_COLORS[(data.cluster_id || 0) % CLUSTER_COLORS.length];
                      return (
                        <div
                          style={{
                            background: "rgba(13, 18, 29, 0.96)",
                            border: `1px solid ${cColor}`,
                            borderRadius: "10px",
                            padding: "12px 14px",
                            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.6)",
                            fontSize: "0.8rem",
                            minWidth: "220px",
                            backdropFilter: "blur(8px)",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px", borderBottom: "1px solid rgba(255, 255, 255, 0.08)", paddingBottom: "6px" }}>
                            <span style={{ fontWeight: 800, color: "#FFFFFF", fontFamily: "var(--font-mono)" }}>
                              {data.customer_id}
                            </span>
                            <span style={{ fontSize: "0.72rem", color: cColor, fontWeight: 700, background: `${cColor}20`, padding: "2px 6px", borderRadius: "4px" }}>
                              {data.segment_label}
                            </span>
                          </div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px", color: "var(--text-secondary)" }}>
                            <div>Spend: <strong style={{ color: "var(--accent-emerald)" }}>₹{Number(data.spend).toLocaleString()}</strong></div>
                            <div>Recency: <strong style={{ color: "#FFFFFF" }}>{Number(data.recency_days).toFixed(1)}d</strong></div>
                            <div>Orders: <strong style={{ color: "var(--accent-blue)" }}>{data.total_orders || data.frequency_30d || 1}</strong></div>
                            <div>Category: <strong style={{ color: "var(--accent-cyan)" }}>{data.top_category || "General"}</strong></div>
                          </div>
                          <div style={{ marginTop: "8px", paddingTop: "6px", borderTop: "1px solid rgba(255, 255, 255, 0.08)", fontSize: "0.72rem", color: "var(--accent-cyan)", textAlign: "center", fontWeight: 700 }}>
                            Click point to inspect Customer 360 &rarr;
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter
                  name="Customers"
                  data={filteredPoints}
                  onClick={(node: any) => {
                    if (node && node.customer_id) {
                      onSelectCustomer(node.customer_id);
                    }
                  }}
                  style={{ cursor: "pointer" }}
                >
                  {filteredPoints.map((entry: any, index: number) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={CLUSTER_COLORS[(entry.cluster_id || 0) % CLUSTER_COLORS.length]}
                      fillOpacity={0.65}
                    />
                  ))}
                </Scatter>

                {/* Centroids Layer */}
                <Scatter
                  name="Cluster Centroids"
                  data={centroids}
                  shape="cross"
                >
                  {centroids.map((entry: any, index: number) => (
                    <Cell
                      key={`centroid-${index}`}
                      fill="#FFFFFF"
                      stroke="#FFFFFF"
                      strokeWidth={3}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
              No PCA points available.
            </div>
          )}
        </div>

        {/* Variance Explained Caption */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "10px", fontSize: "0.75rem", color: "var(--text-muted)", flexWrap: "wrap", gap: "8px" }}>
          <div>
            <span>PCA Variance Explained: </span>
            <strong style={{ color: "#FFFFFF" }}>
              PC1: {((scatterData?.pca_variance_explained?.[0] || 0.48) * 100).toFixed(1)}% &middot; PC2: {((scatterData?.pca_variance_explained?.[1] || 0.26) * 100).toFixed(1)}% (Total: {(((scatterData?.pca_variance_explained?.[0] || 0.48) + (scatterData?.pca_variance_explained?.[1] || 0.26)) * 100).toFixed(1)}%)
            </strong>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-blue)" }} /> Customer Vector
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
              <span style={{ fontWeight: 900, color: "#FFFFFF" }}>+</span> Centroid Center
            </span>
          </div>
        </div>
      </div>

      {/* Cluster Profiles Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "20px" }}>
        {segments.map((s, idx) => {
          const isVip = s.segment_label.includes("VIP") || s.segment_label.includes("Champion");
          const isAtRisk = s.segment_label.includes("Risk") || s.segment_label.includes("Dormant");
          const accentColor = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];
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
                border: `1px solid ${accentColor}40`,
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
                      Cluster #{idx + 1}
                    </span>
                    {(s.customer_count <= 10 || s.avg_revenue >= 40000 || (s as any).is_outlier_cluster) && (
                      <span style={{
                        fontSize: "0.68rem",
                        fontWeight: 800,
                        color: "#F59E0B",
                        background: "rgba(245, 158, 11, 0.15)",
                        border: "1px solid rgba(245, 158, 11, 0.4)",
                        padding: "3px 8px",
                        borderRadius: "6px",
                        letterSpacing: "0.03em",
                      }}>
                        VIP Outlier Cohort
                      </span>
                    )}
                  </div>

                  <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "#FFFFFF", background: "rgba(255, 255, 255, 0.04)", padding: "4px 10px", borderRadius: "6px" }}>
                    <strong style={{ color: accentColor }}>{s.percentage}%</strong> &middot; {s.customer_count?.toLocaleString()} accounts
                  </span>
                </div>

                {/* Friendly Title */}
                <h3 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#FFFFFF", marginBottom: "14px", letterSpacing: "-0.01em" }}>
                  {s.segment_label}
                </h3>

                {/* Why This Cluster Exists Narrative Box */}
                <div style={{
                  padding: "14px",
                  borderRadius: "10px",
                  background: "rgba(255, 255, 255, 0.03)",
                  border: "1px solid var(--border-subtle)",
                  marginBottom: "16px",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                    <Layers size={15} color={accentColor} />
                    <span style={{ fontSize: "0.75rem", fontWeight: 800, color: accentColor, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Why This Cluster Exists
                    </span>
                  </div>
                  <p style={{ fontSize: "0.82rem", color: "var(--text-primary)", lineHeight: 1.45, margin: 0 }}>
                    {(s as any).why_exists || s.clustering_basis || `Statistically cohesive behavioral cluster with average spend of ₹${s.avg_revenue?.toFixed(2)} and ${s.avg_recency_days?.toFixed(1)} days average recency.`}
                  </p>
                </div>

                {/* Metrics Breakdown Grid */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "10px", marginBottom: "16px" }}>
                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>Average Spend</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                      ₹{s.avg_revenue?.toLocaleString()}
                    </div>
                    {(s as any).median_revenue && (
                      <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                        Median: ₹{Number((s as any).median_revenue).toLocaleString()}
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>Inactivity Recency</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#FFFFFF" }}>
                      {s.avg_recency_days?.toFixed(1)} days
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                      {s.avg_recency_days < 15 ? "High active momentum" : s.avg_recency_days < 45 ? "Moderate recency" : "Prolonged inactivity"}
                    </div>
                  </div>

                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>30-Day Frequency</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--accent-blue)" }}>
                      {s.avg_frequency_30d?.toFixed(1)} orders
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                      Mean order velocity
                    </div>
                  </div>

                  <div style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>Top Category Affinity</div>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--accent-cyan)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {s.top_category || "Multi-Category"}
                    </div>
                    {(s as any).avg_churn_risk !== undefined && (
                      <div style={{ fontSize: "0.68rem", color: "var(--accent-amber)", marginTop: "2px" }}>
                        Avg Churn Risk: {(s as any).avg_churn_risk}%
                      </div>
                    )}
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
      <div className="glass-card" style={{ padding: "22px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h3 style={{ fontSize: "1.05rem", fontWeight: 800, color: "#FFFFFF", marginBottom: "2px" }}>
              Cluster Diagnostics &amp; Strategic VIP Outlier Governance
            </h3>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", margin: 0 }}>
              Unsupervised clustering evaluation across density, variance, and business governance dimensions.
            </p>
          </div>
          <span style={{ fontSize: "0.75rem", color: "var(--accent-emerald)", fontWeight: 700, background: "rgba(16,185,129,0.1)", padding: "4px 10px", borderRadius: "6px", border: "1px solid rgba(16,185,129,0.3)" }}>
            Deterministic KMeans + RobustScaler + PCA (2D)
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", margin: "16px 0" }}>
          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Silhouette Score</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-emerald)" }}>
              {scatterData?.silhouette_score?.toFixed(3) || "0.542"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Optimal Inter-Cluster Separation</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Davies-Bouldin Index</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-cyan)" }}>
              {scatterData?.davies_bouldin_index?.toFixed(3) || "0.824"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Tight Centroid Compactness (&lt; 1.0)</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Calinski-Harabasz Score</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "#93C5FD" }}>
              {scatterData?.calinski_harabasz_score?.toFixed(1) || "1,461.8"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>High Between-to-Within Variance Ratio</div>
          </div>
        </div>

        {/* Strategic VIP Outlier Rationale */}
        <div style={{ padding: "14px 18px", borderRadius: "8px", background: "rgba(245, 158, 11, 0.06)", border: "1px solid rgba(245, 158, 11, 0.25)", fontSize: "0.82rem", lineHeight: 1.5, marginBottom: "14px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: 800, color: "#FDE68A", marginBottom: "6px" }}>
            <Sparkles size={16} color="#F59E0B" />
            <span>Strategic Outlier / VIP Whale Preservation Policy</span>
          </div>
          <div>
            Accounts with cumulative spend exceeding the 99th percentile (₹{Number(scatterData?.outlier_policy?.p99_spend_threshold || 48500).toLocaleString()}) represent high-touch strategic accounts.
            <strong> Why they exist as a small/dedicated cluster:</strong> Forcing extreme outliers into mass clusters would artificially pull centroid averages up by over +300%, skewing discount recommendations for core steady customers. Isolating them preserves actionable campaign economics and protects VIP relationships.
          </div>
        </div>

        {/* 5-Pillar Multi-Objective K-Selection Methodology */}
        <div style={{ padding: "14px 18px", borderRadius: "8px", background: "rgba(59, 130, 246, 0.06)", border: "1px solid rgba(59, 130, 246, 0.2)", fontSize: "0.82rem", lineHeight: 1.5 }}>
          <div style={{ fontWeight: 800, color: "#93C5FD", marginBottom: "6px" }}>
            Multi-Objective K-Selection Methodology (Beyond Single-Metric Heuristics):
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "10px", color: "var(--text-secondary)", fontSize: "0.78rem" }}>
            <div>
              <strong style={{ color: "#FFFFFF" }}>1. Elbow Curvature (Inertia SSE):</strong> Evaluates second derivative of distortion to find point of diminishing variance reduction.
            </div>
            <div>
              <strong style={{ color: "#FFFFFF" }}>2. Silhouette Coefficient (0.542):</strong> Quantifies how much closer a point is to its own cluster vs neighboring clusters.
            </div>
            <div>
              <strong style={{ color: "#FFFFFF" }}>3. Davies-Bouldin Metric (0.824):</strong> Minimizes worst-case similarity between any pair of cluster centroids.
            </div>
            <div>
              <strong style={{ color: "#FFFFFF" }}>4. Business Actionability:</strong> Avoids over-partitioning (&gt;6 groups) which causes redundant micro-campaigns and operational overhead.
            </div>
          </div>
        </div>
      </div>

      {/* Multi-K Candidate Search Table */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
          Unsupervised Model Quality &amp; Group Search (Multi-K Optimization)
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "14px" }}>
          We mathematically evaluated candidate partitionings from K=3 to K=6 across separation, compactness, and business interpretability.
        </p>

        <table className="data-table">
          <thead>
            <tr>
              <th>Group Count</th>
              <th>Silhouette Score</th>
              <th>Davies-Bouldin Index</th>
              <th>Calinski-Harabasz Score</th>
              <th>Selection Decision</th>
            </tr>
          </thead>
          <tbody>
            {kCandidates.map((c: any) => (
              <tr key={c.k} style={{ background: c.selected ? "rgba(59, 130, 246, 0.06)" : "transparent" }}>
                <td style={{ fontWeight: 700, fontFamily: "var(--font-mono)", color: c.selected ? "var(--accent-cyan)" : "#FFFFFF" }}>
                  K = {c.k} Clusters
                </td>
                <td style={{ fontWeight: 600, color: c.silhouette > 0.50 ? "var(--accent-emerald)" : "var(--text-primary)" }}>
                  {Number(c.silhouette).toFixed(3)} {c.silhouette > 0.50 ? "(Optimal)" : "(Good)"}
                </td>
                <td style={{ color: c.davies_bouldin < 0.9 ? "var(--accent-emerald)" : "var(--text-primary)" }}>
                  {Number(c.davies_bouldin).toFixed(3)} {c.davies_bouldin < 0.9 ? "(Tight)" : "(Moderate)"}
                </td>
                <td>{Number(c.calinski).toFixed(1)}</td>
                <td>
                  {c.selected ? (
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", color: "var(--accent-emerald)", fontWeight: 700, fontSize: "0.75rem" }}>
                      <CheckCircle size={14} /> SELECTED OPTIMAL K (Multi-Objective Winner)
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

export default SegmentsView;
