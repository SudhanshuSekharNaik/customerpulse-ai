import React, { useEffect, useState } from "react";
import {
  Users,
  DollarSign,
  AlertTriangle,
  Sparkles,
  TrendingUp,
  ArrowRight,
  ShieldAlert,
  Zap,
  Activity,
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  Info,
  Database,
  Clock,
  ShoppingCart,
  Flame,
  Tag,
  ShoppingBag,
} from "lucide-react";
import { api } from "../services/api";
import { StateSummary, BehaviorChange, Recommendation, ActiveDatasetContext, TrafficForecastReport } from "../types";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
  AreaChart,
  Area,
} from "recharts";

interface DashboardViewProps {
  onSelectCustomer: (customerId: string) => void;
  onNavigateTab: (tab: any) => void;
  onOpenUploadModal: () => void;
  onSwitchMode: (mode: string, datasetId?: string, datasetLabel?: string) => void;
  activeContext: ActiveDatasetContext | null;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onSelectCustomer,
  onNavigateTab,
  onOpenUploadModal,
  onSwitchMode,
  activeContext,
}) => {
  const [loading, setLoading] = useState(true);
  const [kpis, setKpis] = useState<any>(null);
  const [states, setStates] = useState<StateSummary[]>([]);
  const [changes, setChanges] = useState<BehaviorChange[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [traffic, setTraffic] = useState<TrafficForecastReport | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getAnalyticsOverview(),
      api.getStates(),
      api.getBehaviorChanges(6),
      api.getRecommendations({ limit: 6 }),
      api.getTrafficForecast().catch(() => null),
    ])
      .then(([kpiData, stateData, changeData, recData, trafficData]) => {
        setKpis(kpiData);
        setStates(stateData);
        setChanges(changeData);
        setRecommendations(recData);
        setTraffic(trafficData);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Dashboard data load failed:", err);
        setLoading(false);
      });
  }, [activeContext]);

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh", color: "var(--text-muted)" }}>
        <Activity className="animate-spin" size={32} />
        <span style={{ marginLeft: "12px", fontSize: "1rem" }}>Loading Executive Decision Pulse...</span>
      </div>
    );
  }

  const STATE_COLORS: Record<string, string> = {
    NEW: "#38BDF8",
    EXPLORING: "#818CF8",
    ENGAGED: "#6366F1",
    CONVERTING: "#34D399",
    LOYAL: "#10B981",
    DECLINING: "#FBBF24",
    AT_RISK: "#F97316",
    DORMANT: "#EF4444",
    RECOVERING: "#A855F7",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* UNIVERSAL DATA MODE HERO BANNER */}
      <div className="glass-card glass-card-glow" style={{
        padding: "24px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "16px",
        background: "linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(18, 24, 38, 0.95) 100%)",
        border: "1px solid rgba(59, 130, 246, 0.3)",
      }}>
        <div style={{ maxWidth: "620px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
            <span style={{ fontSize: "0.72rem", fontWeight: 800, letterSpacing: "0.08em", background: "rgba(59, 130, 246, 0.2)", color: "#93C5FD", padding: "3px 8px", borderRadius: "4px" }}>
              E-COMMERCE &amp; UNIVERSAL CSV INTELLIGENCE
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Active: <strong style={{ color: "var(--accent-cyan)" }}>{activeContext?.dataset_name || "RetailRocket Benchmark"}</strong>
            </span>
          </div>
          <h2 style={{ fontSize: "1.45rem", fontWeight: 800, color: "#FFFFFF", letterSpacing: "-0.01em" }}>
            Upload Any E-Commerce CSV &mdash; Instant Traffic &amp; Decision Intelligence
          </h2>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px", lineHeight: 1.45 }}>
            Optimized for Amazon, Flipkart, Myntra, and custom e-commerce datasets. Automatically updates customer 360 profiles, predicts shopping peak surge windows, models cart abandonment, and outputs action cards.
          </p>
        </div>

        <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
          <button
            className="btn-primary"
            onClick={onOpenUploadModal}
            style={{ padding: "10px 18px", fontSize: "0.88rem" }}
          >
            <UploadCloud size={17} />
            <span>Upload Your CSV</span>
          </button>

          <button
            className="btn-secondary"
            onClick={() => window.open(api.getPdfReportUrl(), "_blank")}
            style={{ padding: "10px 16px", fontSize: "0.85rem", borderColor: "rgba(16, 185, 129, 0.4)", color: "var(--accent-emerald)" }}
            title="Download audit-ready executive PDF decision report"
          >
            <FileSpreadsheet size={15} />
            <span>Export Executive PDF</span>
          </button>

          {activeContext?.active_mode === "UPLOAD" ? (
            <button
              className="btn-secondary"
              onClick={() => onSwitchMode("DEMO")}
              style={{ padding: "10px 16px", fontSize: "0.85rem" }}
            >
              <Database size={15} />
              <span>Use Demo Dataset</span>
            </button>
          ) : (
            <button
              className="btn-secondary"
              onClick={() => onNavigateTab("agent")}
              style={{ padding: "10px 16px", fontSize: "0.85rem" }}
            >
              <Zap size={15} />
              <span>Ask AI Analyst</span>
            </button>
          )}
        </div>
      </div>

      {/* KPI Cards Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "16px" }}>
        <div className="glass-card stat-card">
          <div className="stat-label">
            <span>Total Customers</span>
            <Users size={16} color="var(--accent-blue)" />
          </div>
          <div className="stat-value">{kpis?.total_customers?.toLocaleString() || 0}</div>
          <div className="stat-sub">{kpis?.total_events?.toLocaleString() || 0} Ingested Events</div>
        </div>

        <div className="glass-card stat-card">
          <div className="stat-label">
            <span>Total Revenue</span>
            <DollarSign size={16} color="var(--accent-emerald)" />
          </div>
          <div className="stat-value">₹{kpis?.total_revenue?.toLocaleString() || "0"}</div>
          <div className="stat-sub">Aggregated cohort spend</div>
        </div>

        <div className="glass-card stat-card" style={{ borderColor: "rgba(239, 68, 68, 0.3)" }}>
          <div className="stat-label">
            <span>Revenue at Risk</span>
            <AlertTriangle size={16} color="var(--accent-rose)" />
          </div>
          <div className="stat-value" style={{ color: "#F87171" }}>
            ₹{kpis?.revenue_at_risk?.toLocaleString() || "0"}
          </div>
          <div className="stat-sub">{kpis?.at_risk_customers_count || 0} accounts in declining states</div>
        </div>

        <div className="glass-card stat-card" style={{ borderColor: "rgba(249, 115, 22, 0.3)" }}>
          <div className="stat-label">
            <span>At-Risk Accounts</span>
            <ShieldAlert size={16} color="var(--accent-amber)" />
          </div>
          <div className="stat-value" style={{ color: "#FB923C" }}>
            {kpis?.at_risk_percentage || 0}%
          </div>
          <div className="stat-sub">{kpis?.at_risk_customers_count || 0} accounts in declining states</div>
        </div>

        <div className="glass-card stat-card" style={{ borderColor: "rgba(16, 185, 129, 0.3)" }}>
          <div className="stat-label">
            <span>Addressable Uplift</span>
            <Sparkles size={16} color="var(--accent-emerald)" />
          </div>
          <div className="stat-value" style={{ color: "#34D399" }}>
            ₹{kpis?.addressable_portfolio_uplift?.toLocaleString() || "0"}
          </div>
          <div className="stat-sub">From Next-Best-Actions</div>
        </div>
      </div>

      {/* E-COMMERCE & TRAFFIC SURGE FORECAST SECTION */}
      {traffic && (
        <div className="glass-card" style={{ padding: "22px", borderColor: "rgba(6, 182, 212, 0.4)", background: "rgba(10, 15, 28, 0.85)" }}>
          {/* Header with Next Peak Surge Alert */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px", marginBottom: "20px" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                <span className="badge" style={{ background: "rgba(6, 182, 212, 0.15)", color: "var(--accent-cyan)", border: "1px solid rgba(6, 182, 212, 0.35)" }}>
                  E-COMMERCE TRAFFIC RADAR
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  {activeContext?.mode_label || activeContext?.dataset_name || "E-Commerce"} Behavior Signature
                </span>
              </div>
              <h3 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#FFFFFF" }}>
                Next Peak Traffic &amp; Shopping Surge Forecast
              </h3>
            </div>

            {/* Next Peak Surge Alert Box */}
            <div style={{
              padding: "10px 16px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(239, 68, 68, 0.12) 100%)",
              border: "1px solid rgba(245, 158, 11, 0.4)",
              display: "flex",
              alignItems: "center",
              gap: "12px",
            }}>
              <Flame size={24} color="#FBBF24" />
              <div>
                <div style={{ fontSize: "0.72rem", color: "#FCD34D", fontWeight: 700, textTransform: "uppercase" }}>
                  Next Projected Shopping Peak Window
                </div>
                <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "#FFFFFF" }}>
                  {traffic.next_peak_window.window_description}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
                  +{traffic.next_peak_window.projected_traffic_lift_pct}% traffic surge expected &middot; {traffic.next_peak_window.recommended_action}
                </div>
              </div>
            </div>
          </div>

          {/* Traffic Forecast Grid: Hourly Curve + Funnel Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 0.8fr", gap: "20px" }}>
            {/* Hourly Activity Curve */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "#FFFFFF" }}>
                  Hourly Shopping Traffic Curve (24 Hours)
                </span>
                <span style={{ fontSize: "0.72rem", color: "var(--accent-cyan)" }}>
                  {traffic.next_peak_window.window_description}
                </span>
              </div>

              <div style={{ height: "200px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={traffic.hourly_traffic_curve} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                    <defs>
                      <linearGradient id="trafficGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="label" tick={{ fill: "#94A3B8", fontSize: 10 }} interval={2} />
                    <YAxis tick={{ fill: "#94A3B8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: "#121826", border: "1px solid var(--border-subtle)", borderRadius: "8px", fontSize: "0.8rem" }}
                      formatter={(val: any) => [`${val} Activity Index`, "Traffic"]}
                    />
                    <Area type="monotone" dataKey="traffic_index" stroke="#06B6D4" strokeWidth={2} fillOpacity={1} fill="url(#trafficGradient)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* E-Commerce Funnel & Cart Abandonment Summary */}
            {traffic.ecommerce_funnel.has_funnel_events && traffic.ecommerce_funnel.cart_abandonment_rate_pct !== null ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Cart Abandonment Rate</span>
                    <span style={{ fontSize: "0.95rem", fontWeight: 800, color: "#F87171" }}>
                      {traffic.ecommerce_funnel.cart_abandonment_rate_pct}%
                    </span>
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                    ₹{traffic.ecommerce_funnel.recovered_cart_revenue_potential_inr?.toLocaleString() || 0} Recoverable Cart Value
                  </div>
                </div>

                <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Cart-to-View Ratio</span>
                    <span style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--accent-cyan)" }}>
                      {traffic.ecommerce_funnel.cart_to_view_ratio_pct}%
                    </span>
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                    {traffic.ecommerce_funnel.carts?.toLocaleString() || 0} Items Added to Cart
                  </div>
                </div>

                <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.75rem", color: "var(--accent-emerald)" }}>Checkout Conversion</span>
                    <span style={{ fontSize: "0.95rem", fontWeight: 800, color: "#34D399" }}>
                      {traffic.ecommerce_funnel.checkout_conversion_rate_pct}%
                    </span>
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                    {traffic.ecommerce_funnel.purchases?.toLocaleString() || 0} Completed Purchases
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", padding: "16px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)", height: "100%" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--accent-cyan)", marginBottom: "6px" }}>
                  <Info size={16} />
                  <span style={{ fontSize: "0.82rem", fontWeight: 700 }}>E-Commerce Funnel</span>
                </div>
                <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
                  Not enough data — this file only has completed orders
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", lineHeight: 1.4 }}>
                  Cart abandonment and checkout drop-off require page-view and add-to-cart events.
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--accent-emerald)", marginTop: "8px", fontWeight: 600 }}>
                  {traffic.ecommerce_funnel.purchases?.toLocaleString() || 0} Completed Purchases Recorded
                </div>
              </div>
            )}
          </div>

          {/* Category Surge Radar Chips */}
          <div style={{ marginTop: "16px", borderTop: "1px solid var(--border-subtle)", paddingTop: "14px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "#FFFFFF" }}>
                Top Surging Product Categories ({activeContext?.mode_label || activeContext?.dataset_name || "Active Cohort"})
              </span>
              <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                {traffic.festive_sale_multiplier.event_name} (Multiplier: {traffic.festive_sale_multiplier.traffic_multiplier}x)
              </span>
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {traffic.category_surges.map((cat, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "6px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: `1px solid ${cat.surge_velocity === "HIGH_SURGE" ? "rgba(245, 158, 11, 0.4)" : "var(--border-subtle)"}`,
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    fontSize: "0.75rem",
                  }}
                >
                  <Tag size={13} color={cat.surge_velocity === "HIGH_SURGE" ? "#FBBF24" : "#94A3B8"} />
                  <strong style={{ color: "#FFFFFF" }}>{cat.category_name}</strong>
                  <span style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>
                    {cat.demand_share_pct}%
                  </span>
                  {cat.surge_velocity === "HIGH_SURGE" && (
                    <span style={{ fontSize: "0.65rem", padding: "1px 5px", borderRadius: "4px", background: "rgba(245, 158, 11, 0.2)", color: "#FCD34D", fontWeight: 700 }}>
                      SURGE 🔥
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Grid: State Distribution & Markov Quick Flow */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: "20px" }}>
        {/* State Distribution Chart */}
        <div className="glass-card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF" }}>
                9 customer stages, based on real activity from your data
              </h3>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                How your customers are split between new, active, and at-risk stages
              </p>
            </div>
            <button
              className="btn-secondary"
              style={{ fontSize: "0.75rem", padding: "4px 8px" }}
              onClick={() => onNavigateTab("states")}
            >
              How customers move <ArrowRight size={12} />
            </button>
          </div>

          <div style={{ height: "240px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={states} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="state" tick={{ fill: "#94A3B8", fontSize: 11 }} angle={-25} textAnchor="end" />
                <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "#121826", border: "1px solid var(--border-subtle)", borderRadius: "8px", fontSize: "0.8rem" }}
                  formatter={(val: any) => [`${val} Customers`, "Population"]}
                />
                <Bar dataKey="customer_count" radius={[6, 6, 0, 0]}>
                  {states.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={STATE_COLORS[entry.state] || "#3B82F6"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* State Breakdown Donut */}
        <div className="glass-card" style={{ padding: "20px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
              Portfolio Health Composition
            </h3>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "12px" }}>
              Healthy vs. In-Risk customer population balance
            </p>
          </div>

          <div style={{ height: "180px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={states}
                  dataKey="customer_count"
                  nameKey="state"
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={2}
                >
                  {states.map((entry, index) => (
                    <Cell key={`donut-${index}`} fill={STATE_COLORS[entry.state] || "#3B82F6"} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: "#121826", border: "1px solid var(--border-subtle)", borderRadius: "8px", fontSize: "0.8rem" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", justifyContent: "center" }}>
            {states.slice(0, 5).map((s) => (
              <span key={s.state} className={`badge badge-state-${s.state}`}>
                {s.state}: {s.percentage}%
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Row: Anomaly Feed & Next-Best-Action Feed */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
        {/* Behavior Anomaly Radar Alerts */}
        <div className="glass-card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
            <div>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF" }}>
                Recent Behavior Change Alerts
              </h3>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Statistically significant deviations (Z-score &gt; 2.0, EWMA trend breaks)
              </p>
            </div>
            <button
              className="btn-secondary"
              style={{ fontSize: "0.75rem", padding: "4px 8px" }}
              onClick={() => onNavigateTab("behavior")}
            >
              All Alerts <ArrowRight size={12} />
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {changes.map((c) => (
              <div
                key={c.change_id}
                onClick={() => onSelectCustomer(c.customer_id)}
                style={{
                  padding: "10px 14px",
                  borderRadius: "8px",
                  background: "rgba(255, 255, 255, 0.02)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  cursor: "pointer",
                }}
              >
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, fontSize: "0.85rem", color: "var(--accent-blue)" }}>
                      {c.customer_id}
                    </span>
                    <span className={`badge badge-severity-${c.severity}`}>{c.severity}</span>
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                    {c.metric.replace("_", " ").toUpperCase()}: {c.baseline_value.toFixed(1)} → {c.current_value.toFixed(1)} ({c.pct_change > 0 ? "+" : ""}{c.pct_change.toFixed(1)}%)
                  </div>
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  {c.detection_method}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Priority Next-Best-Actions */}
        <div className="glass-card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
            <div>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF" }}>
                Priority Next-Best-Actions
              </h3>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Auditable decision cards ranked by state, uplift, and ROI
              </p>
            </div>
            <button
              className="btn-secondary"
              style={{ fontSize: "0.75rem", padding: "4px 8px" }}
              onClick={() => onNavigateTab("recommendations")}
            >
              Action Hub <ArrowRight size={12} />
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {recommendations.map((r) => (
              <div
                key={r.recommendation_id}
                onClick={() => onSelectCustomer(r.customer_id)}
                style={{
                  padding: "10px 14px",
                  borderRadius: "8px",
                  background: "rgba(255, 255, 255, 0.02)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  cursor: "pointer",
                }}
              >
                <div style={{ flex: 1, paddingRight: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, fontSize: "0.85rem", color: "var(--accent-cyan)" }}>
                      {r.customer_id}
                    </span>
                    <span style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--accent-blue)", background: "rgba(59, 130, 246, 0.1)", padding: "2px 6px", borderRadius: "4px" }}>
                      {r.action_type}
                    </span>
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text-primary)", marginTop: "2px", fontWeight: 500 }}>
                    {r.what_text}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                    +₹{r.expected_impact?.toFixed(0)}
                  </div>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                    {r.confidence_level} Conf.
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardView;
