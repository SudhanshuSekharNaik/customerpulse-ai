import React, { useEffect, useState } from "react";
import { Radar, AlertTriangle, TrendingDown, ArrowUpRight, Zap, Target } from "lucide-react";
import { api } from "../services/api";
import { BehaviorChange } from "../types";

interface BehaviorViewProps {
  onSelectCustomer: (customerId: string) => void;
}

export const BehaviorView: React.FC<BehaviorViewProps> = ({ onSelectCustomer }) => {
  const [changes, setChanges] = useState<BehaviorChange[]>([]);
  const [severityFilter, setSeverityFilter] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getBehaviorChanges(100, severityFilter || undefined)
      .then((data) => {
        setChanges(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load behavior changes:", err);
        setLoading(false);
      });
  }, [severityFilter]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <div className="glass-card" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
              Customer Behavior Changes &amp; Early Signals
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Tracking sudden changes in customer activity, drop-offs, and unusual shopping patterns
            </p>
          </div>

          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {["", "CRITICAL", "HIGH", "MEDIUM", "LOW"].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={severityFilter === sev ? "btn-primary" : "btn-secondary"}
                style={{ fontSize: "0.75rem", padding: "6px 12px" }}
              >
                {sev === "" ? "All Severities" : sev}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Opportunity Score Sub-Component Breakdown Card */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
          Opportunity Score — how worth targeting this customer is
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "16px" }}>
          A continuous 0 to 100 score combining spend, urgency, engagement, and timing so your team knows exactly who to contact first
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "12px" }}>
          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>1. Customer Value (25%)</div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-blue)" }}>Spend Tier</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Lifetime spend &amp; orders</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>2. Urgency (25%)</div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#EF4444" }}>Churn Risk</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Risk of losing the account</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>3. Recent Interest (20%)</div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-cyan)" }}>Cart &amp; Views</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Active shopping velocity</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>4. Responsiveness (15%)</div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-emerald)" }}>Campaign Lift</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Targeting persuadability</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>5. Timing Window (15%)</div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-amber)" }}>Recency</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Best intervention window</div>
          </div>
        </div>
      </div>

      {/* Behavior Change Radar Alert Feed */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
          Active Behavior Anomaly Feed ({changes.length} Events Logged)
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "16px" }}>
          Click any anomaly to inspect customer features and generate targeted mitigation actions
        </p>

        <table className="data-table">
          <thead>
            <tr>
              <th>Customer ID</th>
              <th>Severity</th>
              <th>Metric Tested</th>
              <th>Baseline Value</th>
              <th>Current Value</th>
              <th>% Deviation</th>
              <th>Detection Algorithm</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={8} style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
                  Scanning anomaly alerts...
                </td>
              </tr>
            ) : changes.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
                  No behavior changes matching severity: {severityFilter}
                </td>
              </tr>
            ) : (
              changes.map((c) => (
                <tr key={c.change_id} onClick={() => onSelectCustomer(c.customer_id)} style={{ cursor: "pointer" }}>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--accent-blue)" }}>
                    {c.customer_id}
                  </td>
                  <td>
                    <span className={`badge badge-severity-${c.severity}`}>{c.severity}</span>
                  </td>
                  <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    {c.metric.replace("_", " ").toUpperCase()}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{c.baseline_value.toFixed(2)}</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{c.current_value.toFixed(2)}</td>
                  <td style={{ fontWeight: 700, color: c.pct_change < 0 ? "#EF4444" : "var(--accent-emerald)" }}>
                    {c.pct_change > 0 ? "+" : ""}{c.pct_change.toFixed(1)}%
                  </td>
                  <td style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {c.detection_method}
                  </td>
                  <td>
                    <button className="btn-secondary" style={{ padding: "3px 8px", fontSize: "0.75rem" }}>
                      Investigate <ArrowUpRight size={12} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
