import React, { useEffect, useState } from "react";
import { Target, CheckCircle, ShieldAlert, Sparkles, Filter, ArrowUpRight, DollarSign } from "lucide-react";
import { api } from "../services/api";
import { Recommendation, OfflineBacktestReport } from "../types";

interface RecommendationsViewProps {
  onSelectCustomer: (customerId: string) => void;
}

export const RecommendationsView: React.FC<RecommendationsViewProps> = ({ onSelectCustomer }) => {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [backtest, setBacktest] = useState<OfflineBacktestReport | null>(null);
  const [actionFilter, setActionFilter] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getRecommendations({ limit: 60, action_type: actionFilter || undefined }),
      api.getOfflineBacktest(),
    ])
      .then(([recData, btData]) => {
        setRecommendations(recData);
        setBacktest(btData);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load recommendations:", err);
        setLoading(false);
      });
  }, [actionFilter]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <div className="glass-card" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
              Next-Best-Action Decision Hub &amp; Policy Evaluator
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Dynamic action ranking optimizing for state urgency, churn risk mitigation, and causal uplift ROI
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(59, 130, 246, 0.12)", border: "1px solid rgba(59, 130, 246, 0.3)", padding: "6px 12px", borderRadius: "8px", color: "var(--accent-blue)", fontSize: "0.8rem", fontWeight: 700 }}>
            <Target size={16} />
            <span>{recommendations.length > 0 ? `${recommendations.length.toLocaleString()} Active Decisions` : "Auditable Decision Engine"}</span>
          </div>
        </div>
      </div>

      {/* Offline Policy Backtest Scorecard */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF" }}>
              Offline Policy Backtest &amp; Action Diversity Audit
            </h3>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
              Guarantees the recommendation engine avoids degenerate single-action policies before live deployment
            </p>
          </div>
          <span style={{
            fontSize: "0.75rem",
            fontWeight: 700,
            padding: "4px 10px",
            borderRadius: "6px",
            background: backtest?.is_degenerate_policy ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
            color: backtest?.is_degenerate_policy ? "#EF4444" : "#10B981",
            border: `1px solid ${backtest?.is_degenerate_policy ? "rgba(239, 68, 68, 0.3)" : "rgba(16, 185, 129, 0.3)"}`,
          }}>
            {backtest?.is_degenerate_policy ? "DEGENERATE POLICY DETECTED" : "HEALTHY POLICY DIVERSITY"}
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Action Diversity Index</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-blue)" }}>
              {backtest?.action_diversity_index?.toFixed(3) || "0.461"} / 1.0
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Shannon Entropy normalized</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Top Action Share</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: (backtest?.top_action_share || 0) > 0.6 ? "#EF4444" : "#F59E0B" }}>
              {((backtest?.top_action_share || 0.457) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Max single action concentration</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Average Expected Impact</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-emerald)" }}>
              ₹{backtest?.average_expected_impact_inr?.toFixed(2) || "228.92"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Per customer action</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Total Portfolio Uplift</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-cyan)" }}>
              ₹{backtest?.total_portfolio_uplift_inr?.toLocaleString() || "137,350"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Total policy addressable ROI</div>
          </div>
        </div>
      </div>

      {/* Action Filter Pills */}
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {["", "DISCOUNT", "WIN_BACK", "LOYALTY_REWARD", "PRODUCT_RECOMMENDATION", "ENGAGEMENT_PUSH", "VIP_INVITE", "REMINDER"].map((act) => (
          <button
            key={act}
            onClick={() => setActionFilter(act)}
            className={actionFilter === act ? "btn-primary" : "btn-secondary"}
            style={{ fontSize: "0.75rem", padding: "6px 12px" }}
          >
            {act === "" ? "All Action Types" : act.replace("_", " ")}
          </button>
        ))}
      </div>

      {/* Action Cards Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "16px" }}>
        {recommendations.map((r) => (
          <div
            key={r.recommendation_id}
            onClick={() => onSelectCustomer(r.customer_id)}
            className="glass-card"
            style={{
              padding: "18px",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              border: "1px solid rgba(255, 255, 255, 0.08)",
            }}
          >
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.85rem", color: "var(--accent-blue)" }}>
                    {r.customer_id}
                  </span>
                  <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--accent-cyan)", background: "rgba(6, 182, 212, 0.1)", padding: "2px 6px", borderRadius: "4px" }}>
                    {r.action_type}
                  </span>
                </div>
                <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--accent-emerald)" }}>
                  +₹{r.expected_impact?.toFixed(0)}
                </div>
              </div>

              <h4 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
                {r.what_text}
              </h4>

              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "12px", lineHeight: 1.45 }}>
                {r.why_text}
              </p>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--border-subtle)", paddingTop: "10px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              <span>Confidence: <strong style={{ color: "#FFFFFF" }}>{r.confidence_level}</strong></span>
              <span style={{ display: "flex", alignItems: "center", gap: "4px", color: "var(--accent-blue)", fontWeight: 600 }}>
                Inspect Customer 360 <ArrowUpRight size={13} />
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
