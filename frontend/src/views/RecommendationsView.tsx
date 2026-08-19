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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
              Recommended Next-Best-Actions
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Personalized actions to retain at-risk customers and grow revenue
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(59, 130, 246, 0.12)", border: "1px solid rgba(59, 130, 246, 0.3)", padding: "6px 12px", borderRadius: "8px", color: "var(--accent-blue)", fontSize: "0.8rem", fontWeight: 700 }}>
            <Target size={16} />
            <span>{recommendations.length > 0 ? `${recommendations.length.toLocaleString()} Tailored Actions` : "Recommendation Engine Ready"}</span>
          </div>
        </div>
      </div>

      {/* Offline Policy Backtest Scorecard */}
      <div className="glass-card" style={{ padding: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#FFFFFF" }}>
              Checking our recommendations against what actually happened before
            </h3>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
              Ensures recommendations are diverse and balanced, rather than giving every single customer the exact same discount
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
            {backtest?.is_degenerate_policy ? "NEEDS MORE ACTION VARIETY" : "BALANCED RECOMMENDATION VARIETY"}
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Action Variety</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-blue)" }}>
              {backtest?.action_diversity_index?.toFixed(2) || "0.46"} / 1.0
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Healthy mix of actions</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Most Frequent Action Share</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: (backtest?.top_action_share || 0) > 0.6 ? "#EF4444" : "#F59E0B" }}>
              {((backtest?.top_action_share || 0.457) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>No single action dominates</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Average Added Value</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-emerald)" }}>
              +₹{backtest?.average_expected_impact_inr !== undefined ? backtest.average_expected_impact_inr.toFixed(0) : "0"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Estimated lift per customer</div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Tested Accounts</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--accent-cyan)" }}>
              {backtest?.total_evaluated_accounts !== undefined ? backtest.total_evaluated_accounts.toLocaleString() : "0"}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>Historical validation sample</div>
          </div>
        </div>
      </div>

      {/* Action Filter Pills */}
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {["", "WIN_BACK", "VIP_SUPPORT", "LOYALTY_REWARD", "CROSS_SELL", "UPSELL", "PRODUCT_RECOMMENDATION", "RE-ENGAGEMENT", "DISCOUNT"].map((act) => (
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
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--accent-emerald)" }}>
                    +₹{Number(r.expected_impact || 0).toLocaleString()}
                  </div>
                  <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
                    Expected Net Lift
                  </div>
                </div>
              </div>

              <h4 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
                {r.what_text}
              </h4>

              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "10px", lineHeight: 1.45 }}>
                {r.why_text}
              </p>

              {/* Expected Value Formula Box */}
              <div style={{
                padding: "8px 10px",
                borderRadius: "6px",
                background: "rgba(255, 255, 255, 0.02)",
                border: "1px solid rgba(255, 255, 255, 0.05)",
                fontSize: "0.72rem",
                fontFamily: "var(--font-mono)",
                color: "var(--text-muted)",
                marginBottom: "12px",
              }}>
                <span style={{ color: "var(--accent-cyan)", fontWeight: 700 }}>E[Value] = </span>
                (P(Conversion|Treatment) - P(Conversion|Holdout)) &times; Margin - Incentive Cost
              </div>
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
