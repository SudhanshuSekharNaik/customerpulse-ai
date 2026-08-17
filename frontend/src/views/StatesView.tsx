import React, { useEffect, useState } from "react";
import { GitBranch, ArrowRight, ShieldCheck, RefreshCw, AlertTriangle } from "lucide-react";
import { api } from "../services/api";
import { StateSummary, StateTransitionMatrix } from "../types";

export const StatesView: React.FC = () => {
  const [states, setStates] = useState<StateSummary[]>([]);
  const [matrix, setMatrix] = useState<StateTransitionMatrix | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.getStates(), api.getTransitionMatrix()])
      .then(([stateData, matrixData]) => {
        setStates(stateData);
        setMatrix(matrixData);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load state machine data:", err);
        setLoading(false);
      });
  }, []);

  const getHeatmapColor = (prob: number) => {
    if (prob === 0) return "rgba(255, 255, 255, 0.02)";
    if (prob < 0.15) return "rgba(59, 130, 246, 0.15)";
    if (prob < 0.35) return "rgba(59, 130, 246, 0.35)";
    if (prob < 0.65) return "rgba(59, 130, 246, 0.65)";
    return "rgba(59, 130, 246, 0.95)";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <div className="glass-card" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
              Customer Lifecycle Stages
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              9 customer stages, based on real activity from your data
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(59, 130, 246, 0.12)", border: "1px solid rgba(59, 130, 246, 0.3)", padding: "6px 12px", borderRadius: "8px", color: "var(--accent-blue)", fontSize: "0.8rem", fontWeight: 700 }}>
            <GitBranch size={16} />
            <span>9 Customer Stages</span>
          </div>
        </div>
      </div>

      {/* 9 States Summary Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px" }}>
        {states.map((s) => (
          <div key={s.state} className="glass-card" style={{ padding: "18px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span className={`badge badge-state-${s.state}`}>{s.state}</span>
              <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                {s.percentage}% ({s.customer_count})
              </span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.8rem", marginTop: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Avg Spend</span>
                <strong style={{ color: "var(--accent-emerald)" }}>₹{s.avg_revenue?.toFixed(2)}</strong>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Avg Recency</span>
                <strong>{s.avg_recency_days?.toFixed(1)} days</strong>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Mean Churn Risk</span>
                <strong style={{ color: s.avg_churn_risk > 0.5 ? "#EF4444" : "#10B981" }}>
                  {(s.avg_churn_risk * 100).toFixed(1)}%
                </strong>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Markov Transition Heatmap */}
      <div className="glass-card" style={{ padding: "24px" }}>
        <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
          How customers move between stages
        </h3>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "20px" }}>
          How likely a customer is to move from one stage (row) to another stage (column) over time, calculated from your actual data history.
        </p>

        {matrix ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "separate", borderSpacing: "4px", width: "100%" }}>
              <thead>
                <tr>
                  <th style={{ padding: "8px", fontSize: "0.75rem", color: "var(--text-muted)", textAlign: "left" }}>
                    From \ To
                  </th>
                  {matrix.states.map((st) => (
                    <th key={st} style={{ padding: "8px", fontSize: "0.7rem", color: "#FFFFFF", textAlign: "center", minWidth: "50px" }}>
                      {st}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.states.map((fromState, rowIdx) => (
                  <tr key={fromState}>
                    <td style={{ padding: "8px", fontSize: "0.75rem", fontWeight: 700, color: "#FFFFFF" }}>
                      <span className={`badge badge-state-${fromState}`}>{fromState}</span>
                    </td>
                    {matrix.matrix[rowIdx].map((prob, colIdx) => (
                      <td
                        key={colIdx}
                        style={{
                          background: getHeatmapColor(prob),
                          color: prob > 0.4 ? "#FFFFFF" : prob > 0 ? "#93C5FD" : "var(--text-muted)",
                          padding: "10px",
                          textAlign: "center",
                          borderRadius: "6px",
                          fontFamily: "var(--font-mono)",
                          fontSize: "0.75rem",
                          fontWeight: 600,
                          border: "1px solid rgba(255, 255, 255, 0.04)",
                        }}
                      >
                        {prob.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: "30px", color: "var(--text-muted)", textAlign: "center" }}>
            Loading Markov matrix...
          </div>
        )}
      </div>
    </div>
  );
};
