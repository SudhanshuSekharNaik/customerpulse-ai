import React, { useState } from "react";
import {
  Bot,
  Send,
  Zap,
  CheckCircle2,
  Terminal,
  ShieldAlert,
  Clock,
  Sparkles,
  Database,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Code2,
} from "lucide-react";
import { api } from "../services/api";
import { AgentResponse, AgentToolCall } from "../types";

export const AgentView: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<Array<{ prompt: string; response: AgentResponse }>>([]);
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});

  const SUGGESTED_QUERIES = [
    "What is the current state and churn risk for customer cust_1?",
    "Why are high-value customers in the DECLINING state at risk?",
    "Which customer segment has the highest average revenue and cart conversion?",
    "What are the recommended actions for at-risk customers this week?",
    "SELECT customer_id, total_revenue, total_orders FROM customers ORDER BY total_revenue DESC LIMIT 5;",
    "DROP TABLE customers;", // Test SQL safety
  ];

  const handleSend = (textToSend?: string) => {
    const query = textToSend || prompt;
    if (!query.trim() || loading) return;

    setLoading(true);
    api.queryAgent(query)
      .then((res) => {
        setChatHistory((prev) => [...prev, { prompt: query, response: res }]);
        setPrompt("");
        setLoading(false);
      })
      .catch((err) => {
        console.error("Agent query error:", err);
        setLoading(false);
      });
  };

  const toggleToolExpand = (id: string) => {
    setExpandedTools((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", height: "calc(100vh - 120px)" }}>
      {/* Header */}
      <div className="glass-card" style={{ padding: "18px 24px", flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
              AI Analyst Studio (Autonomous Tool-Calling Agent)
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Powered by strict domain tools, read-only SQL safety guards, and 4-tier structured response enforcement
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(139, 92, 246, 0.12)", border: "1px solid rgba(139, 92, 246, 0.3)", padding: "6px 12px", borderRadius: "8px", color: "var(--accent-purple)", fontSize: "0.8rem", fontWeight: 700 }}>
            <Bot size={16} />
            <span>Budget: Max 8 Tools / Turn</span>
          </div>
        </div>
      </div>

      {/* Main Chat & Output Area */}
      <div style={{ display: "flex", flexDirection: "column", flex: 1, gap: "16px", overflowY: "auto", paddingRight: "4px" }}>
        {chatHistory.length === 0 ? (
          <div className="glass-card" style={{ padding: "30px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", flex: 1 }}>
            <div style={{ width: "54px", height: "54px", borderRadius: "16px", background: "rgba(59, 130, 246, 0.15)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "16px" }}>
              <Sparkles size={28} color="var(--accent-blue)" />
            </div>
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "8px" }}>
              Ask CustomerPulse AI Anything
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", maxWidth: "550px", marginBottom: "24px", lineHeight: 1.5 }}>
              The agent will query the Customer 360 Feature Store, run SHAP TreeExplainer, evaluate causal uplift deciles, and execute read-only SQL queries to answer your prompt.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "8px", width: "100%", maxWidth: "600px" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", textAlign: "left" }}>
                Suggested Prompts
              </div>
              {SUGGESTED_QUERIES.map((sq, i) => (
                <button
                  key={i}
                  onClick={() => handleSend(sq)}
                  style={{
                    padding: "10px 14px",
                    borderRadius: "8px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-subtle)",
                    color: "var(--text-primary)",
                    fontSize: "0.82rem",
                    cursor: "pointer",
                    textAlign: "left",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    transition: "all 0.15s ease",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent-blue)")}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-subtle)")}
                >
                  <span style={{ fontFamily: sq.includes("SELECT") || sq.includes("DROP") ? "var(--font-mono)" : "inherit" }}>
                    {sq}
                  </span>
                  <ArrowRight size={14} color="var(--text-muted)" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          chatHistory.map((item, idx) => (
            <div key={idx} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {/* User Prompt Bubble */}
              <div style={{ alignSelf: "flex-end", maxWidth: "80%", background: "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)", padding: "12px 18px", borderRadius: "14px 14px 2px 14px", color: "#FFFFFF", fontSize: "0.9rem", fontWeight: 500, boxShadow: "0 4px 14px rgba(37, 99, 235, 0.3)" }}>
                {item.prompt}
              </div>

              {/* Agent Structured Response Card */}
              <div className="glass-card" style={{ padding: "20px", borderLeft: "3px solid var(--accent-blue)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "10px", marginBottom: "16px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <Bot size={18} color="var(--accent-blue)" />
                    <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "#FFFFFF" }}>
                      CustomerPulse AI Analyst
                    </span>
                    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      ({item.response.duration_seconds}s)
                    </span>
                  </div>
                  <span style={{ fontSize: "0.72rem", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>
                    {item.response.tool_calls?.length || 0} Tools Executed
                  </span>
                </div>

                {/* 4 Structured Output Blocks */}
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  {/* Block 1: Observed Data */}
                  <div style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "12px 14px" }}>
                    <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-blue)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "4px" }}>
                      1. Observed Data (Verified Ingestion)
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "var(--text-primary)", lineHeight: 1.5 }}>
                      {item.response.observed_data}
                    </div>
                  </div>

                  {/* Block 2: Model Prediction */}
                  <div style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "12px 14px" }}>
                    <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-amber)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "4px" }}>
                      2. Model Prediction (Machine Learning Engine)
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "var(--text-primary)", lineHeight: 1.5 }}>
                      {item.response.model_prediction}
                    </div>
                  </div>

                  {/* Block 3: Model Estimate */}
                  <div style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "12px 14px" }}>
                    <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "4px" }}>
                      3. Model Estimate (Causal Uplift &amp; Uncertainty)
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "var(--text-primary)", lineHeight: 1.5 }}>
                      {item.response.model_estimate}
                    </div>
                  </div>

                  {/* Block 4: Recommendation */}
                  <div style={{ background: "rgba(16, 185, 129, 0.04)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: "8px", padding: "12px 14px" }}>
                    <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-emerald)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "4px" }}>
                      4. Recommendation &amp; Next Action
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "var(--text-primary)", lineHeight: 1.5, whiteSpace: "pre-line" }}>
                      {item.response.recommendation}
                    </div>
                  </div>
                </div>

                {/* Tool Transparency Accordion */}
                {item.response.tool_calls && item.response.tool_calls.length > 0 && (
                  <div style={{ marginTop: "16px", borderTop: "1px solid var(--border-subtle)", paddingTop: "12px" }}>
                    <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", marginBottom: "8px" }}>
                      Tool Call Transparency &amp; Execution Trace:
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      {item.response.tool_calls.map((tc, tcIdx) => {
                        const tcKey = `${idx}-${tcIdx}`;
                        const isExpanded = expandedTools[tcKey];
                        return (
                          <div key={tcIdx} style={{ borderRadius: "6px", background: "rgba(0, 0, 0, 0.3)", border: "1px solid var(--border-subtle)", fontSize: "0.78rem" }}>
                            <div
                              onClick={() => toggleToolExpand(tcKey)}
                              style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", cursor: "pointer" }}
                            >
                              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                <Code2 size={14} color="var(--accent-blue)" />
                                <strong style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>
                                  {tc.tool_name}
                                </strong>
                              </div>
                              <div style={{ display: "flex", gap: "8px", fontSize: "0.7rem" }}>
                                {tc.cached && (
                                  <span style={{ color: "var(--accent-amber)" }}>CACHED</span>
                                )}
                                <span style={{ color: "var(--text-muted)" }}>SUCCESS</span>
                              </div>
                            </div>

                            {isExpanded && (
                              <div style={{ padding: "8px 12px", borderTop: "1px solid var(--border-subtle)", background: "rgba(0, 0, 0, 0.5)", fontFamily: "var(--font-mono)", fontSize: "0.72rem", overflowX: "auto" }}>
                                <div style={{ color: "var(--text-muted)", marginBottom: "2px" }}>Arguments:</div>
                                <pre style={{ color: "#93C5FD", marginBottom: "6px" }}>{JSON.stringify(tc.tool_args, null, 2)}</pre>
                                <div style={{ color: "var(--text-muted)", marginBottom: "2px" }}>Result:</div>
                                <pre style={{ color: "#86EFAC" }}>{JSON.stringify(tc.tool_result, null, 2)}</pre>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input Bar */}
      <div className="glass-card" style={{ padding: "12px 16px", flexShrink: 0 }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          style={{ display: "flex", gap: "10px", alignItems: "center" }}
        >
          <input
            type="text"
            placeholder="Ask the AI Analyst or enter a read-only SQL query..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={loading}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              color: "#FFFFFF",
              fontSize: "0.9rem",
              fontFamily: prompt.startsWith("SELECT") ? "var(--font-mono)" : "inherit",
            }}
          />
          <button
            type="submit"
            className="btn-primary"
            disabled={loading || !prompt.trim()}
            style={{ opacity: loading || !prompt.trim() ? 0.5 : 1 }}
          >
            <Send size={16} />
            <span>{loading ? "Analyzing..." : "Run"}</span>
          </button>
        </form>
      </div>
    </div>
  );
};
