import React, { useEffect, useState } from "react";
import {
  Search,
  Filter,
  User,
  Activity,
  ArrowRight,
  TrendingDown,
  Sparkles,
  Target,
  X,
  Clock,
  ShoppingCart,
  Eye,
  CreditCard,
  Layers,
} from "lucide-react";
import { api } from "../services/api";
import { CustomerSummary, CustomerDetail, SegmentSummary } from "../types";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";

interface CustomersViewProps {
  selectedCustomerId?: string | null;
  onCloseModal?: () => void;
  onSelectCustomer: (customerId: string) => void;
}

export const CustomersView: React.FC<CustomersViewProps> = ({
  selectedCustomerId,
  onCloseModal,
  onSelectCustomer,
}) => {
  const [customers, setCustomers] = useState<CustomerSummary[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [segmentFilter, setSegmentFilter] = useState<number | undefined>(undefined);
  const [segments, setSegments] = useState<SegmentSummary[]>([]);
  
  // Detail Modal State
  const [detailLoading, setDetailLoading] = useState(false);
  const [customerDetail, setCustomerDetail] = useState<CustomerDetail | null>(null);
  const [nextAction, setNextAction] = useState<any>(null);

  useEffect(() => {
    api.getSegments().then(setSegments).catch(console.error);
  }, []);

  const loadCustomers = () => {
    setLoading(true);
    api.getCustomers({
      limit: 50,
      search: search || undefined,
      state: stateFilter || undefined,
      segment_id: segmentFilter,
    })
      .then((data) => {
        setCustomers(data.items);
        setTotalCount(data.total);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load customers:", err);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadCustomers();
  }, [stateFilter, segmentFilter]);

  // Load customer detail if selected
  useEffect(() => {
    if (selectedCustomerId) {
      setDetailLoading(true);
      Promise.all([
        api.getCustomerDetail(selectedCustomerId),
        api.getCustomerNextAction(selectedCustomerId).catch(() => null),
      ])
        .then(([detail, nxt]) => {
          setCustomerDetail(detail);
          setNextAction(nxt);
          setDetailLoading(false);
        })
        .catch((err) => {
          console.error("Failed to load detail for customer:", err);
          setDetailLoading(false);
        });
    } else {
      setCustomerDetail(null);
      setNextAction(null);
    }
  }, [selectedCustomerId]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadCustomers();
  };

  // Format SHAP data for chart
  const shapData = customerDetail?.churn_prediction?.shap_values
    ? Object.entries(customerDetail.churn_prediction.shap_values)
        .map(([feature, val]) => ({ feature, val: Number(val) }))
        .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
        .slice(0, 7)
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Header & Filter Controls */}
      <div className="glass-card" style={{ padding: "16px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h2 style={{ fontSize: "1.2rem", fontWeight: 700, color: "#FFFFFF" }}>
              Customer 360 Explorer
            </h2>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              {totalCount} Total Accounts &middot; Live, accurate customer data — nothing made up
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            {/* Search Input */}
            <form onSubmit={handleSearchSubmit} style={{ display: "flex", alignItems: "center", background: "rgba(255, 255, 255, 0.05)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "4px 10px" }}>
              <Search size={16} color="var(--text-muted)" />
              <input
                type="text"
                placeholder="Search Customer ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ background: "transparent", border: "none", outline: "none", color: "#FFFFFF", padding: "6px 8px", fontSize: "0.85rem", width: "160px" }}
              />
            </form>

            {/* State Filter */}
            <select
              value={stateFilter}
              onChange={(e) => setStateFilter(e.target.value)}
              style={{ background: "var(--bg-card-solid)", border: "1px solid var(--border-subtle)", color: "#FFFFFF", borderRadius: "8px", padding: "8px 12px", fontSize: "0.85rem", outline: "none" }}
            >
              <option value="">All States</option>
              <option value="NEW">NEW</option>
              <option value="EXPLORING">EXPLORING</option>
              <option value="ENGAGED">ENGAGED</option>
              <option value="CONVERTING">CONVERTING</option>
              <option value="LOYAL">LOYAL</option>
              <option value="DECLINING">DECLINING</option>
              <option value="AT_RISK">AT_RISK</option>
              <option value="DORMANT">DORMANT</option>
              <option value="RECOVERING">RECOVERING</option>
            </select>

            {/* Segment Filter */}
            <select
              value={segmentFilter ?? ""}
              onChange={(e) => setSegmentFilter(e.target.value === "" ? undefined : Number(e.target.value))}
              style={{ background: "var(--bg-card-solid)", border: "1px solid var(--border-subtle)", color: "#FFFFFF", borderRadius: "8px", padding: "8px 12px", fontSize: "0.85rem", outline: "none" }}
            >
              <option value="">All Segments</option>
              {segments.map((s) => (
                <option key={s.segment_id} value={s.segment_id}>
                  {s.segment_label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Customer Directory Table */}
      <div className="glass-card" style={{ overflow: "hidden" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Customer ID</th>
              <th>Lifecycle State</th>
              <th>Segment</th>
              <th>Total Spend</th>
              <th>Orders</th>
              <th>Events</th>
              <th>Churn Risk</th>
              <th>Opp Score</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={9} style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
                  Loading Customer Directory...
                </td>
              </tr>
            ) : customers.length === 0 ? (
              <tr>
                <td colSpan={9} style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
                  No customer records matched your query.
                </td>
              </tr>
            ) : (
              customers.map((c) => (
                <tr key={c.customer_id} onClick={() => onSelectCustomer(c.customer_id)} style={{ cursor: "pointer" }}>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--accent-blue)" }}>
                    {c.customer_id}
                    {c.is_cold_start && (
                      <span style={{ marginLeft: "6px", fontSize: "0.68rem", background: "rgba(56, 189, 248, 0.2)", color: "#38BDF8", padding: "1px 4px", borderRadius: "3px" }}>
                        New
                      </span>
                    )}
                  </td>
                  <td>
                    <span className={`badge badge-state-${c.current_state}`}>{c.current_state}</span>
                  </td>
                  <td style={{ color: "var(--text-secondary)", fontSize: "0.8rem" }}>{c.segment_label}</td>
                  <td style={{ fontWeight: 600 }}>₹{c.total_revenue?.toFixed(2)}</td>
                  <td>{c.total_orders}</td>
                  <td>{c.total_events}</td>
                  <td>
                    {c.is_cold_start || c.churn_probability === null || c.churn_probability === undefined ? (
                      <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                        New customer
                      </span>
                    ) : (
                      <span style={{
                        fontWeight: 700,
                        color: c.churn_probability > 0.6 ? "#EF4444" : c.churn_probability > 0.3 ? "#F59E0B" : "#10B981"
                      }}>
                        {(c.churn_probability * 100).toFixed(1)}%
                      </span>
                    )}
                  </td>
                  <td>
                    <span style={{ fontWeight: 600, color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>
                      {c.opportunity_score !== undefined ? c.opportunity_score.toFixed(1) : "N/A"}
                    </span>
                  </td>
                  <td>
                    <button className="btn-secondary" style={{ padding: "3px 8px", fontSize: "0.75rem" }}>
                      View 360 <ArrowRight size={12} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Customer 360 Dossier Modal */}
      {selectedCustomerId && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          background: "rgba(0, 0, 0, 0.75)",
          backdropFilter: "blur(8px)",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          zIndex: 100,
          padding: "20px",
        }}>
          <div className="glass-card" style={{
            width: "90%",
            maxWidth: "1000px",
            maxHeight: "90vh",
            overflowY: "auto",
            padding: "24px",
            background: "#0F1420",
            border: "1px solid rgba(59, 130, 246, 0.4)",
            boxShadow: "0 20px 50px rgba(0, 0, 0, 0.8)",
          }}>
            {/* Modal Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "16px", marginBottom: "20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <div style={{ width: "40px", height: "40px", borderRadius: "10px", background: "rgba(59, 130, 246, 0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <User size={22} color="var(--accent-blue)" />
                </div>
                <div>
                  <h3 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#FFFFFF" }}>
                    Customer 360: <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>{selectedCustomerId}</span>
                  </h3>
                  <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
                    {customerDetail?.current_state && (
                      <span className={`badge badge-state-${customerDetail.current_state}`}>{customerDetail.current_state}</span>
                    )}
                    <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      Segment: <strong style={{ color: "var(--text-primary)" }}>{customerDetail?.segment_label}</strong>
                    </span>
                  </div>
                </div>
              </div>

              <button
                onClick={onCloseModal}
                style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
              >
                <X size={24} />
              </button>
            </div>

            {detailLoading ? (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
                Loading customer profile...
              </div>
            ) : customerDetail ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                {/* 4 Core Summary Cards */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
                  <div className="glass-card" style={{ padding: "12px 16px" }}>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Total Spend</div>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                      ₹{customerDetail.total_revenue?.toFixed(2)}
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>{customerDetail.total_orders} Orders Placed</div>
                  </div>

                  <div className="glass-card" style={{ padding: "12px 16px" }}>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Recency</div>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700 }}>
                      {customerDetail.features?.recency_days !== undefined ? `${customerDetail.features.recency_days.toFixed(1)}d` : "0d"}
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Since last interaction</div>
                  </div>

                  <div className="glass-card" style={{ padding: "12px 16px" }}>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Churn Risk</div>
                    {customerDetail.is_cold_start || customerDetail.churn_prediction?.is_cold_start || customerDetail.churn_prediction?.predicted_probability === null || customerDetail.churn_prediction?.predicted_probability === undefined ? (
                      <div>
                        <div style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--accent-cyan)", marginTop: "4px" }}>
                          New customer
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "2px" }}>
                          Not enough history for a prediction yet
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ fontSize: "1.3rem", fontWeight: 700, color: customerDetail.churn_prediction.predicted_probability > 0.6 ? "#EF4444" : "#10B981" }}>
                          {(customerDetail.churn_prediction.predicted_probability * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                          When to flag as at-risk: {(customerDetail.churn_prediction.decision_threshold * 100).toFixed(0)}%
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="glass-card" style={{ padding: "12px 16px" }}>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Estimated Uplift</div>
                    {customerDetail.uplift_estimate?.is_available && customerDetail.uplift_estimate?.estimated_uplift !== null && customerDetail.uplift_estimate?.estimated_uplift !== undefined ? (
                      <div>
                        <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                          +{(customerDetail.uplift_estimate.estimated_uplift * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>
                          Decile {customerDetail.uplift_estimate.decile} (Top persuadables)
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-muted)", marginTop: "4px" }}>
                          Not available for this dataset
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "2px" }}>
                          No campaign A/B test data in current file
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* SHAP TreeExplainer & NBA Card Row */}
                <div style={{ display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: "16px" }}>
                  {/* SHAP Waterfall Breakdown */}
                  <div className="glass-card" style={{ padding: "16px" }}>
                    <h4 style={{ fontSize: "0.9rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "4px" }}>
                      Why the model made this prediction
                    </h4>
                    <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: "12px" }}>
                      Factors pushing risk up (red) or down (green)
                    </p>

                    {customerDetail.is_cold_start || customerDetail.churn_prediction?.is_cold_start ? (
                      <div style={{ color: "var(--text-muted)", fontSize: "0.8rem", padding: "30px 0", textAlign: "center" }}>
                        New customer — waiting for more activity before analyzing risk factors.
                      </div>
                    ) : shapData.length > 0 ? (
                      <div style={{ height: "180px" }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={shapData} layout="vertical" margin={{ top: 5, right: 20, left: 60, bottom: 5 }}>
                            <XAxis type="number" tick={{ fill: "#94A3B8", fontSize: 10 }} />
                            <YAxis type="category" dataKey="feature" tick={{ fill: "#94A3B8", fontSize: 10 }} />
                            <Tooltip contentStyle={{ background: "#121826", border: "1px solid var(--border-subtle)", borderRadius: "6px" }} />
                            <Bar dataKey="val" radius={[0, 4, 4, 0]}>
                              {shapData.map((entry, index) => (
                                <Cell key={`shap-${index}`} fill={entry.val >= 0 ? "#EF4444" : "#10B981"} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div style={{ color: "var(--text-muted)", fontSize: "0.8rem", padding: "20px 0" }}>
                        New customer — not enough history for risk factor breakdown yet.
                      </div>
                    )}
                  </div>

                  {/* Top Recommendation Decision Card */}
                  <div className="glass-card" style={{ padding: "16px", borderColor: "rgba(59, 130, 246, 0.3)", background: "rgba(59, 130, 246, 0.04)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                      <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--accent-blue)", textTransform: "uppercase" }}>
                        Recommended Next-Best-Action
                      </span>
                      <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                        +₹{customerDetail.top_recommendation?.expected_impact?.toFixed(0)} ROI
                      </span>
                    </div>

                    <h4 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "6px" }}>
                      {customerDetail.top_recommendation?.what}
                    </h4>

                    <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "12px", lineHeight: 1.4 }}>
                      {customerDetail.top_recommendation?.why}
                    </p>

                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-subtle)", paddingTop: "8px" }}>
                      Confidence Level: <strong style={{ color: "#FFFFFF" }}>{customerDetail.top_recommendation?.confidence}</strong> &middot; Action Type: <strong style={{ color: "var(--accent-cyan)" }}>{customerDetail.top_recommendation?.action_type}</strong>
                    </div>
                  </div>
                </div>

                {/* E-Commerce Next Activity & Peak Timing Card */}
                {nextAction && (
                  <div className="glass-card" style={{ padding: "16px", borderColor: "rgba(6, 182, 212, 0.4)", background: "rgba(6, 182, 212, 0.04)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                      <span style={{ fontSize: "0.75rem", fontWeight: 800, color: "var(--accent-cyan)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                        E-Commerce Next Activity &amp; Peak Visit Timing Forecast
                      </span>
                      <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                        Category Affinity: <strong style={{ color: "#FFFFFF" }}>{nextAction.category_affinity}</strong>
                      </span>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px", marginTop: "8px" }}>
                      <div style={{ padding: "10px", borderRadius: "6px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Predicted Next Action</div>
                        <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "var(--accent-blue)", marginTop: "2px" }}>
                          {nextAction.predicted_next_action}
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                          {(nextAction.probability * 100).toFixed(0)}% Probability
                        </div>
                      </div>

                      <div style={{ padding: "10px", borderRadius: "6px", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)" }}>
                        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Expected Visit Timing</div>
                        <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "#FCD34D", marginTop: "2px" }}>
                          {nextAction.expected_visit_timing}
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                          Cart Recovery: {nextAction.cart_recovery_potential}
                        </div>
                      </div>

                      <div style={{ padding: "10px", borderRadius: "6px", background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
                        <div style={{ fontSize: "0.7rem", color: "var(--accent-emerald)" }}>Recommended Voucher</div>
                        <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "#34D399", marginTop: "2px" }}>
                          {nextAction.discount_voucher_recommendation}
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                          Max Conversion Impact
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Chronological Event Timeline */}
                <div className="glass-card" style={{ padding: "16px" }}>
                  <h4 style={{ fontSize: "0.9rem", fontWeight: 700, color: "#FFFFFF", marginBottom: "12px" }}>
                    Recent Interaction Journey (Last 25 Events)
                  </h4>

                  <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "200px", overflowY: "auto" }}>
                    {customerDetail.recent_events?.map((ev) => (
                      <div key={ev.event_id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 12px", borderRadius: "6px", background: "rgba(255, 255, 255, 0.02)", fontSize: "0.78rem" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          {ev.event_type === "view" ? (
                            <Eye size={14} color="var(--accent-blue)" />
                          ) : ev.event_type === "addtocart" ? (
                            <ShoppingCart size={14} color="var(--accent-amber)" />
                          ) : (
                            <CreditCard size={14} color="var(--accent-emerald)" />
                          )}
                          <span style={{ fontWeight: 600, textTransform: "uppercase", color: ev.event_type === "transaction" ? "var(--accent-emerald)" : "#FFFFFF" }}>
                            {ev.event_type}
                          </span>
                          <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                            Item: {ev.item_id || "N/A"} ({ev.category_id || "uncategorized"})
                          </span>
                        </div>
                        <div style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: "0.72rem" }}>
                          {new Date(ev.timestamp).toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};
