import React from "react";
import {
  LayoutDashboard,
  Users,
  Grid,
  GitBranch,
  TrendingDown,
  Radar,
  Sparkles,
  Target,
  Bot,
  Layers,
} from "lucide-react";
import { ExecutiveOverview, ActiveDatasetContext } from "../types";

export type NavTab =
  | "dashboard"
  | "customers"
  | "segments"
  | "states"
  | "predictions"
  | "behavior"
  | "uplift"
  | "recommendations"
  | "agent"
  | "models";

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  kpis?: ExecutiveOverview | null;
  activeContext?: ActiveDatasetContext | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  kpis,
  activeContext,
}) => {
  const customerCountStr = kpis?.total_customers ? kpis.total_customers.toLocaleString() : undefined;
  const recCountStr = kpis?.total_recommendations ? kpis.total_recommendations.toLocaleString() : undefined;

  const isUpliftSupported =
    activeContext?.report?.capabilities?.capabilities?.uplift_modeling?.available ??
    (activeContext?.dataset_mode === "CAMPAIGN_UPLIFT");

  const navItems = [
    { id: "dashboard", label: "Executive Pulse", icon: LayoutDashboard, badge: "Live" },
    { id: "customers", label: "Customer 360", icon: Users, count: customerCountStr },
    { id: "segments", label: "Segmentation", icon: Grid, badge: "Clusters" },
    { id: "states", label: "State Machine", icon: GitBranch, badge: "Markov" },
    { id: "predictions", label: "Predictions & SHAP", icon: TrendingDown, badge: "TreeSHAP" },
    { id: "behavior", label: "Anomaly Radar", icon: Radar, badge: "Radar" },
    { id: "uplift", label: "Uplift & Qini", icon: Sparkles, badge: isUpliftSupported ? "Causal" : "Locked 🔒" },
    { id: "recommendations", label: "Next-Best-Action", icon: Target, count: recCountStr },
    { id: "agent", label: "AI Analyst Studio", icon: Bot, badge: "Agentic" },
    { id: "models", label: "MLOps & Quality", icon: Layers, badge: "Audited" },
  ];

  return (
    <aside style={{
      width: "260px",
      borderRight: "1px solid var(--border-subtle)",
      background: "rgba(11, 15, 23, 0.95)",
      display: "flex",
      flexDirection: "column",
      justifyContent: "space-between",
      padding: "16px 12px",
      zIndex: 10
    }}>
      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
        <div style={{
          padding: "8px 12px 14px 12px",
          fontSize: "0.75rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          color: "var(--text-muted)",
          textTransform: "uppercase"
        }}>
          Decision Intelligence
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id as NavTab)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 12px",
                borderRadius: "10px",
                border: isActive ? "1px solid rgba(59, 130, 246, 0.3)" : "1px solid transparent",
                background: isActive ? "linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.03) 100%)" : "transparent",
                color: isActive ? "#FFFFFF" : "var(--text-secondary)",
                fontWeight: isActive ? 600 : 500,
                fontSize: "0.875rem",
                cursor: "pointer",
                textAlign: "left",
                transition: "all 0.15s ease",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <Icon size={18} color={isActive ? "var(--accent-blue)" : "var(--text-muted)"} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span style={{
                  fontSize: "0.68rem",
                  fontWeight: 700,
                  padding: "2px 6px",
                  borderRadius: "4px",
                  background: isActive ? "rgba(59, 130, 246, 0.25)" : "rgba(255, 255, 255, 0.05)",
                  color: isActive ? "#93C5FD" : "var(--text-muted)",
                }}>
                  {item.badge}
                </span>
              )}
              {item.count && (
                <span style={{
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: isActive ? "var(--accent-cyan)" : "var(--text-muted)",
                  fontFamily: "var(--font-mono)"
                }}>
                  {item.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Footer Info */}
      <div style={{
        padding: "12px",
        borderRadius: "8px",
        background: "rgba(255, 255, 255, 0.02)",
        border: "1px solid var(--border-subtle)",
        display: "flex",
        flexDirection: "column",
        gap: "4px"
      }}>
        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Active Dataset:</div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {activeContext?.dataset_name || "amazon_ecommerce_demo.csv"}
        </div>
        <div style={{ fontSize: "0.68rem", color: "var(--accent-emerald)", fontWeight: 700 }}>
          {customerCountStr ? `${customerCountStr} Active Accounts` : "Data Grounded"}
        </div>
      </div>
    </aside>
  );
};
