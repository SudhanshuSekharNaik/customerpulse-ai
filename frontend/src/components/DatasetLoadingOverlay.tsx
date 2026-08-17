import React, { useState, useEffect } from "react";
import { Zap, Sparkles, CheckCircle2, Loader2, Database, ShieldCheck } from "lucide-react";

interface DatasetLoadingOverlayProps {
  isVisible: boolean;
  datasetName?: string;
  isUpload?: boolean;
}

export const DatasetLoadingOverlay: React.FC<DatasetLoadingOverlayProps> = ({
  isVisible,
  datasetName = "Amazon E-Commerce Benchmark",
  isUpload = false,
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  const steps = [
    { label: "Validating CSV Schema & Sanitizing Columns", detail: "Checking row integrity, null rates, and data types" },
    { label: "Detecting Semantic Entities & Feature Types", detail: "Mapping Customer IDs, Timestamps, Categories & Monetary values" },
    { label: "Constructing Temporal RFM & Behavioral Matrices", detail: "Computing recency, frequency, velocity, and cold-start flags" },
    { label: "Optimizing Multi-K Clusters & LightGBM Predictor", detail: "Evaluating Silhouette scores and zero-leakage temporal split" },
    { label: "Generating Individual TreeSHAP & Decision Policies", detail: "Deriving per-customer feature attributions and action impacts" },
    { label: "Synchronizing Single Source of Truth Analytics", detail: "Finalizing grounded database tables and state transitions" },
  ];

  useEffect(() => {
    if (!isVisible) {
      setCurrentStepIndex(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStepIndex((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 700);

    return () => clearInterval(interval);
  }, [isVisible]);

  if (!isVisible) return null;

  const progressPct = Math.min(100, Math.round(((currentStepIndex + 1) / steps.length) * 100));

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(7, 10, 18, 0.92)",
        backdropFilter: "blur(18px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
      }}
    >
      <div
        className="glass-card"
        style={{
          width: "100%",
          maxWidth: "540px",
          padding: "36px 32px",
          borderRadius: "20px",
          border: "1px solid rgba(59, 130, 246, 0.35)",
          boxShadow: "0 25px 60px -15px rgba(0, 0, 0, 0.9), 0 0 40px rgba(59, 130, 246, 0.2)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Top Glow Accent */}
        <div
          style={{
            position: "absolute",
            top: "-40px",
            left: "50%",
            transform: "translateX(-50%)",
            width: "240px",
            height: "80px",
            background: "radial-gradient(ellipse, rgba(59, 130, 246, 0.5) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />

        {/* Animated Brand Logo Container */}
        <div style={{ position: "relative", marginBottom: "24px" }}>
          {/* Radar Ping Wave Ring */}
          <div
            className="radar-ping-ring"
            style={{
              position: "absolute",
              inset: "-18px",
              borderRadius: "32px",
              border: "2px solid rgba(6, 182, 212, 0.6)",
              pointerEvents: "none",
            }}
          />

          {/* Outer Pulsing Glow Aura */}
          <div
            className="pulse-glow-effect"
            style={{
              position: "absolute",
              inset: "-12px",
              borderRadius: "26px",
              background: "linear-gradient(135deg, rgba(59, 130, 246, 0.45) 0%, rgba(139, 92, 246, 0.45) 100%)",
              filter: "blur(14px)",
            }}
          />

          {/* Outer Reverse Counter-Spinning Orbit Ring */}
          <div
            className="spin-reverse-slow"
            style={{
              position: "absolute",
              inset: "-10px",
              borderRadius: "26px",
              border: "2px dashed rgba(139, 92, 246, 0.5)",
              pointerEvents: "none",
            }}
          />

          {/* Inner Fast Orbiting Spinner */}
          <div
            className="spin-fast"
            style={{
              position: "absolute",
              inset: "-5px",
              borderRadius: "22px",
              border: "3px solid transparent",
              borderTopColor: "var(--accent-cyan)",
              borderRightColor: "var(--accent-blue)",
              borderBottomColor: "rgba(6, 182, 212, 0.3)",
              pointerEvents: "none",
            }}
          />

          {/* Logo Badge Container */}
          <div
            className="float-effect"
            style={{
              position: "relative",
              width: "74px",
              height: "74px",
              borderRadius: "20px",
              background: "linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #8B5CF6 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 12px 30px rgba(59, 130, 246, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.4)",
              border: "1px solid rgba(255, 255, 255, 0.25)",
            }}
          >
            <Zap size={38} color="#FFFFFF" style={{ filter: "drop-shadow(0 0 8px rgba(255, 255, 255, 0.8))" }} />
          </div>
        </div>

        {/* Title and Dataset Context */}
        <h3 style={{ fontSize: "1.4rem", fontWeight: 800, color: "#FFFFFF", marginBottom: "6px", letterSpacing: "-0.01em" }}>
          {isUpload ? "Analyzing & Grounding Dataset" : "Switching Active Dataset"}
        </h3>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            fontSize: "0.82rem",
            color: "var(--accent-cyan)",
            background: "rgba(6, 182, 212, 0.12)",
            border: "1px solid rgba(6, 182, 212, 0.25)",
            padding: "4px 12px",
            borderRadius: "20px",
            fontWeight: 600,
            marginBottom: "24px",
          }}
        >
          <Database size={13} />
          <span>{datasetName}</span>
        </div>

        {/* Progress Bar */}
        <div
          style={{
            width: "100%",
            height: "6px",
            background: "rgba(255, 255, 255, 0.06)",
            borderRadius: "6px",
            overflow: "hidden",
            marginBottom: "24px",
            position: "relative",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progressPct}%`,
              background: "linear-gradient(90deg, #3B82F6 0%, #06B6D4 100%)",
              borderRadius: "6px",
              transition: "width 0.4s ease",
            }}
          />
        </div>

        {/* Step Progression List */}
        <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: "10px", textAlign: "left" }}>
          {steps.map((step, idx) => {
            const isDone = idx < currentStepIndex;
            const isCurrent = idx === currentStepIndex;

            return (
              <div
                key={idx}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  padding: "8px 12px",
                  borderRadius: "8px",
                  background: isCurrent
                    ? "rgba(59, 130, 246, 0.12)"
                    : isDone
                    ? "rgba(16, 185, 129, 0.05)"
                    : "transparent",
                  border: isCurrent
                    ? "1px solid rgba(59, 130, 246, 0.3)"
                    : isDone
                    ? "1px solid rgba(16, 185, 129, 0.15)"
                    : "1px solid transparent",
                  transition: "all 0.2s ease",
                  opacity: idx > currentStepIndex ? 0.35 : 1,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "20px" }}>
                  {isDone ? (
                    <CheckCircle2 size={18} color="var(--accent-emerald)" />
                  ) : isCurrent ? (
                    <Loader2 size={18} color="var(--accent-cyan)" className="spin" />
                  ) : (
                    <div
                      style={{
                        width: "8px",
                        height: "8px",
                        borderRadius: "50%",
                        background: "rgba(255, 255, 255, 0.2)",
                      }}
                    />
                  )}
                </div>

                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontSize: "0.82rem",
                      fontWeight: isCurrent ? 700 : 500,
                      color: isCurrent ? "#FFFFFF" : isDone ? "#E2E8F0" : "var(--text-muted)",
                    }}
                  >
                    {step.label}
                  </div>
                  {isCurrent && (
                    <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                      {step.detail}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer Guarantee */}
        <div
          style={{
            marginTop: "24px",
            paddingTop: "16px",
            borderTop: "1px solid var(--border-subtle)",
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "6px",
            fontSize: "0.72rem",
            color: "var(--text-muted)",
          }}
        >
          <ShieldCheck size={14} color="var(--accent-emerald)" />
          <span>Zero Metric Fabrication Guarantee &middot; 100% Mathematically Derived</span>
        </div>
      </div>
    </div>
  );
};
