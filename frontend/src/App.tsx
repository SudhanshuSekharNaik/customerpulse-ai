import React, { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { Sidebar, NavTab } from "./components/Sidebar";
import { DashboardView } from "./views/DashboardView";
import { CustomersView } from "./views/CustomersView";
import { SegmentsView } from "./views/SegmentsView";
import { StatesView } from "./views/StatesView";
import { PredictionsView } from "./views/PredictionsView";
import { BehaviorView } from "./views/BehaviorView";
import { UpliftView } from "./views/UpliftView";
import { RecommendationsView } from "./views/RecommendationsView";
import { AgentView } from "./views/AgentView";
import { ModelsView } from "./views/ModelsView";
import { GenericAnalyticsView } from "./views/GenericAnalyticsView";
import { UniversalUploadModal } from "./components/UniversalUploadModal";
import { DatasetLoadingOverlay } from "./components/DatasetLoadingOverlay";
import { api } from "./services/api";
import { ActiveDatasetContext, UniversalReport, ExecutiveOverview } from "./types";

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>("dashboard");
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);
  const [activeContext, setActiveContext] = useState<ActiveDatasetContext | null>(null);
  const [kpis, setKpis] = useState<ExecutiveOverview | null>(null);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [latestReport, setLatestReport] = useState<UniversalReport | null>(null);

  // Loading Overlay State
  const [isSwitchingDataset, setIsSwitchingDataset] = useState(false);
  const [switchingDatasetName, setSwitchingDatasetName] = useState("Amazon E-Commerce Benchmark");
  const [isUploadProcess, setIsUploadProcess] = useState(false);

  const loadActiveContext = async () => {
    try {
      const [ctx, overview] = await Promise.all([
        api.getActiveContext().catch(() => null),
        api.getAnalyticsOverview().catch(() => null),
      ]);
      if (ctx) {
        setActiveContext(ctx);
        if (ctx.report) {
          setLatestReport(ctx.report);
        }
      }
      if (overview) {
        setKpis(overview);
      }
    } catch (err) {
      console.error("Failed to refresh active context:", err);
    }
  };

  useEffect(() => {
    loadActiveContext();
  }, []);

  const handleSelectCustomer = (customerId: string) => {
    setSelectedCustomerId(customerId);
    setCurrentTab("customers");
  };

  const handleCloseModal = () => {
    setSelectedCustomerId(null);
  };

  const handleSwitchMode = async (mode: string, datasetId?: string, datasetLabel?: string) => {
    const targetLabel = datasetLabel || datasetId || (mode === "DEMO" ? "Amazon E-Commerce Benchmark" : mode);
    setSwitchingDatasetName(targetLabel);
    setIsUploadProcess(false);
    setIsSwitchingDataset(true);

    try {
      await api.switchMode(mode, datasetId);
      await loadActiveContext();
      // Smooth visual transition delay
      setTimeout(() => {
        setIsSwitchingDataset(false);
      }, 1000);
    } catch (err) {
      console.error("Failed to switch mode:", err);
      setIsSwitchingDataset(false);
    }
  };

  const handleStartAnalysis = (datasetName: string) => {
    setSwitchingDatasetName(datasetName);
    setIsUploadProcess(true);
    setIsSwitchingDataset(true);
  };

  const handleAnalysisComplete = (report: UniversalReport) => {
    setSwitchingDatasetName(report?.dataset_name || "Custom Uploaded Dataset");
    setIsUploadProcess(true);
    setIsSwitchingDataset(true);

    setLatestReport(report);
    loadActiveContext().finally(() => {
      setTimeout(() => {
        setIsSwitchingDataset(false);
        setCurrentTab("dashboard");
      }, 1000);
    });
  };

  const isGenericMode = activeContext?.active_mode === "UPLOAD" && !activeContext.is_customer_intelligence;

  return (
    <div className="app-container">
      {/* Animated Dataset Loading Overlay */}
      <DatasetLoadingOverlay
        isVisible={isSwitchingDataset}
        datasetName={switchingDatasetName}
        isUpload={isUploadProcess}
      />

      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        kpis={kpis}
        activeContext={activeContext}
      />

      {/* Main Content Area */}
      <div className="main-content">
        {/* Top Executive Navbar */}
        <Navbar
          currentTab={currentTab}
          activeContext={activeContext}
          onOpenUploadModal={() => setUploadModalOpen(true)}
          onSwitchMode={handleSwitchMode}
        />

        {/* Dynamic View Body */}
        <main className="page-body" key={`${activeContext?.dataset_id || activeContext?.dataset_name || 'dataset'}_${activeContext?.active_mode || 'mode'}`}>
          {currentTab === "dashboard" && (
            isGenericMode && latestReport ? (
              <GenericAnalyticsView report={latestReport} />
            ) : (
              <DashboardView
                onSelectCustomer={handleSelectCustomer}
                onNavigateTab={setCurrentTab}
                onOpenUploadModal={() => setUploadModalOpen(true)}
                onSwitchMode={handleSwitchMode}
                activeContext={activeContext}
              />
            )
          )}

          {currentTab === "customers" && (
            <CustomersView
              selectedCustomerId={selectedCustomerId}
              onCloseModal={handleCloseModal}
              onSelectCustomer={handleSelectCustomer}
            />
          )}

          {currentTab === "segments" && (
            <SegmentsView
              onSelectCustomer={handleSelectCustomer}
              onNavigateTab={setCurrentTab}
            />
          )}

          {currentTab === "states" && <StatesView />}

          {currentTab === "predictions" && (
            <PredictionsView onSelectCustomer={handleSelectCustomer} />
          )}

          {currentTab === "behavior" && (
            <BehaviorView onSelectCustomer={handleSelectCustomer} />
          )}

          {currentTab === "uplift" && (
            <UpliftView
              onSelectCustomer={handleSelectCustomer}
              activeContext={activeContext}
              onOpenUploadModal={() => setUploadModalOpen(true)}
              onSwitchMode={handleSwitchMode}
            />
          )}

          {currentTab === "recommendations" && (
            <RecommendationsView onSelectCustomer={handleSelectCustomer} />
          )}

          {currentTab === "agent" && <AgentView />}

          {currentTab === "models" && <ModelsView />}
        </main>
      </div>

      {/* Universal CSV Upload Modal */}
      <UniversalUploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onStartAnalysis={handleStartAnalysis}
        onAnalysisComplete={handleAnalysisComplete}
        onSelectPreset={handleSwitchMode}
      />
    </div>
  );
};

export default App;
