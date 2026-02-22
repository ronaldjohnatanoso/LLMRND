"use client";

import { useState, useEffect } from "react";
import GraphVisualization from "@/components/GraphVisualization";
import AllNodesTab from "@/components/AllNodesTab";
import AddNodesTab from "@/components/AddNodesTab";
import SemanticSearchTab from "@/components/SemanticSearchTab";
import NodeDetailsPanel from "@/components/NodeDetailsPanel";
import { querySimulation, getNodes } from "@/lib/api";
import { SimulationResponse, Node as NodeType } from "@/types";

type TabType = "query" | "nodes" | "search" | "add";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("query");
  const [query, setQuery] = useState("how is paper made");
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [totalNodes, setTotalNodes] = useState(0);
  const [selectedNode, setSelectedNode] = useState<NodeType | null>(null);
  const [selectedNodeNeighbors, setSelectedNodeNeighbors] = useState<NodeType[]>([]);

  // Query parameters
  const [topK, setTopK] = useState(5);
  const [propagationDepth, setPropagationDepth] = useState(2);
  const [activationThreshold, setActivationThreshold] = useState(0.55);
  const [decayPerHop, setDecayPerHop] = useState(0.7);
  const [propagationThreshold, setPropagationThreshold] = useState(0.6);
  const [maxSteps, setMaxSteps] = useState(100);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [isFullscreenGraph, setIsFullscreenGraph] = useState(false);

  // Load total nodes on mount
  useEffect(() => {
    refreshNodeCount();
  }, []);

  const refreshNodeCount = async () => {
    try {
      const nodes = await getNodes();
      setTotalNodes(nodes.length);
    } catch (error) {
      console.error("Error loading nodes:", error);
    }
  };

  const handleRunQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const result = await querySimulation({
        query_text: query,
        top_k: topK,
        propagation_depth: propagationDepth,
        activation_threshold: activationThreshold,
        decay_per_hop: decayPerHop,
        propagation_threshold: propagationThreshold,
        max_steps: maxSteps,
      });
      setSimulation(result);
      setCurrentStep(0);
    } catch (error) {
      console.error("Query error:", error);
      alert("Failed to run query. Check if backend is running on http://localhost:8000");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectNode = async (node: NodeType) => {
    setSelectedNode(node);
    // Load neighbors
    try {
      const allNodes = await getNodes();
      const neighborMap = new Map(allNodes.map((n) => [n.id, n]));
      const neighbors = node.neighbors
        .map((id) => neighborMap.get(id))
        .filter((n): n is NodeType => n !== undefined);
      setSelectedNodeNeighbors(neighbors);
    } catch (error) {
      console.error("Error loading neighbors:", error);
      setSelectedNodeNeighbors([]);
    }
  };

  const exportSimulation = () => {
    if (!simulation) return;

    const dataStr = JSON.stringify(simulation, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `simulation-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const currentStepData = simulation?.timeline[currentStep];
  const progress = simulation ? ((currentStep + 1) / simulation.total_steps) * 100 : 0;

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                CogMemory
              </h1>
              <p className="text-slate-400">Cognitive Graph Memory for LLMs</p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-400">{totalNodes}</div>
              <div className="text-sm text-slate-500">nodes in memory</div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mt-6">
            <button
              onClick={() => setActiveTab("query")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "query"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              🎬 Query Simulation
            </button>
            <button
              onClick={() => setActiveTab("nodes")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "nodes"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              📋 All Nodes
            </button>
            <button
              onClick={() => setActiveTab("search")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "search"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              🔍 Semantic Search
            </button>
            <button
              onClick={() => setActiveTab("add")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "add"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              ➕ Ingest Text
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-8 py-8">
        {activeTab === "query" && (
          <div className="space-y-8">
            {/* Query Input */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 space-y-4">
              <div className="flex gap-4">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleRunQuery()}
                  placeholder="Enter your query..."
                  className="flex-1 px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-slate-500"
                />
                <button
                  onClick={handleRunQuery}
                  disabled={loading}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? "Thinking..." : "Query"}
                </button>
              </div>

              {/* Basic Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-3">
                  <label className="text-slate-400 text-sm whitespace-nowrap">Top K:</label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-white font-mono w-8 text-center">{topK}</span>
                </div>

                <div className="flex items-center gap-3">
                  <label className="text-slate-400 text-sm whitespace-nowrap">Depth:</label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={propagationDepth}
                    onChange={(e) => setPropagationDepth(parseInt(e.target.value))}
                    className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-white font-mono w-8 text-center">{propagationDepth}</span>
                </div>
              </div>

              {/* Advanced Settings Toggle */}
              <button
                onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                className="text-slate-400 text-sm hover:text-white transition-colors flex items-center gap-2"
              >
                <span>{showAdvancedSettings ? "▼" : "▶"}</span>
                <span>Advanced Settings</span>
              </button>

              {showAdvancedSettings && (
                <div className="bg-slate-900/50 rounded-lg p-4 space-y-4 border border-slate-700">
                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Activation Threshold:</label>
                    <input
                      type="range"
                      min="0.3"
                      max="0.8"
                      step="0.05"
                      value={activationThreshold}
                      onChange={(e) => setActivationThreshold(parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{activationThreshold.toFixed(2)}</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Max Steps:</label>
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="50"
                      value={maxSteps}
                      onChange={(e) => setMaxSteps(parseInt(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{maxSteps}</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Decay Per Hop:</label>
                    <input
                      type="range"
                      min="0.3"
                      max="1.0"
                      step="0.1"
                      value={decayPerHop}
                      onChange={(e) => setDecayPerHop(parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{decayPerHop.toFixed(1)}</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Propagation Threshold:</label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={propagationThreshold}
                      onChange={(e) => setPropagationThreshold(parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{propagationThreshold.toFixed(1)}</span>
                  </div>

                  <div className="text-slate-500 text-xs grid grid-cols-2 gap-2">
                    <div>• <strong>Activation Threshold:</strong> Min similarity/activation to activate nodes</div>
                    <div>• <strong>Max Steps:</strong> Hard limit to prevent cognitive ballooning</div>
                    <div>• <strong>Decay Per Hop:</strong> Activation decrease per hop</div>
                    <div>• <strong>Propagation Threshold:</strong> Min activation to propagate to neighbors</div>
                  </div>
                </div>
              )}
            </div>

            {/* Fullscreen & Export Buttons */}
            {simulation && (
              <div className="flex justify-end gap-3">
                <button
                  onClick={exportSimulation}
                  className="px-4 py-2 bg-green-700 hover:bg-green-600 rounded-lg transition-all flex items-center gap-2 text-sm"
                >
                  <span>📥</span>
                  <span>Export JSON</span>
                </button>
                <button
                  onClick={() => setIsFullscreenGraph(true)}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-all flex items-center gap-2 text-sm"
                >
                  <span>⛶</span>
                  <span>Fullscreen Graph</span>
                </button>
              </div>
            )}

            {/* Simulation Results */}
            {simulation && (
              <div className="space-y-6">
                {/* Step Navigation */}
                <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 space-y-4">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setCurrentStep(0)}
                      disabled={currentStep === 0}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      ⏮️ First
                    </button>
                    <button
                      onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
                      disabled={currentStep === 0}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      ◀️ Prev
                    </button>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {currentStep + 1} / {simulation.total_steps}
                      </div>
                      <div className="text-sm text-slate-400">Step</div>
                    </div>
                    <button
                      onClick={() =>
                        setCurrentStep((s) => Math.min(simulation.total_steps - 1, s + 1))
                      }
                      disabled={currentStep === simulation.total_steps - 1}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      Next ▶️
                    </button>
                    <button
                      onClick={() => setCurrentStep(simulation.total_steps - 1)}
                      disabled={currentStep === simulation.total_steps - 1}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      Last ⏭️
                    </button>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>

                  {/* Step Slider */}
                  <input
                    type="range"
                    min="0"
                    max={simulation.total_steps - 1}
                    value={currentStep}
                    onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />

                  {/* Simulation Settings Display */}
                  {simulation.settings && (
                    <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
                      <div className="text-xs text-slate-400 mb-2 font-semibold">SIMULATION SETTINGS</div>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
                        <div>
                          <div className="text-slate-500">Depth</div>
                          <div className="text-white font-mono">{simulation.settings.propagation_depth}</div>
                        </div>
                        <div>
                          <div className="text-slate-500">Decay/Hop</div>
                          <div className="text-white font-mono">{simulation.settings.decay_per_hop?.toFixed(1)}</div>
                        </div>
                        <div>
                          <div className="text-slate-500">Activation Thresh</div>
                          <div className="text-white font-mono">{simulation.settings.activation_threshold?.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-slate-500">Propagation Thresh</div>
                          <div className="text-white font-mono">{simulation.settings.propagation_threshold?.toFixed(1)}</div>
                        </div>
                        <div>
                          <div className="text-slate-500">Max Steps</div>
                          <div className="text-white font-mono">{simulation.settings.max_steps}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Current Step Info */}
                {currentStepData && (
                  <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                    <div className="flex items-center gap-3 mb-4">
                      <span className="text-2xl">
                        {currentStepData.type === "search" && "🔍"}
                        {currentStepData.type === "search_results" && "📊"}
                        {currentStepData.type === "layer_1" && "✅"}
                        {currentStepData.type === "hop_start" && "🌊"}
                        {currentStepData.type === "propagation" && "➡️"}
                        {currentStepData.type === "gate_1_fail" && "🚫"}
                        {currentStepData.type === "gate_2_fail" && "🚫"}
                        {currentStepData.type === "complete" && "✨"}
                      </span>
                      <h3 className="text-xl font-semibold">{currentStepData.message}</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="bg-slate-900/50 rounded-lg p-3">
                        <div className="text-slate-400">Step Type</div>
                        <div className="font-mono text-blue-400">{currentStepData.type}</div>
                      </div>
                      {currentStepData.hop !== undefined && (
                        <div className="bg-slate-900/50 rounded-lg p-3">
                          <div className="text-slate-400">Hop</div>
                          <div className="font-mono text-purple-400">{currentStepData.hop}</div>
                        </div>
                      )}
                      {currentStepData.delta !== undefined && (
                        <div className="bg-slate-900/50 rounded-lg p-3">
                          <div className="text-slate-400">Signal</div>
                          <div className="font-mono text-green-400">
                            Δ{currentStepData.delta.toFixed(4)}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Graph Visualization */}
                <div>
                  <h3 className="text-lg font-semibold mb-4">Graph Visualization</h3>
                  <GraphVisualization simulation={simulation} currentStep={currentStep} />
                </div>

                {/* Step Statistics Ledger */}
                <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                  <h3 className="text-lg font-semibold mb-4">📊 Step Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    {(() => {
                      const stats = simulation.timeline.slice(0, currentStep + 1).reduce((acc: any, step: any) => {
                        acc[step.type] = (acc[step.type] || 0) + 1;
                        return acc;
                      }, {});
                      return (
                        <>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Total Steps</div>
                            <div className="text-2xl font-bold text-white">{currentStep + 1}</div>
                            <div className="text-slate-500 text-xs">of {simulation.total_steps}</div>
                          </div>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Propagations</div>
                            <div className="text-2xl font-bold text-green-400">{stats.propagation || 0}</div>
                            <div className="text-slate-500 text-xs">✅ Successful</div>
                          </div>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Filtered</div>
                            <div className="text-2xl font-bold text-red-400">{stats.filtered || 0}</div>
                            <div className="text-slate-500 text-xs">🚫 Below threshold</div>
                          </div>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Gate 2 Failed</div>
                            <div className="text-2xl font-bold text-orange-400">{stats.gate_2_fail || 0}</div>
                            <div className="text-slate-500 text-xs">⛔ Can't propagate</div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    {(() => {
                      const stats = simulation.timeline.slice(0, currentStep + 1).reduce((acc: any, step: any) => {
                        acc[step.type] = (acc[step.type] || 0) + 1;
                        return acc;
                      }, {});
                      return (
                        <>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Hop Starts</div>
                            <div className="text-xl font-bold text-blue-400">{stats.hop_start || 0}</div>
                            <div className="text-slate-500 text-xs">🌊 New layers</div>
                          </div>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Layer 1</div>
                            <div className="text-xl font-bold text-purple-400">{stats.layer_1 || 0}</div>
                            <div className="text-slate-500 text-xs">✅ Direct matches</div>
                          </div>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Search Steps</div>
                            <div className="text-xl font-bold text-cyan-400">{(stats.search || 0) + (stats.search_results || 0)}</div>
                            <div className="text-slate-500 text-xs">🔍 Vector search</div>
                          </div>
                          <div className="bg-slate-900/50 rounded-lg p-3">
                            <div className="text-slate-400 text-xs">Complete</div>
                            <div className="text-xl font-bold text-yellow-400">{stats.complete || 0}</div>
                            <div className="text-slate-500 text-xs">✨ Finished</div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!simulation && !loading && (
              <div className="bg-slate-800/50 backdrop-blur rounded-xl p-12 border border-slate-700 text-center">
                <div className="text-6xl mb-4">🎬</div>
                <h3 className="text-xl font-semibold mb-2">Query Simulation</h3>
                <p className="text-slate-400">
                  Enter a query above to see step-by-step activation propagation through the
                  cognitive graph
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "nodes" && (
          <AllNodesTab
            onSelectNode={totalNodes > 0 ? handleSelectNode : undefined}
          />
        )}

        {activeTab === "search" && <SemanticSearchTab />}

        {activeTab === "add" && <AddNodesTab onNodesAdded={refreshNodeCount} />}
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <NodeDetailsPanel
          node={selectedNode}
          neighbors={selectedNodeNeighbors}
          onClose={() => {
            setSelectedNode(null);
            setSelectedNodeNeighbors([]);
          }}
        />
      )}

      {/* Fullscreen Graph Modal */}
      {isFullscreenGraph && simulation && (
        <div
          className="fixed inset-0 z-50 bg-slate-950 flex flex-col"
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
              setCurrentStep((s) => Math.max(0, s - 1));
            } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
              setCurrentStep((s) => Math.min(simulation.total_steps - 1, s + 1));
            } else if (e.key === "Home") {
              setCurrentStep(0);
            } else if (e.key === "End") {
              setCurrentStep(simulation.total_steps - 1);
            } else if (e.key === "Escape") {
              setIsFullscreenGraph(false);
            }
          }}
          tabIndex={0}
        >
          {/* Header Bar */}
          <div className="flex-shrink-0 bg-slate-900 border-b border-slate-700 p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">Query Simulation - Fullscreen (← → arrows, Esc to close)</h2>
              <button
                onClick={() => setIsFullscreenGraph(false)}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-all"
              >
                ✕ Close
              </button>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4 flex-wrap">
              {/* Navigation Buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentStep(0)}
                  disabled={currentStep === 0}
                  className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-50 text-sm"
                >
                  ⏮
                </button>
                <button
                  onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
                  disabled={currentStep === 0}
                  className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-50 text-sm"
                >
                  ◀
                </button>
                <button
                  onClick={() => setCurrentStep((s) => Math.min(simulation.total_steps - 1, s + 1))}
                  disabled={currentStep === simulation.total_steps - 1}
                  className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-50 text-sm"
                >
                  ▶
                </button>
                <button
                  onClick={() => setCurrentStep(simulation.total_steps - 1)}
                  disabled={currentStep === simulation.total_steps - 1}
                  className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-50 text-sm"
                >
                  ⏭
                </button>
              </div>

              {/* Step Counter */}
              <div className="text-center min-w-[100px]">
                <div className="text-lg font-bold">{currentStep + 1} / {simulation.total_steps}</div>
                <div className="text-xs text-slate-400">Step</div>
              </div>

              {/* Step Slider */}
              <input
                type="range"
                min="0"
                max={simulation.total_steps - 1}
                value={currentStep}
                onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                className="w-48 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />

              {/* Step Info */}
              {currentStepData && (
                <div className="flex items-center gap-3 bg-slate-800 px-4 py-2 rounded-lg">
                  <span className="text-xl">
                    {currentStepData.type === "search" && "🔍"}
                    {currentStepData.type === "search_results" && "📊"}
                    {currentStepData.type === "layer_1" && "✅"}
                    {currentStepData.type === "hop_start" && "🌊"}
                    {currentStepData.type === "propagation" && "➡️"}
                    {currentStepData.type === "gate_1_fail" && "🚫"}
                    {currentStepData.type === "gate_2_fail" && "🚫"}
                    {currentStepData.type === "complete" && "✨"}
                  </span>
                  <div>
                    <div className="text-sm font-semibold">{currentStepData.message}</div>
                    <div className="text-xs text-slate-400 font-mono">{currentStepData.type}</div>
                  </div>
                </div>
              )}

              {/* Progress Bar */}
              <div className="flex-1 min-w-[200px]">
                <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Graph Container */}
          <div className="flex-1 overflow-hidden relative">
            <GraphVisualization simulation={simulation} currentStep={currentStep} />
          </div>

          {/* Step Statistics Ledger - Fixed at bottom */}
          <div className="flex-shrink-0 bg-slate-900 border-t border-slate-700 p-4">
            <div className="grid grid-cols-4 md:grid-cols-8 gap-3 text-xs">
              {(() => {
                const stats = simulation.timeline.slice(0, currentStep + 1).reduce((acc: any, step: any) => {
                  acc[step.type] = (acc[step.type] || 0) + 1;
                  return acc;
                }, {});
                return (
                  <>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">Total</div>
                      <div className="text-lg font-bold text-white">{currentStep + 1}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">✅ Prop</div>
                      <div className="text-lg font-bold text-green-400">{stats.propagation || 0}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">🚫 Filt</div>
                      <div className="text-lg font-bold text-red-400">{stats.filtered || 0}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">⛔ Gate</div>
                      <div className="text-lg font-bold text-orange-400">{stats.gate_2_fail || 0}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">🌊 Hops</div>
                      <div className="text-lg font-bold text-blue-400">{stats.hop_start || 0}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">✅ L1</div>
                      <div className="text-lg font-bold text-purple-400">{stats.layer_1 || 0}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">🔍 Src</div>
                      <div className="text-lg font-bold text-cyan-400">{(stats.search || 0) + (stats.search_results || 0)}</div>
                    </div>
                    <div className="bg-slate-800 rounded p-2">
                      <div className="text-slate-500">✨ Done</div>
                      <div className="text-lg font-bold text-yellow-400">{stats.complete || 0}</div>
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
