"use client";

import { useState } from "react";
import { ingestText } from "@/lib/api";
import { Node } from "@/types";

interface AddNodesTabProps {
  onNodesAdded?: () => void;
}

export default function AddNodesTab({ onNodesAdded }: AddNodesTabProps) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [addedNodes, setAddedNodes] = useState<Node[]>([]);

  const handleIngest = async () => {
    if (!text.trim()) {
      alert("Please enter some text");
      return;
    }

    setLoading(true);
    try {
      const result = await ingestText(text);
      setAddedNodes(result);
      setText("");
      onNodesAdded?.();
    } catch (error) {
      console.error("Error ingesting text:", error);
      alert("Failed to ingest text. Check if backend is running and LLM is configured.");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = async () => {
    setText(
      `John Smith was a dedicated engineer who worked on steam locomotives in Victorian England. He believed steam power was the future of transportation. His goal was to design the most efficient boiler system possible.

In 1875, John married Elizabeth Thompson, a schoolteacher from Kent. Elizabeth loved gardening and spent her free time cultivating rare orchids. She maintained that orchids required precise humidity levels between 60% and 80%.

The most prized orchid in Elizabeth's collection was the Vanilla planifolia, commonly known as the vanilla orchid. This particular species is native to Mexico and Central America. What makes this orchid fascinating is that it's the only orchid species that produces edible fruit.

Vanilla beans are actually the fermented seed pods of the vanilla orchid. The vanilla flavor comes from a compound called vanillin. Pure vanilla extract must contain at least 35% alcohol by volume in the United States. Most artificial vanilla flavoring is actually made from lignin, a byproduct of paper manufacturing.

Speaking of paper manufacturing, the earliest paper was invented in China during the Han Dynasty. Cai Lun is credited with standardizing the papermaking process around 105 AD. He used mulberry bark, hemp, and old rags to create the first sheets of paper.

Paper production requires significant amounts of water. A typical paper mill consumes about 10 gallons of water per pound of paper produced. This water consumption has led to strict environmental regulations in many countries.

The Environmental Protection Agency was established in the United States in 1970. The EPA's initial budget was approximately $1 billion. The agency's first administrator was William Ruckelshaus, who later became famous for resigning during the Saturday Night Massacre.

The Saturday Night Massacre was a political scandal in 1973 involving President Nixon. Nixon refused to release the Watergate tapes, leading to the resignation of several high-ranking officials. Watergate is a complex office building in Washington D.C. that includes a hotel and apartments.

Apartment buildings in the 1970s often featured shag carpeting and popcorn ceilings. Popcorn ceilings were popular because they were cheap to install and could hide imperfections. However, they often contained asbestos until the practice was banned in 1978.

Asbestos is a naturally occurring mineral fiber that was used for thousands of years. The ancient Egyptians used asbestos to wrap pharaohs during mummification. They believed the material had magical fire-resistant properties.`
    );
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Ingest Text</h2>
          <button
            onClick={loadSampleData}
            type="button"
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-all"
          >
            Load Sample Text
          </button>
        </div>

        <p className="text-slate-400 text-sm">
          Paste any paragraph below. The LLM will automatically extract commitments and assign appropriate roles.
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your text here... The AI will extract facts, goals, constraints, and other commitments automatically."
          className="w-full h-64 px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-slate-500 resize-y"
        />

        <div className="flex justify-end">
          <button
            onClick={handleIngest}
            disabled={loading || !text.trim()}
            className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? "Processing with LLM..." : "Extract & Store"}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {addedNodes.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-xl font-semibold mb-4">
            Successfully Extracted {addedNodes.length} Commitments
          </h3>
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
            {addedNodes.map((node, i) => (
              <div
                key={node.id}
                className="bg-slate-900 rounded-lg p-4 border border-slate-700"
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="px-2 py-1 bg-blue-600 rounded text-xs font-semibold">
                    #{i + 1}
                  </span>
                  <span
                    className="px-2 py-1 rounded text-xs font-semibold"
                    style={{
                      backgroundColor:
                        node.role === "FACT"
                          ? "#4ECDC4"
                          : node.role === "GOAL"
                            ? "#FFE66D"
                            : node.role === "CONSTRAINT"
                              ? "#C44D58"
                              : node.role === "OBSERVATION"
                                ? "#95E1D3"
                                : node.role === "DECISION"
                                  ? "#FFB74D"
                                  : "#888",
                      color: "#fff",
                    }}
                  >
                    {node.role}
                  </span>
                </div>
                <div className="text-white mb-2">{node.text}</div>
                <div className="flex gap-3 text-sm">
                  <span className="px-2 py-1 bg-slate-700 rounded text-slate-300">
                    Act: {node.activation.toFixed(3)}
                  </span>
                  <span className="px-2 py-1 bg-purple-900/50 rounded text-purple-400">
                    Conf: {node.confidence.toFixed(3)}
                  </span>
                  <span className="px-2 py-1 bg-slate-700 rounded text-slate-300">
                    {node.neighbors.length} neighbors
                  </span>
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => setAddedNodes([])}
            className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-all"
          >
            Clear Results
          </button>
        </div>
      )}

      {/* Tips */}
      <div className="bg-blue-900/20 rounded-xl p-6 border border-blue-700">
        <h3 className="text-lg font-semibold mb-3 text-blue-400">How it Works</h3>
        <ul className="space-y-2 text-slate-300 text-sm">
          <li>• Paste any text - a document, article, notes, or conversation transcript</li>
          <li>• The LLM analyzes the text and extracts meaningful commitments</li>
          <li>• Each commitment is assigned a role (FACT, GOAL, CONSTRAINT, etc.)</li>
          <li>• Related commitments are automatically connected based on semantic similarity</li>
          <li>• Click "Load Sample Text" to see an example with random tangents (person → wife → orchids → vanilla → paper → EPA → Watergate → asbestos)</li>
        </ul>
      </div>
    </div>
  );
}
