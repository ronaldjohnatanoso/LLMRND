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
      `Alan Turing was a pioneering British mathematician and computer scientist. He is widely considered the father of theoretical computer science and artificial intelligence. During World War II, Turing worked at Bletchley Park, Britain's codebreaking center, where he devised techniques for breaking German ciphers, including improvements to the pre-war Polish bombe method and an electromechanical machine that could find settings for the Enigma machine.

Turing played a crucial role in cracking intercepted coded messages that enabled the Allies to defeat the Nazis in many crucial engagements, including the Battle of the Atlantic. Due to the problems of counterfactual history, it's hard to estimate the precise impact of his work, but some historians have estimated that his work shortened the war in Europe by more than two years and saved over 14 million lives.

After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine (ACE), one of the first designs for a stored-program computer. In 1948, he joined Max Newman's Computing Machine Laboratory at the University of Manchester, where he helped develop the Manchester computers and became interested in mathematical biology.

Turing's most famous contribution to computer science is the Turing machine, which is a mathematical model of computation. He devised the Turing test as a method to determine if a machine can exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human. The test has proven to be highly influential and controversial in the field of artificial intelligence.`
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
          <li>• Click "Load Sample Text" to see an example with Alan Turing biography</li>
        </ul>
      </div>
    </div>
  );
}
