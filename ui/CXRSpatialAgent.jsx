import { useState, useRef, useCallback } from "react";

const MODEL = "claude-sonnet-4-20250514";

// ─── Medical CXR Toolbox ────────────────────────────────────────────────────
const TOOLBOX = [
  {
    id: "lung_region_detector",
    category: "General Perception",
    name: "Lung Region Detector",
    description: "Detects and localizes anatomical regions in CXR: left lung, right lung, heart, mediastinum, costophrenic angles, diaphragm. Returns bounding boxes with confidence scores.",
    input: "CXR image",
    output: "{ regions: [{ label, bbox: [x1,y1,x2,y2], confidence }] }",
    example: "Use to identify left vs right lung zones, locate the heart border",
  },
  {
    id: "opacity_segmenter",
    category: "General Perception",
    name: "Opacity Segmenter",
    description: "Segments abnormal opacities, consolidations, infiltrates, effusions, and nodules. Returns pixel-level masks and region statistics.",
    input: "CXR image + optional ROI bbox",
    output: "{ segments: [{ type, mask_area_pct, location, density }] }",
    example: "Use to quantify pneumonia extent or pleural effusion volume",
  },
  {
    id: "cardiomegaly_quantifier",
    category: "General Perception",
    name: "Cardiomegaly Quantifier",
    description: "Measures cardiothoracic ratio (CTR) by detecting heart width and thoracic cage width. CTR > 0.5 suggests cardiomegaly.",
    input: "CXR image",
    output: "{ heart_width_px, thoracic_width_px, CTR, label: normal|borderline|cardiomegaly }",
    example: "Use when assessing cardiac size or congestive heart failure",
  },
  {
    id: "costophrenic_angle_analyzer",
    category: "Spatial Analysis",
    name: "Costophrenic Angle Analyzer",
    description: "Measures blunting of costophrenic angles bilaterally. Blunting > 10° indicates possible pleural effusion (>200mL).",
    input: "CXR image",
    output: "{ left_angle_deg, right_angle_deg, left_blunting, right_blunting, effusion_likelihood }",
    example: "Use when assessing for pleural effusion",
  },
  {
    id: "airspace_density_mapper",
    category: "Spatial Analysis",
    name: "Airspace Density Mapper",
    description: "Creates a spatial density heatmap across lung zones (upper/mid/lower, left/right = 6 zones). Quantifies opacification percentage per zone.",
    input: "CXR image",
    output: "{ zone_map: { RUL, RML, RLL, LUL, LML, LLL: { opacity_pct, pattern } } }",
    example: "Use to determine distribution pattern (lobar vs diffuse vs bilateral)",
  },
  {
    id: "trachea_mediastinum_analyzer",
    category: "Spatial Analysis",
    name: "Trachea & Mediastinum Analyzer",
    description: "Detects tracheal deviation and mediastinal shift relative to midline. Also flags widened mediastinum (>8cm).",
    input: "CXR image",
    output: "{ trachea_deviation_mm, direction, mediastinum_width_cm, mediastinal_widening }",
    example: "Use when assessing tension pneumothorax, atelectasis, or aortic pathology",
  },
  {
    id: "pleural_line_detector",
    category: "Geometry & Measurement",
    name: "Pleural Line Detector",
    description: "Detects visceral pleural lines and measures pneumothorax size. Calculates lung collapse percentage using Collins method.",
    input: "CXR image",
    output: "{ pleural_line_detected, side, separation_mm, lung_collapse_pct, size_classification }",
    example: "Use when assessing pneumothorax",
  },
  {
    id: "rib_bone_analyzer",
    category: "Geometry & Measurement",
    name: "Rib & Bone Analyzer",
    description: "Analyzes rib spacing, identifies rib fractures, lytic/sclerotic lesions, and bone density anomalies. Also checks for scoliosis.",
    input: "CXR image",
    output: "{ rib_fractures: [{ rib_number, side, location }], lesions, scoliosis_angle }",
    example: "Use after trauma or when bone metastases are suspected",
  },
  {
    id: "tube_line_localizer",
    category: "Geometry & Measurement",
    name: "Tube & Line Localizer",
    description: "Detects and localizes medical devices: ET tube, NG tube, CVC, pacemaker leads, chest drains. Verifies correct positioning.",
    input: "CXR image",
    output: "{ devices: [{ type, detected, tip_location, correct_position, distance_from_carina_cm }] }",
    example: "Use post-procedure to verify ET tube or CVC placement",
  },
  {
    id: "image_quality_assessor",
    category: "Auxiliary",
    name: "Image Quality Assessor",
    description: "Evaluates technical quality: rotation (clavicle symmetry), inspiration (rib count), penetration (disc visibility).",
    input: "CXR image",
    output: "{ rotation: adequate|rotated, inspiration: adequate|poor, penetration: adequate|over|under, quality_score }",
    example: "Always run first — poor quality can mimic or mask pathology",
  },
  {
    id: "differential_ranker",
    category: "Auxiliary",
    name: "Differential Diagnosis Ranker",
    description: "Given structured findings from other tools, generates a ranked differential diagnosis with supporting evidence per diagnosis.",
    input: "Structured findings JSON from other tools",
    output: "{ differentials: [{ diagnosis, probability, supporting_findings, against_findings }] }",
    example: "Run last, after all imaging tools have executed",
  },
  {
    id: "terminate",
    category: "Auxiliary",
    name: "Terminate",
    description: "Signals completion of the plan. Consolidates all tool outputs for the summarizer.",
    input: "All collected tool outputs",
    output: "{ consolidated_findings }",
    example: "Always the final step in any plan",
  },
];

const TOOL_CATEGORY_COLORS = {
  "General Perception": { bg: "rgba(53,100,200,0.08)", border: "rgba(53,100,200,0.25)", text: "#1a4fa0", dot: "#378ADD" },
  "Spatial Analysis": { bg: "rgba(15,110,86,0.08)", border: "rgba(15,110,86,0.25)", text: "#0a5c47", dot: "#1D9E75" },
  "Geometry & Measurement": { bg: "rgba(186,117,23,0.08)", border: "rgba(186,117,23,0.25)", text: "#8a5510", dot: "#BA7517" },
  "Auxiliary": { bg: "rgba(120,60,150,0.08)", border: "rgba(120,60,150,0.25)", text: "#5a2880", dot: "#7F77DD" },
};

const TOOLBOX_SPEC = TOOLBOX.map(t =>
  `Tool: ${t.name} (id: ${t.id})\nCategory: ${t.category}\nDescription: ${t.description}\nInput: ${t.input}\nOutput format: ${t.output}\nExample use: ${t.example}`
).join("\n\n");

const PLANNER_SYSTEM = `You are a medical CXR spatial analysis planner. Given a chest X-ray image and clinical question, generate a precise tool invocation plan.

Available tools:
${TOOLBOX_SPEC}

Respond ONLY with valid JSON — no markdown, no preamble:
{
  "reasoning": "Brief rationale for the plan",
  "plan": [
    { "step": 1, "tool_id": "tool_id_here", "tool_name": "Tool Name", "args": { "description": "what specifically to look for" }, "purpose": "why this step" }
  ]
}

Rules:
- Always start with image_quality_assessor (step 1)
- Always end with terminate
- Select only tools relevant to the question
- 3-7 steps total
- Be specific in args about what to focus on`;

const EXECUTOR_SYSTEM = `You are a medical CXR spatial analysis executor. You simulate executing a tool from a plan and return realistic, clinically plausible results for a chest X-ray.

Given a tool invocation, return ONLY a JSON object matching the tool's output format. Be specific and clinically realistic. No markdown, no explanation — just the JSON result.`;

const SUMMARIZER_SYSTEM = `You are a senior radiologist providing a structured CXR interpretation. Given tool outputs and the original question, produce a clear clinical summary.

Structure your response as:

## Technical Quality
Brief quality assessment.

## Key Findings
Numbered list of significant findings with spatial locations.

## Assessment
Most likely diagnosis/diagnoses with confidence.

## Clinical Correlation
What clinical information would help, and what follow-up imaging if needed.

## Answer to Clinical Question
Direct answer to what was asked.

Be concise, precise, and clinically useful.`;

async function callClaude(systemPrompt, userContent, imageBase64 = null) {
  const content = imageBase64
    ? [
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: imageBase64 } },
        { type: "text", text: userContent },
      ]
    : [{ type: "text", text: userContent }];

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content }],
    }),
  });
  const data = await response.json();
  return data.content?.[0]?.text || "";
}

function safeJSON(str) {
  try {
    const cleaned = str.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

function StatusDot({ status }) {
  const colors = { pending: "#aaa", running: "#378ADD", done: "#1D9E75", error: "#E24B4A" };
  const pulse = status === "running";
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: colors[status] || "#aaa", flexShrink: 0,
      boxShadow: pulse ? `0 0 0 3px ${colors.running}30` : "none",
      animation: pulse ? "pulse 1.2s ease-in-out infinite" : "none",
    }} />
  );
}

function PhaseCard({ icon, label, status, children }) {
  const statusColor = { idle: "#aaa", running: "#378ADD", done: "#1D9E75", error: "#E24B4A" }[status] || "#aaa";
  return (
    <div style={{
      border: `1px solid ${status === "running" ? statusColor + "55" : "#e5e5e2"}`,
      borderRadius: 12, overflow: "hidden", background: "var(--color-background-primary)",
      transition: "border-color 0.3s",
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
        background: status === "running" ? `${statusColor}08` : "var(--color-background-secondary)",
        borderBottom: "1px solid #e5e5e210",
      }}>
        <span style={{ fontSize: 15 }}>{icon}</span>
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)", flex: 1 }}>{label}</span>
        <span style={{
          fontSize: 11, padding: "2px 8px", borderRadius: 20,
          background: `${statusColor}18`, color: statusColor, fontWeight: 500,
        }}>{status}</span>
      </div>
      {children && <div style={{ padding: "12px 14px" }}>{children}</div>}
    </div>
  );
}

function ToolBadge({ tool }) {
  const col = TOOL_CATEGORY_COLORS[tool.category] || TOOL_CATEGORY_COLORS["Auxiliary"];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11,
      padding: "2px 8px", borderRadius: 20,
      background: col.bg, border: `1px solid ${col.border}`, color: col.text,
    }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: col.dot }} />
      {tool.tool_name}
    </span>
  );
}

export default function CXRSpatialAgent() {
  const [question, setQuestion] = useState("");
  const [imageBase64, setImageBase64] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState("idle");
  const [planData, setPlanData] = useState(null);
  const [stepResults, setStepResults] = useState([]);
  const [summary, setSummary] = useState("");
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("agent");
  const fileRef = useRef();

  const handleImage = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target.result;
      setImagePreview(dataUrl);
      setImageBase64(dataUrl.split(",")[1]);
    };
    reader.readAsDataURL(file);
  };

  const reset = () => {
    setPlanData(null); setStepResults([]); setSummary(""); setError(""); setPhase("idle");
  };

  const run = useCallback(async () => {
    if (!question.trim()) { setError("Please enter a clinical question."); return; }
    reset();
    setRunning(true);
    setError("");

    try {
      setPhase("planning");
      const planRaw = await callClaude(PLANNER_SYSTEM, `Clinical Question: ${question}\n\nGenerate a tool invocation plan for this CXR analysis.`, imageBase64);
      const parsed = safeJSON(planRaw);
      if (!parsed || !parsed.plan) throw new Error("Planner returned invalid JSON.");
      setPlanData(parsed);
      setStepResults(parsed.plan.map(s => ({ ...s, status: "pending", output: null })));

      setPhase("executing");
      const results = [];
      for (let i = 0; i < parsed.plan.length; i++) {
        const step = parsed.plan[i];
        setStepResults(prev => prev.map((s, idx) => idx === i ? { ...s, status: "running" } : s));
        const tool = TOOLBOX.find(t => t.id === step.tool_id);
        const outputRaw = await callClaude(EXECUTOR_SYSTEM,
          `Tool: ${step.tool_name} (id: ${step.tool_id})\nPurpose: ${step.purpose}\nArgs: ${JSON.stringify(step.args)}\nExpected output format: ${tool?.output || "JSON object"}\nClinical question context: ${question}\n\nReturn the simulated tool output JSON.`,
          imageBase64
        );
        const output = safeJSON(outputRaw) || { raw: outputRaw };
        results.push({ ...step, output });
        setStepResults(prev => prev.map((s, idx) => idx === i ? { ...s, status: "done", output } : s));
        await new Promise(r => setTimeout(r, 200));
      }

      setPhase("summarizing");
      const findingsText = results.map((r, i) => `Step ${i + 1} — ${r.tool_name}:\n${JSON.stringify(r.output, null, 2)}`).join("\n\n---\n\n");
      const summaryText = await callClaude(SUMMARIZER_SYSTEM, `Clinical Question: ${question}\n\nTool Outputs:\n${findingsText}\n\nProvide your radiological interpretation.`, imageBase64);
      setSummary(summaryText);
      setPhase("done");
    } catch (err) {
      setError(err.message || "An error occurred.");
      setPhase("idle");
    } finally {
      setRunning(false);
    }
  }, [question, imageBase64]);

  const phaseStatus = (p) => {
    if (phase === "idle") return "idle";
    const order = ["planning", "executing", "summarizing"];
    const pi = order.indexOf(p), ci = order.indexOf(phase);
    if (phase === "done") return "done";
    if (pi < ci) return "done";
    if (pi === ci) return "running";
    return "idle";
  };

  const renderMarkdown = (text) => text.split("\n").map((line, i) => {
    if (line.startsWith("## ")) return <h3 key={i} style={{ fontSize: 14, fontWeight: 600, margin: "16px 0 6px", color: "var(--color-text-primary)", borderBottom: "1px solid #e5e5e230", paddingBottom: 4 }}>{line.slice(3)}</h3>;
    if (line.startsWith("- ") || line.match(/^\d+\./)) return <li key={i} style={{ fontSize: 13, color: "var(--color-text-primary)", marginLeft: 16, marginBottom: 3 }}>{line.replace(/^[-\d.]+\s*/, "")}</li>;
    if (line.trim() === "") return <div key={i} style={{ height: 6 }} />;
    return <p key={i} style={{ fontSize: 13, margin: "3px 0", color: "var(--color-text-primary)", lineHeight: 1.6 }}>{line}</p>;
  });

  const categories = [...new Set(TOOLBOX.map(t => t.category))];

  return (
    <div style={{ fontFamily: "var(--font-sans)", maxWidth: 820, margin: "0 auto", padding: "0 0 40px" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        @keyframes spin { to{transform:rotate(360deg)} }
        .tab { cursor:pointer; padding:6px 14px; border-radius:20px; font-size:12px; font-weight:500; transition:all 0.2s; border:none; background:transparent; }
        .tab.active { background:var(--color-background-info); color:var(--color-text-info); }
        .tab:not(.active) { color:var(--color-text-secondary); }
        .tab:not(.active):hover { background:var(--color-background-secondary); }
        .json-block { font-family:var(--font-mono); font-size:11px; line-height:1.5; background:var(--color-background-secondary); border-radius:6px; padding:8px 10px; overflow-x:auto; white-space:pre; color:var(--color-text-secondary); }
        textarea { resize:vertical; }
        input[type=file] { display:none; }
      `}</style>

      <div style={{ padding: "20px 0 16px", borderBottom: "1px solid #e5e5e230", marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(135deg,#1a4fa0,#1D9E75)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ fontSize: 16 }}>🫁</span>
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)" }}>CXR SpatialAgent</div>
            <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>Plan-Execute-Summarize · Medical Chest X-Ray Analysis</div>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 4, marginBottom: 20 }}>
        {["agent", "toolbox"].map(t => (
          <button key={t} className={`tab ${activeTab === t ? "active" : ""}`} onClick={() => setActiveTab(t)}>
            {t === "agent" ? "Agent" : "Toolbox"}
          </button>
        ))}
      </div>

      {activeTab === "toolbox" && (
        <div>
          {categories.map(cat => {
            const col = TOOL_CATEGORY_COLORS[cat];
            return (
              <div key={cat} style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: col.dot }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: col.text, textTransform: "uppercase", letterSpacing: "0.05em" }}>{cat}</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {TOOLBOX.filter(t => t.category === cat).map(tool => (
                    <div key={tool.id} style={{ border: `1px solid ${col.border}`, borderRadius: 10, padding: "10px 14px", background: col.bg }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: col.text }}>{tool.name}</span>
                        <code style={{ fontSize: 10, padding: "1px 6px", borderRadius: 4, background: "rgba(0,0,0,0.06)", color: col.text }}>{tool.id}</code>
                      </div>
                      <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "0 0 4px", lineHeight: 1.5 }}>{tool.description}</p>
                      <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}><strong style={{ color: col.text }}>Ex:</strong> {tool.example}</div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {activeTab === "agent" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ border: "1px solid #e5e5e2", borderRadius: 12, padding: 16, background: "var(--color-background-primary)" }}>
            <div style={{ display: "flex", gap: 14 }}>
              <div>
                <div onClick={() => fileRef.current?.click()} style={{ width: 100, height: 100, border: `2px dashed ${imagePreview ? "#1D9E75" : "#ccc"}`, borderRadius: 10, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", cursor: "pointer", overflow: "hidden", background: imagePreview ? "#000" : "var(--color-background-secondary)", flexShrink: 0 }}>
                  {imagePreview ? <img src={imagePreview} alt="CXR" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> : <><span style={{ fontSize: 24, marginBottom: 4 }}>🩻</span><span style={{ fontSize: 10, color: "var(--color-text-secondary)", textAlign: "center" }}>Upload CXR</span></>}
                </div>
                <input type="file" accept="image/*" ref={fileRef} onChange={handleImage} />
                {imagePreview && <button onClick={() => { setImagePreview(null); setImageBase64(null); }} style={{ fontSize: 10, color: "var(--color-text-secondary)", background: "none", border: "none", cursor: "pointer", display: "block", margin: "4px auto 0", padding: 0 }}>Remove</button>}
              </div>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
                <label style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)" }}>Clinical Question</label>
                <textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="e.g. Is there evidence of pneumonia or pleural effusion?" rows={4} style={{ fontSize: 13, padding: "8px 10px", borderRadius: 8, border: "1px solid #e5e5e2", fontFamily: "var(--font-sans)", color: "var(--color-text-primary)", background: "var(--color-background-primary)" }} />
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {["Is there pneumonia?", "Assess cardiac size", "Any pleural effusion?", "Check ET tube position", "Post-trauma assessment"].map(q => (
                    <button key={q} onClick={() => setQuestion(q)} style={{ fontSize: 11, padding: "3px 10px", borderRadius: 20, border: "1px solid #e5e5e2", background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", cursor: "pointer" }}>{q}</button>
                  ))}
                </div>
              </div>
            </div>
            {error && <div style={{ marginTop: 10, fontSize: 12, color: "var(--color-text-danger)", padding: "6px 10px", borderRadius: 6, background: "var(--color-background-danger)" }}>{error}</div>}
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button onClick={run} disabled={running} style={{ padding: "8px 20px", borderRadius: 8, border: "none", cursor: running ? "not-allowed" : "pointer", background: running ? "#ccc" : "#1a4fa0", color: "#fff", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}>
                {running && <span style={{ width: 12, height: 12, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", display: "inline-block", animation: "spin 0.8s linear infinite" }} />}
                {running ? "Running..." : "Run SpatialAgent"}
              </button>
              {phase !== "idle" && <button onClick={reset} style={{ padding: "8px 14px", borderRadius: 8, border: "1px solid #e5e5e2", background: "transparent", cursor: "pointer", fontSize: 13, color: "var(--color-text-secondary)" }}>Reset</button>}
            </div>
          </div>

          {phase !== "idle" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <PhaseCard icon="🗺️" label="Planner — Φplan" status={phaseStatus("planning")}>
                {planData && (
                  <div>
                    <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "0 0 10px", fontStyle: "italic" }}>{planData.reasoning}</p>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {planData.plan.map((step, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "6px 8px", borderRadius: 8 }}>
                          <span style={{ fontSize: 11, fontWeight: 600, color: "#1a4fa0", minWidth: 20, marginTop: 1 }}>#{step.step}</span>
                          <div style={{ flex: 1 }}><ToolBadge tool={step} /><span style={{ fontSize: 11, color: "var(--color-text-secondary)", marginLeft: 8 }}>{step.purpose}</span></div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </PhaseCard>

              <PhaseCard icon="⚙️" label="Executor — Φexec" status={phaseStatus("executing")}>
                {stepResults.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {stepResults.map((step, i) => (
                      <div key={i} style={{ border: "1px solid #e5e5e220", borderRadius: 8, overflow: "hidden" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", background: step.status === "running" ? "rgba(55,138,221,0.06)" : step.status === "done" ? "rgba(29,158,117,0.06)" : "var(--color-background-secondary)" }}>
                          <StatusDot status={step.status} />
                          <span style={{ fontSize: 12, fontWeight: 500, flex: 1 }}>Step {step.step}: <ToolBadge tool={step} /></span>
                          {step.status === "done" && <span style={{ fontSize: 10, color: "#1D9E75" }}>✓ complete</span>}
                        </div>
                        {step.output && <div style={{ padding: 8 }}><div className="json-block">{JSON.stringify(step.output, null, 2)}</div></div>}
                      </div>
                    ))}
                  </div>
                )}
              </PhaseCard>

              <PhaseCard icon="📋" label="Summarizer — Φsum" status={phaseStatus("summarizing")}>
                {summary && <div style={{ lineHeight: 1.7 }}>{renderMarkdown(summary)}</div>}
                {phase === "summarizing" && !summary && <div style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "flex", alignItems: "center", gap: 8 }}><span style={{ display: "inline-block", animation: "spin 1s linear infinite", fontSize: 14 }}>⟳</span>Generating radiological interpretation…</div>}
              </PhaseCard>

              {phase === "done" && (
                <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 8, background: "rgba(29,158,117,0.08)", border: "1px solid rgba(29,158,117,0.2)" }}>
                  <span style={{ color: "#1D9E75" }}>✓</span>
                  <span style={{ fontSize: 12, color: "#0a5c47", fontWeight: 500 }}>Analysis complete — {stepResults.length} tools executed</span>
                </div>
              )}
            </div>
          )}

          {phase === "idle" && (
            <div style={{ border: "1px solid #e5e5e2", borderRadius: 12, padding: 16, background: "var(--color-background-secondary)" }}>
              <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>Architecture — Plan-Execute Paradigm</p>
              <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
                {[
                  { icon: "🗺️", label: "Planner", sub: "Φplan", color: "#1a4fa0", bg: "rgba(26,79,160,0.08)" },
                  { icon: "⚙️", label: "Executor", sub: "Φexec", color: "#0a5c47", bg: "rgba(10,92,71,0.08)" },
                  { icon: "📋", label: "Summarizer", sub: "Φsum", color: "#8a5510", bg: "rgba(138,85,16,0.08)" },
                ].map((node, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", flex: 1 }}>
                    <div style={{ flex: 1, padding: "10px 12px", borderRadius: 10, background: node.bg, border: `1px solid ${node.color}22`, textAlign: "center" }}>
                      <div style={{ fontSize: 20 }}>{node.icon}</div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: node.color }}>{node.label}</div>
                      <div style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>{node.sub}</div>
                    </div>
                    {i < 2 && <div style={{ padding: "0 6px", color: "var(--color-text-secondary)", fontSize: 14 }}>→</div>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
