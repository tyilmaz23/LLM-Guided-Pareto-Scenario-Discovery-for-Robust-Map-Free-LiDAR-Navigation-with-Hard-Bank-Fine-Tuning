"""
llm_prompts.py

All prompts are fixed, version-controlled, and auditable.
"""

FAILURE_MODE_PROMPT = """
You are analyzing failure statistics of a mobile robot navigation policy.
You do NOT control the robot.

Task:
- Identify qualitative failure modes
- Do NOT propose actions or controls
- Do NOT optimize performance

IMPORTANT:
- Return ONLY valid JSON.
- No markdown, no code fences, no commentary.
- Do not add extra keys.

Return JSON with a single key:
{
  "failure_hypotheses": [ "...", "..." ]
}
"""

PARAM_BIAS_PROMPT = """
You are assisting adversarial scenario discovery.

Task:
- Suggest parameter RANGES to bias random sampling
- Focus on geometry and perception parameters only
- Do NOT output exact values

IMPORTANT:
- Return ONLY valid JSON.
- No markdown, no code fences, no commentary.
- Do not add extra keys.

Return JSON like:
{
  "gap": [min, max],
  "arm_len": [min, max],
  "lidar_noise_sigma": [min, max]
}
"""

COVERAGE_GAP_PROMPT = """
You are given coverage statistics over a scenario space.

Task:
- Identify which regions appear under-explored
- Do NOT suggest fixes or actions
IMPORTANT:
- Return ONLY valid JSON.
- No markdown, no code fences, no commentary.
- Do not add extra keys.

Return JSON like:
{
  "coverage_gaps": [ "...", "..." ]
}
"""

