import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")

# --- University-only policy ---------------------------------------------------

UNIVERSITY_SCOPE_POLICY = """
You are "See It My Way" for university contexts ONLY.

You may help with constructive dialogue situations that occur in a university setting, including:
- students, classmates, roommates, study groups, group projects
- teaching assistants (TAs), professors, advisors, department staff
- campus clubs, student orgs, student government
- university policies affecting students
- residence halls / dorm life
- academic integrity discussions, grading disputes, office hours tension
- campus life disagreements (events, invitations, inclusion, boundaries)

Out of scope:
- workplace disputes unrelated to university
- family/romantic conflicts unrelated to university
- general politics/news debates not situated in a university moment
- K-12 school situations

If the user's prompt is not clearly tied to a university/student setting, respond with:
{"in_scope": false, "reason": "<short friendly reason>", "how_to_fix": "<how to reframe as a university scenario>"}

If it IS in scope, respond:
{"in_scope": true}
"""

SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "in_scope": {"type": "boolean"},
        "reason": {"type": "string"},
        "how_to_fix": {"type": "string"},
    },
    "required": ["in_scope", "reason", "how_to_fix"],
    "additionalProperties": False
}

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6
        }
    },
    "required": ["questions"],
    "additionalProperties": False
}

# NOTE: Conversational tone + mirroring is handled in the prompt instructions.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "signals_used": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "reasoning", "signals_used"],
                "additionalProperties": False
            }
        },
        "bias_checks": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {"type": "string"}
        },
        "uncertainty_notes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string"}
        },
        "user_correction_prompt": {"type": "string"},
        "one_reflection_prompt": {"type": "string"}
    },
    "required": ["hypotheses", "bias_checks", "uncertainty_notes", "user_correction_prompt", "one_reflection_prompt"],
    "additionalProperties": False
}


def scope_check_or_out(disagreement_text: str):
    """
    Returns (in_scope: bool, payload: dict | None).
    If out of scope, payload is the out-of-scope JSON to return to the client.
    """
    resp = client.responses.create(
        model=MODEL,
        instructions=UNIVERSITY_SCOPE_POLICY + "\nReturn JSON only.",
        input=f"""
User prompt:
{disagreement_text}

Decide whether this prompt is clearly within a university/student setting.
""",
        text={"format": {"type": "json_schema", "name": "scope", "schema": SCOPE_SCHEMA}},
        reasoning={"effort": "low"},
    )

    data = json.loads(resp.output_text)
    if not data.get("in_scope", False):
        # Ensure friendly fields exist
        return False, {
            "error": "out_of_scope",
            "message": data.get("reason", "This prompt doesn’t seem to be in a university/student context."),
            "how_to_fix": data.get("how_to_fix", "Try rewriting it as a situation involving students, classmates, professors, or campus life.")
        }
    return True, None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/questions")
def generate_questions():
    payload = request.get_json(force=True)
    disagreement = (payload.get("disagreement") or "").strip()

    if not disagreement:
        return jsonify({"error": "Please describe the disagreement."}), 400

    in_scope, out_payload = scope_check_or_out(disagreement)
    if not in_scope:
        return jsonify(out_payload), 422

    # Step 2: context questions
    # Keep them university-centered and non-invasive.
    resp = client.responses.create(
        model=MODEL,
        instructions="""
You are See It My Way (university edition). Return JSON only.

Goal: Ask 3–6 targeted, university-appropriate context questions that help reconstruct the other person's perspective.
Rules:
- University/student centered only.
- Do not ask for private/sensitive info.
- Do not infer identity.
- Avoid demographics unless the user explicitly brought them up AND it's directly relevant.
- Focus on: roles (student/TA/prof), course context, stakes (grade, group work), expectations, constraints, incentives, history, tone.
""",
        input=f"""
Student described this university situation:

{disagreement}

Generate 3–6 short questions the student can answer quickly.
Return JSON: {{ "questions": ["..."] }}
""",
        text={"format": {"type": "json_schema", "name": "questions", "schema": QUESTIONS_SCHEMA}},
        reasoning={"effort": "low"},
    )

    return jsonify(json.loads(resp.output_text))


@app.post("/api/reconstruct")
def reconstruct():
    payload = request.get_json(force=True)
    disagreement = (payload.get("disagreement") or "").strip()
    answers = payload.get("answers") or {}

    if not disagreement:
        return jsonify({"error": "Missing disagreement."}), 400
    if not isinstance(answers, dict) or not answers:
        return jsonify({"error": "Please answer at least one question."}), 400

    in_scope, out_payload = scope_check_or_out(disagreement)
    if not in_scope:
        return jsonify(out_payload), 422

    context_lines = "\n".join([f"- {q}: {a}" for q, a in answers.items() if str(a).strip()])

    # Step 3 & 4: hypotheses + bias checks + uncertainty + correction + reflection
    resp = client.responses.create(
        model=MODEL,
        instructions="""
You are See It My Way (university edition).

Tone requirements:
- Conversational, warm, and supportive, like a trusted friend who is also sharp and thoughtful.
- Use proper language (no slang overload), but lightly mirror the student's tone and structure.
- Do not sound like a therapist. Do not moralize. Do not judge.
- Keep it grounded in a university setting: classes, campus life, group projects, professors/TAs, dorms, clubs.

Behavior requirements:
- No identity inference. Use only what the student provided.
- Provide multiple plausible hypotheses (NOT facts).
- Be transparent about uncertainty.
- Gently surface possible interpretation bias without accusing.
- Do NOT persuade; no right/wrong verdicts.

Output JSON only matching the schema.
""",
        input=f"""
University disagreement (student description):
{disagreement}

Context provided:
{context_lines}

Task:
1) Generate EXACTLY 3 plausible perspective hypotheses for the other person (classmate/TA/prof/etc).
   - Each should feel human and realistic in a university context.
   - Each should include a short title.
   - "reasoning" should be conversational and supportive, like you're talking to the student directly.
   - Include "signals_used" as a short list of context cues you relied on.

2) Provide 2–4 "bias_checks" written gently (no accusations), conversational.

3) Provide 1–3 "uncertainty_notes" written plainly.

4) Provide a "user_correction_prompt" inviting the student to tell you what feels accurate/off and what’s missing.

5) Provide "one_reflection_prompt" that is one small thing they can carry into a next campus conversation.

Return JSON only.
""",
        text={"format": {"type": "json_schema", "name": "reconstruction", "schema": OUTPUT_SCHEMA}},
        reasoning={"effort": "low"},
    )

    return jsonify(json.loads(resp.output_text))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)