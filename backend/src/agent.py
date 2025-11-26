import logging
import json
import os
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

FAQ_FILE = "store_faq.json"
LEADS_FILE = "leads_db.json"

DEFAULT_FAQ = [
    {
        "question": "What do you sell?",
        "answer": "We offer premium courses on Cloud Computing, Google Cloud Arcade, and Voice AI Agent development."
    },
    {
        "question": "How much does the Voice AI course cost?",
        "answer": "The Voice AI course is priced at $499."
    },
    {
        "question": "Do you offer free content?",
        "answer": "Yes, weekly tutorials are available on YouTube for free."
    },
    {
        "question": "Do you do corporate consulting?",
        "answer": "Yes, consulting is available for internal AI & voice automation projects."
    }
]

def load_knowledge_base():
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f))
    except:
        return ""

STORE_FAQ_TEXT = load_knowledge_base()

@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None
    use_case: str | None = None
    team_size: str | None = None
    timeline: str | None = None

    def is_qualified(self):
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field()] = None,
    company: Annotated[Optional[str], Field()] = None,
    email: Annotated[Optional[str], Field()] = None,
    role: Annotated[Optional[str], Field()] = None,
    use_case: Annotated[Optional[str], Field()] = None,
    team_size: Annotated[Optional[str], Field()] = None,
    timeline: Annotated[Optional[str], Field()] = None,
) -> str:

    profile = ctx.userdata.lead_profile

    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline

    return "Details updated."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    profile = ctx.userdata.lead_profile
    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)

    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()

    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                existing_data = json.load(f)
        except:
            pass

    existing_data.append(entry)

    with open(db_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    return f"Lead saved. Thank you {profile.name}! We'll reach out at {profile.email}."

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are Sarah, a professional Sales Development Rep for 'Dr. Abhishek Store'.

            Use the FAQ below to answer questions:
            {STORE_FAQ_TEXT}

            After answering a question, ask for lead details like:
            - Name
            - Email
            - Use case (what they want to build)

            Call `update_lead_profile` whenever the user shares details.
            Call `submit_lead_and_end` when they say goodbye or done.
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    userdata = Userdata(lead_profile=LeadProfile())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-natalie", style="Promo", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
