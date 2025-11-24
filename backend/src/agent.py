import logging
import json
import os
from datetime import datetime
from typing import Annotated
from dataclasses import dataclass, field, asdict

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
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ------------------ Check-in State ------------------

@dataclass
class CheckInState:
    mood: str | None = None
    energy: str | None = None
    objectives: list[str] = field(default_factory=list)
    advice_given: str | None = None

    def is_complete(self):
        return all([self.mood, self.energy, len(self.objectives) > 0])

    def to_dict(self):
        return asdict(self)


@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str
    session_start: datetime = field(default_factory=datetime.now)


# ------------------ Persistence ------------------

WELLNESS_LOG_FILE = "wellness_log.json"


def get_log_path():
    base = os.path.dirname(__file__)
    backend = os.path.abspath(os.path.join(base, ".."))
    return os.path.join(backend, WELLNESS_LOG_FILE)


def load_history():
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []


def save_checkin_entry(entry: CheckInState):
    path = get_log_path()
    history = load_history()
    record = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "energy": entry.energy,
        "objectives": entry.objectives,
        "summary": entry.advice_given,
    }
    history.append(record)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


# ------------------ Tools ------------------

@function_tool
async def record_mood_and_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="User mood")],
    energy: Annotated[str, Field(description="User energy level")],
):
    ctx.userdata.current_checkin.mood = mood
    ctx.userdata.current_checkin.energy = energy
    return "Mood and energy recorded."


@function_tool
async def record_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[list[str], Field(description="1-3 goals for the day")],
):
    ctx.userdata.current_checkin.objectives = objectives
    return "Objectives recorded."


@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    final_advice_summary: Annotated[str, Field(description="Closing summary")],
):
    state = ctx.userdata.current_checkin
    state.advice_given = final_advice_summary

    if not state.is_complete():
        return "I still need your mood, energy, and goals."

    save_checkin_entry(state)

    recap = (
        f"Today you are feeling {state.mood} with {state.energy} energy. "
        f"Your goals are: {', '.join(state.objectives)}. "
        f"Summary: {final_advice_summary}. "
        f"Your check-in has been saved."
    )
    return recap


# ------------------ Agent ------------------

class WellnessAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
You are a daily wellness companion.
Use the history context below to personalize the conversation.

Previous history:
{history_context}

Steps:
1. Ask for mood + energy.
2. Ask for 1-3 goals.
3. Provide simple non-medical advice.
4. Call complete_checkin at the end.

Do not give medical advice or diagnoses.
            """,
            tools=[
                record_mood_and_energy,
                record_objectives,
                complete_checkin,
            ],
        )


# ------------------ Entrypoint ------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    history = load_history()
    if history:
        last = history[-1]
        history_context = (
            f"Last check-in: {last.get('timestamp')}. "
            f"Mood: {last.get('mood')}. Energy: {last.get('energy')}. "
            f"Goals: {', '.join(last.get('objectives', []))}."
        )
    else:
        history_context = "No previous check-ins."

    userdata = Userdata(
        current_checkin=CheckInState(),
        history_summary=history_context
    )

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie",
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=WellnessAgent(history_context=history_context),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
