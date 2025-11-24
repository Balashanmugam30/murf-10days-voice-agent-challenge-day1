import logging
import json
import os
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

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

CONTENT_FILE = "biology_content.json"

DEFAULT_CONTENT = [
    {
        "id": "dna",
        "title": "DNA",
        "summary": "DNA is the molecule that carries genetic instructions and is shaped like a double helix.",
        "sample_question": "What is the full form of DNA and what is its structure called?"
    },
    {
        "id": "cell",
        "title": "Cell",
        "summary": "The cell is the basic unit of life. Organisms may be unicellular or multicellular.",
        "sample_question": "What is the difference between prokaryotic and eukaryotic cells?"
    },
    {
        "id": "nucleus",
        "title": "Nucleus",
        "summary": "The nucleus contains the cell's DNA and controls growth and reproduction.",
        "sample_question": "Why is the nucleus called the control center of the cell?"
    },
    {
        "id": "cell_cycle",
        "title": "Cell Cycle",
        "summary": "The cell cycle includes Interphase and the Mitotic phase.",
        "sample_question": "Which phase does a cell spend most of its time in?"
    }
]

def load_content():
    path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
    if not os.path.exists(path):
        with open(path, "w", encoding='utf-8') as f:
            json.dump(DEFAULT_CONTENT, f, indent=4)
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

COURSE_CONTENT = load_content()

@dataclass
class TutorState:
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str):
        topic = next((t for t in COURSE_CONTENT if t["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None


@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="Topic ID (dna, cell, nucleus, cell_cycle)")]
):
    state = ctx.userdata.tutor_state
    if state.set_topic(topic_id.lower()):
        return f"Topic set to {state.current_topic_data['title']}. Ask if user wants Learn, Quiz, or Teach Back."
    available = ", ".join([t["id"] for t in COURSE_CONTENT])
    return f"Invalid topic. Available: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="learn, quiz, teach_back")]
):
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    session = ctx.userdata.agent_session

    if session:
        if mode == "learn":
            session.tts.update_options(voice="en-US-matthew", style="Promo")
            return "Learn mode enabled."
        elif mode == "quiz":
            session.tts.update_options(voice="en-US-alicia", style="Conversational")
            return "Quiz mode enabled."
        elif mode == "teach_back":
            session.tts.update_options(voice="en-US-ken", style="Promo")
            return "Teach-back mode enabled."
    return "Mode switched."

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="User's explanation during teach-back")]
):
    return "Evaluate the explanation. Score accuracy/clarity out of 10 and give corrections."


class TutorAgent(Agent):
    def __init__(self):
        topics = ", ".join([t["id"] for t in COURSE_CONTENT])
        super().__init__(
            instructions=f"""
            You are a Biology Tutor.

            Available topics: {topics}

            Modes:
            - Learn: explain the topic
            - Quiz: ask sample question
            - Teach_back: ask user to explain the topic

            You must:
            1. Ask user which topic they want to study.
            2. Call select_topic when they choose a topic.
            3. Call set_learning_mode when they say learn/quiz/teach.
            4. In teach_back, ask them to explain and call evaluate_teaching afterward.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    userdata = Userdata(tutor_state=TutorState())
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Promo",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata
    )

    userdata.agent_session = session

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
