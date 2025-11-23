import logging
import json
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# ---------------------- ASSISTANT BARISTA ----------------------

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly barista taking a coffee order.

Your job is to collect these fields:
1. drinkType
2. size
3. milk
4. extras
5. name

Ask only one question at a time.
When all fields are collected, summarize the order politely like a barista.
""",
        )


# ---------------------- SAVE ORDER TO JSON ----------------------

def save_order(order):
    filename = f"order_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(filename, "w") as f:
        json.dump(order, f, indent=4)
    print("Order saved to:", filename)
    return filename


# ---------------------- PREWARM ----------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# ---------------------- ENTRYPOINT ----------------------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # ------- Voice Agent Pipeline -------
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # ------- Metrics --------
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        print("Usage summary:", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # ------- ORDER TRACKING -------
    order = {
        "drinkType": None,
        "size": None,
        "milk": None,
        "extras": None,
        "name": None,
    }

    @session.on("transcription")
    def handle_text(text, **kwargs):
        t = text.lower()

        # Extract fields simply using keyword matching
        if any(d in t for d in ["latte", "cappuccino", "americano", "espresso"]):
            if "latte" in t:
                order["drinkType"] = "latte"
            if "cappuccino" in t:
                order["drinkType"] = "cappuccino"
            if "americano" in t:
                order["drinkType"] = "americano"
            if "espresso" in t:
                order["drinkType"] = "espresso"

        if "small" in t:
            order["size"] = "small"
        if "medium" in t:
            order["size"] = "medium"
        if "large" in t:
            order["size"] = "large"

        if "oat" in t:
            order["milk"] = "oat milk"
        if "soy" in t:
            order["milk"] = "soy milk"
        if "cow milk" in t or "regular milk" in t:
            order["milk"] = "regular milk"

        if "extra" in t or "whip" in t or "cream" in t:
            order["extras"] = "extras"

        if "my name is" in t:
            order["name"] = t.split("my name is")[-1].strip()

        # If all fields collected → save order
        if all(order.values()):
            save_order(order)

    # ------- Start session -------
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# ---------------------- START WORKER ----------------------

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
