import os
import json
import logging
from dataclasses import dataclass
from typing import Optional

import boto3
from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    ChatContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero


# ---------------------------------------------------------
# Environment + logging setup
# ---------------------------------------------------------
load_dotenv()
logger = logging.getLogger("lingua-trainer")

S3_STORAGE = "lingua-feedback-data"
s3_resource = boto3.client("s3")


# ---------------------------------------------------------
# Learner record
# ---------------------------------------------------------
@dataclass
class LearnerRecord:
    purpose: Optional[str] = None
    occupation: Optional[str] = None
    estimated_skill: Optional[str] = None
    practice_topic: Optional[str] = None
    highlights: Optional[str] = None
    improvements: Optional[str] = None


# ---------------------------------------------------------
# Agents
# ---------------------------------------------------------
class GreetingCoach(Agent):
    """Opens the session and collects learner’s purpose and occupation."""

    def __init__(self, ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions="""
            You are Lingua, an English conversation guide.
            Start with a cheerful hello.
            Ask the learner why they want to practice English (work, studies, travel, etc.).
            Next, ask about their current role or background.
            Remind them to keep replies in English.
            """,
            chat_ctx=ctx,
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def store_basics(
        self, ctx: RunContext[LearnerRecord], *, purpose: str, occupation: str
    ):
        ctx.userdata.purpose = purpose
        ctx.userdata.occupation = occupation
        logger.info(f"Basics collected: purpose={purpose}, occupation={occupation}")
        return SkillEvaluator(chat_ctx=self.chat_ctx), "Great! Let's quickly check your English fluency."


class SkillEvaluator(Agent):
    """Asks a few test questions to get an estimated skill rating."""

    def __init__(self, ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions="""
            Ask 2–3 short English questions to test fluency.
            Based on answers, assign a skill rating (e.g., 30%, 60%, 90%).
            Don’t explain what the number means.
            When ready, send them to practice scenarios.
            """,
            chat_ctx=ctx,
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def assign_rating(self, ctx: RunContext[LearnerRecord], *, rating: str):
        ctx.userdata.estimated_skill = rating
        return ScenarioTrainer(chat_ctx=self.chat_ctx), f"Your level is around {rating}. Let’s practice with a scenario."


class ScenarioTrainer(Agent):
    """Conducts roleplay practice."""

    def __init__(self, ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions="""
            Pick one practice scenario suitable for the learner’s purpose and level:
            - Interview roleplay
            - Academic admission talk
            - Office meeting
            - Travel conversation
            Explain the scenario and ask 2–3 related questions.
            Then move to feedback.
            """,
            chat_ctx=ctx,
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def mark_topic(self, ctx: RunContext[LearnerRecord], *, topic: str):
        ctx.userdata.practice_topic = topic
        return PerformanceReviewer(chat_ctx=self.chat_ctx), f"Nice job! You’ve finished the {topic} scenario."


class PerformanceReviewer(Agent):
    """Provides feedback + generates bilingual report."""

    def __init__(self, ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions="""
            Summarize learner’s performance:
            - Mention their estimated level
            - Highlight key strengths
            - Suggest improvement areas
            Provide the report in English and Arabic.
            Wrap up by wishing them success.
            """,
            chat_ctx=ctx,
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def wrap_up(
        self, ctx: RunContext[LearnerRecord], *, strengths: str, improvements: str
    ):
        self.session.interrupt()

        ctx.userdata.highlights = strengths
        ctx.userdata.improvements = improvements

        logger.info(
            f"[PerformanceReviewer] Feedback → strengths={strengths}, improvements={improvements}"
        )

        english_summary = {
            "Purpose": ctx.userdata.purpose or "N/A",
            "Occupation": ctx.userdata.occupation or "N/A",
            "Skill Estimate": ctx.userdata.estimated_skill or "N/A",
            "Scenario Practiced": ctx.userdata.practice_topic or "N/A",
            "Strengths": strengths,
            "Areas to Improve": improvements,
        }

        arabic_summary = {
            "الهدف": english_summary["Purpose"],
            "المهنة": english_summary["Occupation"],
            "المستوى المقدر": english_summary["Skill Estimate"],
            "المشهد التدريبي": english_summary["Scenario Practiced"],
            "نقاط القوة": strengths,
            "نقاط التحسين": improvements,
        }

        report = {"english": english_summary, "arabic": arabic_summary}

        job = get_job_context()
        room_id = job.room.name
        filename = f"lingua_feedback_{room_id}.json"

        try:
            s3_resource.put_object(
                Bucket=S3_STORAGE,
                Key=filename,
                Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"Report successfully uploaded → {filename}")
        except Exception as ex:
            logger.error(f"S3 upload error: {ex}")

        await self.session.generate_reply(
            instructions="Say goodbye warmly and encourage learner.",
            allow_interruptions=False,
        )

        await job.api.room.delete_room(api.DeleteRoomRequest(room=room_id))


# ---------------------------------------------------------
# Worker bootstrap
# ---------------------------------------------------------
def warmup(proc: JobProcess):
    proc.userdata["voice_detector"] = silero.VAD.load()


async def main_entry(ctx: JobContext):
    await ctx.connect()

    chat_state = ChatContext.empty()
    profile = LearnerRecord()

    session = AgentSession[LearnerRecord](
        vad=ctx.proc.userdata["voice_detector"],
        stt=openai.STT.with_azure(
            azure_deployment="gpt-4o-mini-transcribe",
            api_version="2024-02-15-preview",
            language="en",
        ),
        llm=openai.LLM.with_azure(
            azure_deployment="gpt-4o-mini", api_version="2024-02-15-preview"
        ),
        tts=openai.TTS.with_azure(
            azure_deployment="gpt-4o-mini-tts", api_version="2024-02-15-preview"
        ),
        userdata=profile,
    )

    usage_tracker = metrics.UsageCollector()

    @session.on("metrics_collected")
    def collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_tracker.collect(ev.metrics)

    async def report_usage():
        logger.info(f"Usage summary: {usage_tracker.get_summary()}")

    ctx.add_shutdown_callback(report_usage)

    await session.start(
        agent=GreetingCoach(chat_ctx=chat_state),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=main_entry, prewarm_fnc=warmup))
