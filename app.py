import logging
import json
import os
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

load_dotenv()
logger = logging.getLogger("english-coach")

# AWS S3 Configuration
S3_BUCKET_NAME = "englishly-reports"
S3_CLIENT = boto3.client("s3")

@dataclass
class LearnerProfile:
    goal: Optional[str] = None
    background: Optional[str] = None
    level: Optional[str] = None
    scenario: Optional[str] = None
    strengths: Optional[str] = None
    areas_to_improve: Optional[str] = None


class AssessmentIntroAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions="""
            Your name is Englishly.
            You are an English-speaking AI coach. 
            Start with a friendly introduction.
            Ask the user about their English learning goal (e.g., job, travel, study).
            Then ask about their background (e.g., student, doctor, engineer).
            After collecting both, hand off to the assessment agent.
            Only speak English. If the user uses another language, gently remind them to use English.
            """,
            chat_ctx=chat_ctx
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def set_user_profile(
        self,
        context: RunContext[LearnerProfile],
        goal: str,
        background: str,
    ):
        context.userdata.goal = goal
        context.userdata.background = background
        logger.info(f"User profile captured: Goal={goal}, Background={background}")
        return ProficiencyAssessmentAgent(chat_ctx=self.chat_ctx), "Thanks! Let's assess your English level now."


class ProficiencyAssessmentAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions="""
            Ask the user a few simple English questions to evaluate their speaking proficiency.
            Based on their answers, estimate their proficiency level (out of 100%).
            Do not explain the levels.
            When ready, hand off to the ScenarioAgent with the estimated level.
            """,
            chat_ctx=chat_ctx
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def set_level(
        self,
        context: RunContext[LearnerProfile],
        level: str,
    ):
        context.userdata.level = level
        return ScenarioAgent(chat_ctx=self.chat_ctx), f"Got it! Your level is estimated at {level}. Let's move on."


class ScenarioAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions="""
            Based on the user's goal and level, pick a suitable practice scenario:
            - Job interview based on their background
            - University admission
            - Business meeting
            - Travel conversation

            Announce the scenario and start a short roleplay session.
            Ask 2-3 scenario-based questions.
            Then, hand off to FeedbackAgent.
            """,
            chat_ctx=chat_ctx
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def set_scenario(
        self,
        context: RunContext[LearnerProfile],
        scenario: str,
    ):
        context.userdata.scenario = scenario
        return FeedbackAgent(chat_ctx=self.chat_ctx), f"Great! You've completed the {scenario} practice."


class FeedbackAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions="""
            Provide feedback based on the user's overall performance:
            - Proficiency score
            - Strengths
            - Areas to improve

            Present the report in English and Arabic.
            After giving the report, say goodbye and end the session.
            """,
            chat_ctx=chat_ctx
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def end_session(
        self, 
        context: RunContext[LearnerProfile],
        strengths: str,
        areas_to_improve: str):

        self.session.interrupt()

        # Store feedback in userdata
        context.userdata.strengths = strengths
        context.userdata.areas_to_improve = areas_to_improve
        logger.info(f"[FeedbackAgent] Strengths: {strengths}, Areas to Improve: {areas_to_improve}")

        goal = context.userdata.goal or "N/A"
        background = context.userdata.background or "N/A"
        level = context.userdata.level or "N/A"
        scenario = context.userdata.scenario or "N/A"
        

        english = {
            "Goal": goal,
            "Background": background,
            "Estimated Level": level,
            "Scenario Practiced": scenario,
            "Strengths": strengths,
            "Areas to Improve": areas_to_improve,
        }

        arabic = {
            "الهدف": goal,
            "الخلفية": background,
            "المستوى المتوقع": level,
            "السيناريو": scenario,
            "نقاط القوة": strengths,
            "نقاط التحسين": areas_to_improve
        }

        report = {"english": english, "arabic": arabic}

        # Upload JSON report to S3
        job_ctx = get_job_context()
        room_name = job_ctx.room.name
        filename = f"english_feedback_report_{room_name}.json"
        
        try:
            S3_CLIENT.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=filename,
                Body=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json"
            )
            logger.info(f"Report uploaded to S3: {filename}")
        except Exception as e:
            logger.error(f"Failed to upload report to S3: {e}")

        await self.session.generate_reply(
            instructions="Say goodbye and wish the learner good luck.",
            allow_interruptions=False
        )

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Initialize empty ChatContext
    chat_ctx = ChatContext.empty()

    session = AgentSession[LearnerProfile](
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT.with_azure(
            azure_deployment="gpt-4o-mini-transcribe",
            api_version="2024-02-15-preview",
            language="en"
        ),
        llm=openai.LLM.with_azure(
            azure_deployment="gpt-4o-mini",
            api_version="2024-02-15-preview"
        ),
        tts=openai.TTS.with_azure(
            azure_deployment="gpt-4o-mini-tts",
            api_version="2024-02-15-preview"
        ),
        userdata=LearnerProfile(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=AssessmentIntroAgent(chat_ctx=chat_ctx),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
