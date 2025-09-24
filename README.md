# English-Speaking-Assessment-Agent

# Lingua Trainer 🎙️

Lingua Trainer is an **AI-powered English conversation coach** built using **LiveKit Agents**, **Azure OpenAI**, and **AWS S3**.  
It guides learners through a structured flow:

1. **Greeting & Onboarding** – Learner introduces their goal and background.  
2. **Skill Check** – The agent asks simple questions to estimate English fluency.  
3. **Scenario Practice** – Roleplay in practical contexts like interviews, travel, or study.  
4. **Feedback Report** – Learner receives bilingual (English + Arabic) feedback with strengths and improvement areas.  

The session ends with a **personalized performance report** saved to S3.

---

## 🔑 Features
- **Multi-agent flow** (Greeting → Level Check → Scenario → Feedback).  
- **Real-time STT, LLM, and TTS** via Azure OpenAI.  
- **Voice Activity Detection** using Silero VAD.  
- **Usage metrics tracking** with LiveKit.  
- **Automatic report export** in JSON format to AWS S3 (English + Arabic).  

---

## ⚙️ Tech Stack
- [LiveKit Agents](https://github.com/livekit/agents) – conversational pipeline.  
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) – STT, LLM, TTS.  
- [Silero VAD](https://github.com/snakers4/silero-vad) – voice activity detection.  
- [AWS S3](https://aws.amazon.com/s3/) – report storage.  
- [Python 3.10+](https://www.python.org/)  

---

## 📂 Project Structure
.
├── lingua_trainer.py # Main application code
├── README.md # Documentation
├── requirements.txt # Python dependencies
└── .env # Environment variables (not committed)

yaml
Copy code

---

## 🔧 Setup Instructions

### 1. Clone repo & install dependencies
```bash
git clone https://github.com/your-username/lingua-trainer.git
cd lingua-trainer
pip install -r requirements.txt
2. Configure environment
Create a .env file with:

env
Copy code
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=your_region

AZURE_OPENAI_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
3. Run the app
bash
Copy code
python lingua_trainer.py
📊 Reports
At the end of each session, a JSON report is uploaded to S3:

json
Copy code
{
  "english": {
    "Purpose": "Job interview",
    "Occupation": "Engineer",
    "Skill Estimate": "70%",
    "Scenario Practiced": "Interview roleplay",
    "Strengths": "Good grammar",
    "Areas to Improve": "Pronunciation"
  },
  "arabic": {
    "الهدف": "Job interview",
    "المهنة": "Engineer",
    "المستوى المقدر": "70%",
    "المشهد التدريبي": "Interview roleplay",
    "نقاط القوة": "Good grammar",
    "نقاط التحسين": "Pronunciation"
  }
}
🧩 Workflow Overview
GreetingCoach → Collects purpose & background.

SkillEvaluator → Tests English fluency.

ScenarioTrainer → Runs practice conversation.

PerformanceReviewer → Provides feedback & uploads report.

📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and extend it.

🚀 Future Enhancements
Add more scenarios (customer support, negotiations, etc.).

Enable PDF report generation in addition to JSON.

Add multilingual coaching modes (Spanish, French).
