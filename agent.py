from time import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

df = pd.read_csv("mental_health_monitoring_dataset.csv")
stress_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["Stress_Score"] = df["Stress_Level"].map(stress_map)

print("Stress_Score column created successfully")

print(df.columns.tolist())
print("Mental Health AI Agent Ready")

def plot_stress_distribution():
    df["Stress_Level"].hist()
    plt.title("Stress Level Distribution")
    plt.show()
while True:
    
    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break

    # Average Speed
    if "average speed" in question.lower():
        result = df["Average_Speed"].mean()
        print(f"\nAgent: The average speed is {result:.2f}")
        continue

    # Average Stress Level
    elif "stress level distribution" in question.lower():

        result = df["Stress_Level"].value_counts()

        print("\nStress Level Distribution:")
        print(result)

        continue

    # Most Common Driving Condition
    if "driving condition" in question.lower():
        result = df["Driving_Conditions"].mode()[0]
        print(f"\nAgent: The most common driving condition is {result}")
        continue

    # Most Common Mood
    if "mood" in question.lower():
        result = df["Mood"].mode()[0]
        print(f"\nAgent: The most common mood is {result}")
        continue
    
    elif "sleep and stress" in question.lower():
        corr = sleep_stress_correlation()
        print(f"Correlation: {corr:.2f}")
        continue
    elif "factors contribute most to stress" in question.lower():

        print("\nAnalyzing stress relationships...")

    numeric_cols = [
        "Heart_Rate",
        "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic",
        "Skin_Temperature",
        "Respiration_Rate",
        "Sleep_Duration",
        "Average_Speed",
        "Work_Hours",
        "Resilience_Factors"
    ]

    correlations = {}

    for col in numeric_cols:
        correlations[col] = abs(
            df[col].corr(df["Stress_Score"])
        )

    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop Factors Related to Stress:")

    for factor, corr in sorted_corr[:5]:
        print(f"{factor}: {corr:.3f}")

    continue
    
    # Fallback to Gemini
    prompt = f"""
    You are a Mental Health Data Analyst.

    Dataset Columns:
    {list(df.columns)}

    User Question:
    {question}

    Answer using the dataset columns.
    """
    import time

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        print("\nAgent:", response.text)
        break

    except Exception as e:
        print(f"Attempt {attempt+1} failed")

        if attempt < 2:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Gemini service unavailable.")

    print("\nAgent:", response.text)
    