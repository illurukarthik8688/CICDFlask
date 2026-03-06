from google import genai
import os
from self_healing.healing_actions import execute_action

def get_healing_action(stage, job, task):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "rebuild_pipeline"
        
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an AI DevOps engineer.

    A CI/CD pipeline failed.

    Stage: {stage}
    Job: {job}
    Task: {task}

    Suggest the best recovery action.
    Reply ONLY with the exact name of the action, no explanation, no markdown.

    Possible actions:
    - restart_service
    - rebuild_pipeline
    - clear_cache
    - ignore
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Clean the response to ensure it's just the action string
    return response.text.strip()


if __name__ == "__main__":

    action = get_healing_action("Deploy", "deploy_to_dev", "deploy")

    print("LLM Suggested Action:")
    print(action)
    
    # Actually execute the action
    execute_action(action)