import subprocess, os
from dotenv import load_dotenv

def auto_push_to_github(message="Manual sync"):
    load_dotenv()
    user = os.getenv("GITHUB_USER")
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")

    if not all([user, token, repo]):
        print("⚠️ Missing GitHub credentials in .env — skipping push.")
        return

    remote_url = f"https://{user}:{token}@github.com/{user}/{repo}.git"
    subprocess.run(["git", "config", "user.name", user], check=False)
    subprocess.run(["git", "config", "user.email", f"{user}@users.noreply.github.com"], check=False)
    subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=False)
    subprocess.run(["git", "add", "-A"], check=False)
    subprocess.run(["git", "commit", "-m", message], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=False)
    print("🚀 Repository successfully synced.")

auto_push_to_github("Full pipeline update (Git+DVC+MLflow)")
