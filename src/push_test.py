import os, subprocess
from dotenv import load_dotenv

def auto_push_to_github(message="Test push from script"):
    load_dotenv()
    user = os.getenv("GITHUB_USER")
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")

    if not all([user, token, repo]):
        print("⚠️ Missing GitHub credentials in .env — skipping push.")
        return

    try:
        remote_url = f"https://{user}:{token}@github.com/{user}/{repo}.git"
        subprocess.run(["git", "config", "user.name", user], check=False)
        subprocess.run(["git", "config", "user.email", f"{user}@users.noreply.github.com"], check=False)
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=False)
        subprocess.run(["git", "add", "-A"], check=False)
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=False)
        print("🚀 Successfully pushed to GitHub.")
    except Exception as e:
        print(f"❌ GitHub push failed: {e}")

if __name__ == "__main__":
    auto_push_to_github("Test push: verifying PAT automation")
