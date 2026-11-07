# Push the final code and show passing CI (Windows)

1. On github.com create a **new public repository** (for example `finwise-credit-risk`). Leave "Add a README" **unticked**.
2. In Command Prompt, in the project folder (the one containing `serving` and `data_pipeline.py`):
```bat
git init -b main
git add -A
git commit -m "Final: serving API, React UI, tests, CI, model card"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```
   (If asked to log in, use the browser sign-in; if it asks for a password, use a Personal Access Token.)
3. Put your real repo name in the README badge, then push again:
```bat
powershell -Command "(Get-Content README.md) -replace 'YOUR_USERNAME/YOUR_REPO','YOUR_USERNAME/finwise-credit-risk' | Set-Content README.md"
git add README.md
git commit -m "Add CI badge"
git push
```
4. Open the repo's **Actions** tab. Two jobs run: *Backend tests + quality gates* and *Frontend build*. Wait for green ticks (about 2-4 minutes).
   If red: click the failed step, read the last lines, and send them to me.
5. Submit these four links: repo URL, the green Actions run URL, `MODEL_CARD.md` on GitHub (`.../blob/main/MODEL_CARD.md`), and the demo video.
