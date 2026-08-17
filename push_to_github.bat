@echo off
set "PATH=%USERPROFILE%\.mingit\cmd;%PATH%"
echo ========================================================
echo Pushing CustomerPulse AI to GitHub Repository...
echo URL: https://github.com/SudhanshuSekharNaik/customerpulse-ai.git
echo ========================================================
git push -u origin main
echo ========================================================
pause
