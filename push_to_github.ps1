$env:PATH = "$env:USERPROFILE\.mingit\cmd;$env:PATH"
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Pushing CustomerPulse AI to GitHub Repository..." -ForegroundColor Green
Write-Host "URL: https://github.com/SudhanshuSekharNaik/customerpulse-ai.git" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
git push -u origin main
